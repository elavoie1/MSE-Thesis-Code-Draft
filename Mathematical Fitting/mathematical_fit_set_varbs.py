# -*- coding: utf-8 -*-
"""
visualize function and compute chi-square for a single set of variables

@author: emili
"""
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import numpy as np
from matplotlib.ticker import LogLocator
import matplotlib as mpl
from matplotlib import rc
import paleopy as paleopy
import swordfish as sf
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, quad
from scipy.special import erf
from scipy.ndimage.filters import gaussian_filter1d
import configparser
from tqdm import tqdm
from WIMpy import DMUtils as DMU
import os 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.interpolate import InterpolatedUnivariateSpline
from datetime import datetime
from scipy.stats import chi2
from scipy.special import erfc,erf
startTime = datetime.now()



# import all measured track lengths in halite 
Height_File_Name = 'profile csv/halite - all detected pits.csv'

# open track length csv and convert to df
length_csv = read_csv(Height_File_Name)
pits = pd.DataFrame(length_csv)

inplane = pd.DataFrame()

# filter through aspect ratio
for i in range(len(pits.iloc[:,3])):
    if pits.iloc[i,3] < 0.05:
        inplane = inplane.append(pits.iloc[i,:])

# convert to nm then convert using etching rate of 14.47 nm/min for 10 min
inplane = inplane.iloc[:,0] * 1000 - 144.7

inplane_filtered_max = []

# only use < 400 nm lengths because above that counts are ~ zero
for i in range(len(inplane)):
    if inplane.iloc[i] < 400:
        inplane_filtered_max.append(inplane.iloc[i])

inplane_filtered = []

# remove neg vals and noise
inplane_filtered = [ele for ele in inplane_filtered_max if ele > 10]

num_bins = 35

plt.figure(figsize=(10,4))
#plot histogram inplane track diameter
exp_count, bins, _ = hist = plt.hist(inplane_filtered, num_bins, color= 'lightsalmon', 
                             edgecolor='black', linewidth=0.5)

#plt.plot([45,45], [0, max(exp_count)], color = 'black', 
        # linestyle='-.', linewidth = 0.5)

plt.title('Converted Track Lengths in Halite')
plt.xlabel('Track Length (nm)')
plt.ylabel('Frequency')

# plt.ylim(0, 75)
# plt.xlim(10,1000) # rid of noise near 0


# variables to help convert mass - vol - area
scanned_area = 14144.265 * 1e-8 #cm2
surface_density = 4.29e6 #tracks/cm2
vol_density = 8.89e9 #tracks/cm3
mass_wanted = 0.000001 #kg
mass = 0.00216 # kg/cm3
mass_frac = mass_wanted/mass
density_per_mass = vol_density * mass_frac
num_tracks = len(inplane_filtered)

constant = density_per_mass/num_tracks


lengths_per_vol = inplane_filtered #* round(constant) #NOT converting to volume...



#______________________________________________________________________________
## code to determine Th and neutron backgrounds - from Baum, et. al
class Mineral:
    def __init__(self, mineral):
        
        #mineral in config
        
        self.name = mineral
        
        
        config = configparser.ConfigParser()
        config.read("Data/MineralList.txt")
        data = config[mineral]
    
        nuclist = data["nuclei"].split(",")
        self.nuclei = [x.strip(' ') for x in nuclist]
        
        self.N_nuclei = len(self.nuclei)
        
        self.stoich = np.asarray(data["stoich"].split(","), dtype=float)
        
        #self.abun = np.asarray(data["abundances"].split(","), dtype=float)
        self.N_p = np.asarray(data["N_p"].split(","), dtype=float)
        self.N_n = np.asarray(data["N_n"].split(","), dtype=float)
        
        #Check that there's the right number of everything
        if (len(self.stoich) != self.N_nuclei):
            raise ValueError("Number of stoich. ratio entries doesn't match number of nuclei for mineral <" + self.name + ">...")
        if (len(self.N_p) != self.N_nuclei):
            raise ValueError("Number of N_p entries doesn't match number of nuclei for mineral <" + self.name + ">...")
        if (len(self.N_p) != self.N_nuclei):
            raise ValueError("Number of N_n entries doesn't match number of nuclei for mineral <" + self.name + ">...")
        
        self.shortname = data["shortname"]
        self.U_frac = float(data["U_frac"]) #Uranium fraction by weight
        
        #Calculate some derived stuff
        self.molarmass = np.sum(self.stoich*(self.N_p + self.N_n))
        self.abun = self.stoich*(self.N_p + self.N_n)/self.molarmass

        
        self.dEdx_interp = []
        self.Etox_interp = []
        self.xtoE_interp = []
        
        self.Etox_interp_Th = None
        
        if (self.shortname == "Zab"):
            self.loadSRIMdata(modifier="CC2338")
        elif (self.shortname == "Syl"):
            self.loadSRIMdata(modifier="CC1")
        else:
            self.loadSRIMdata()
        
        self.NeutronBkg_interp = []
        
        self.loadNeutronBkg()
        
        #self.loadFissionBkg()
        
        #Do we need these cumbersome dictionaries...?
        self.dEdx_nuclei = dict(zip(self.nuclei, self.dEdx_interp))
        self.Etox_nuclei = dict(zip(self.nuclei, self.Etox_interp))
        self.xtoE_nuclei = dict(zip(self.nuclei, self.xtoE_interp))
        self.ratio_nuclei = dict(zip(self.nuclei, self.abun))

    #--------------------------------   
    def showProperties(self):
        print("Mineral name:", self.name)
        print("    N_nuclei:", self.N_nuclei)
        print("    Molar mass:", self.molarmass, " g/mol")
        print("    nucleus \t*\t abun.  *\t (N_p, N_n)")
        print(" **************************************************")
        for i in range(self.N_nuclei):
            print("    " + self.nuclei[i] + "\t\t*\t" + str(self.abun[i]) + "\t*\t(" +str(self.N_p[i]) 
                  + ", " + str(self.N_n[i]) + ")")
         
    #--------------------------------   
    def loadSRIMdata(self, modifier=None):
        #The modifier can be used to identify a particular version of the SRIM
        #track length files (e.g. modifier="CC2338")
        
        SRIMfolder = "Data/dRdESRIM/"

        self.Etox_interp = []
        self.xtoE_interp = []
        self.dEdx_interp = []
    
        for nuc in self.nuclei:
            #Construct the SRIM output filename
            infile = SRIMfolder + nuc + "-" + self.shortname
            if not(modifier == None):
                infile += "-" + modifier
            infile += ".txt"
        
            E, dEedx, dEndx = np.loadtxt(infile, usecols=(0,1,2), unpack=True)
            dEdx = dEedx + dEndx    #Add electronic stopping to nuclear stopping
            dEdx *= 1.e-3           # Convert keV/micro_m to keV/nm
            x = cumtrapz(1./dEdx,x=E, initial=0)    #Calculate integrated track lengths
        
            #Generate interpolation function (x(E), E(x), dEdx(x))
            self.Etox_interp.append(interp1d(E, x, bounds_error=False, fill_value='extrapolate'))
            self.xtoE_interp.append(interp1d(x, E, bounds_error=False, fill_value='extrapolate'))
            self.dEdx_interp.append(interp1d(x, dEdx, bounds_error=False, fill_value='extrapolate'))    
    
        #Load in the Thorium track lengths...
        #Construct the SRIM output filename
        infile = SRIMfolder + "Th-" + self.shortname
        if not(modifier == None):
            infile += "-" + modifier
        infile += ".txt"
        
        E, dEedx, dEndx = np.loadtxt(infile, usecols=(0,1,2), unpack=True)
        dEdx = dEedx + dEndx    #Add electronic stopping to nuclear stopping
        dEdx *= 1.e-3           # Convert keV/micro_m to keV/nm
        x = cumtrapz(1./dEdx,x=E, initial=0)    #Calculate integrated track lengths
        self.Etox_interp_Th = interp1d(E, x, bounds_error=False, fill_value='extrapolate')
    
    

    
    #--------------------------------
    def showSRIM(self):
        print("Plotting SRIM data for " + self.name + ":")
        x_list = np.logspace(0,4,100)

        fig, axarr = plt.subplots(figsize=(10,4),nrows=1, ncols=2)
        ax1, ax2 = axarr
        for i in range(self.N_nuclei):
            ax1.loglog(x_list, self.dEdx_interp[i](x_list),label=self.nuclei[i])
        ax1.set_ylabel("dE/dx [keV/nm]")
        ax1.set_xlabel("x [nm]")
        ax1.legend()
                
        E_list = np.logspace(-3, 3, 500) # keV    
        
        for i in range(self.N_nuclei):
            ax2.loglog(E_list, self.Etox_interp[i](E_list),label=self.nuclei[i])
        ax2.set_ylabel("x [nm]")
        ax2.set_xlabel("E [keV]")
        ax2.legend()
        
        plt.savefig(self.name + 'SRIM.pdf', bbox_inches='tight')
        plt.show()
        
        
    #--------------------------------
    def dRdx(self, x_bins, sigma, m, gaussian=False):
        x_width = np.diff(x_bins)
        x = x_bins[:-1] + x_width/2
        #Returns in events/kg/Myr/nm

        
        dRdx = np.zeros_like(x)
        for i, nuc in enumerate(self.nuclei):
            # Ignore recoiling hydrogen nuclei
            if (nuc != "H"):
                Etemp = self.xtoE_nuclei[nuc](x)
                dRdx_nuc = (DMU.dRdE_standard(Etemp, self.N_p[i], self.N_n[i], m, sigma, \
                                        vlag=248.0, sigmav=166.0, vesc=550.0)*self.dEdx_nuclei[nuc](x))
                dRdx += self.ratio_nuclei[nuc]*dRdx_nuc
            
        if gaussian:
            dRdx = gaussian_filter1d(dRdx,1)+1e-20
        return dRdx*1e6*365

    def dRdx_generic_vel(self, x_bins, sigma, m, eta, gaussian=False):
        x_width = np.diff(x_bins)
        x = x_bins[:-1] + x_width/2
        #Returns in events/kg/Myr/nm

        
        dRdx = np.zeros_like(x)
        for i, nuc in enumerate(self.nuclei):
            # Ignore recoiling hydrogen nuclei
            if (nuc != "H"):
                Etemp = self.xtoE_nuclei[nuc](x)
                dRdx_nuc = (DMU.dRdE_generic(Etemp, self.N_p[i], self.N_n[i], m, sigma, eta)*self.dEdx_nuclei[nuc](x))
                dRdx += self.ratio_nuclei[nuc]*dRdx_nuc
            
        if gaussian:
            dRdx = gaussian_filter1d(dRdx,1)+1e-20
        return dRdx*1e6*365
    
    #--------------------------------
    def dRdx_nu(self,x_bins, components=False, gaussian=False):
        x_width = np.diff(x_bins)
        x = x_bins[:-1] + x_width/2
        #Returns in events/kg/Myr/nm
        nu_list = ['DSNB', 'atm', 'hep', '8B', '15O', '17F', '13N', 'pep','pp','7Be-384','7Be-861']
    
        E_list = np.logspace(-3, 3, 5000) # keV
    
        if components:
            dRdx = []
            for j, nu_source in enumerate(nu_list):
                dRdx_temp = np.zeros_like(x)
                for i, nuc in enumerate(self.nuclei):
                    if (nuc != "H"):
                        xtemp = self.Etox_nuclei[nuc](E_list)
                        dRdx_nuc = (np.vectorize(DMU.dRdE_CEvNS)(E_list, self.N_p[i], self.N_n[i], flux_name=nu_source)
                                                            *self.dEdx_nuclei[nuc](xtemp))
                        temp_interp = interp1d(xtemp, dRdx_nuc, fill_value='extrapolate')
                        dRdx_temp += self.ratio_nuclei[nuc]*temp_interp(x)
                    
                if gaussian:
                    dRdx.append(gaussian_filter1d(dRdx_temp*1e6*365,1)+1e-20)
                else:
                    dRdx.append(dRdx_temp*1e6*365+1e-20)
        else:
            dRdx = np.zeros_like(x)
            for i, nuc in enumerate(self.nuclei):
                if (nuc != "H"):
                    xtemp = self.Etox_nuclei[nuc](E_list)
                    dRdx_nuc = (np.vectorize(DMU.dRdE_CEvNS)(E_list, self.N_p[i], self.N_n[i], flux_name='all')
                                                        *self.dEdx_nuclei[nuc](xtemp))
                    temp_interp = interp1d(xtemp, dRdx_nuc, fill_value='extrapolate')
                    dRdx += self.ratio_nuclei[nuc]*temp_interp(x)*1e6*365
            if gaussian:
                dRdx = gaussian_filter1d(dRdx*1e6*365,1)+1e-20
                
        return dRdx
    
    def xT_Thorium(self):
        E_Thorium = 72. #keV
        return self.Etox_interp_Th(E_Thorium)
    
    def norm_Thorium(self, T):
        #T is in years. Returns events/kg/Myr
        T_half_238 = 4.468e9
        T_half_234 = 2.455e5
        
        lam_238 = np.log(2)/T_half_238
        lam_234 = np.log(2)/T_half_234
        
        #Avocado's constant
        N_A = 6.022140857e23
        

        n238_permass = self.U_frac*N_A*1e3/238.0 #Number of U238 atoms *per kg*
        Nalpha = n238_permass*(lam_238/(lam_234 - lam_238))*(np.exp(-lam_238*T) - np.exp(-lam_234*T))
        return Nalpha/(T*1e-6)
        
    def loadNeutronBkg(self):
        
        fname = "Data/" + self.name + "_ninduced_wan.dat"

        #Read in the column headings so you know which element is which
        f = open(fname)
        head = f.readlines()[1]
        columns = head.split(",")
        columns = [c.strip() for c in columns]
        ncols = len(columns)
        f.close()
        
        data = np.loadtxt(fname)
        E_list = data[:,0]
        
        self.NeutronBkg_interp = []
        
        for i, nuc in enumerate(self.nuclei):
            dRdE_list = 0.0*E_list
            #How many characters is the length of the element name you're looking for
            nchars = len(nuc)
            for j in range(ncols):
                #Check if this is the correct element
                if (columns[j][0:nchars] == nuc):
                    dRdE_list += data[:,j]
            
            (self.NeutronBkg_interp).append(interp1d(E_list, dRdE_list,bounds_error=False,fill_value=0.0))
            
    def dRdx_neutrons(self, x_bins):
        x_width = np.diff(x_bins)
        x = x_bins[:-1] + x_width/2
        #Returns in events/kg/Myr/nm
        
        
        dRdx = np.zeros_like(x)
        for i, nuc in enumerate(self.nuclei):
            if (nuc != "H"):
                E_list = self.xtoE_nuclei[nuc](x) 
                dRdx_nuc = self.NeutronBkg_interp[i](E_list)*self.dEdx_nuclei[nuc](x)
                dRdx += dRdx_nuc #Isotope fractions are already included in the tabulated neutron spectra
                
        return dRdx*self.U_frac/0.1e-9 #Tables were generated for a Uranium fraction of 0.1 ppb

def plotSpectrum(mineral):
    
    x_bins = np.linspace(1,400,num=len(inplane_filtered))

    plt.figure(figsize=(7,5))

    plt.title(mineral.name + r" Track Length Spectrum")

    # Signal spectrum for a given DM cross section and mass
    # These functions SHOULD work if we have exact same file formatting as authors
    # need to re-do and paleopy functions
    #plt.loglog(x_bins[:-1], mineral.dRdx(x_bins, 1e-45, 5), label=r'$5\,\mathrm{GeV}$')
    #plt.loglog(x_bins[:-1], mineral.dRdx(x_bins, 1e-45, 50), label=r'$50\,\mathrm{GeV}$')
    #plt.loglog(x_bins[:-1], mineral.dRdx(x_bins, 1e-45, 500), label=r'$500\,\mathrm{GeV}$')

    # Background spectrum for neutrinos
    #plt.loglog(x_bins[:-1], mineral.dRdx_nu(x_bins), linestyle='--',label=r'Neutrinos')

    # Neutron-induced backgrounds
    plt.plot(x_bins[:-1], mineral.dRdx_neutrons(x_bins), linestyle=':', label='Neutrons')

    
    
    #Plot the line from Thorium
    #x_Th = mineral.xT_Thorium()
    #plt.loglog([x_Th, x_Th], [1e-10, mineral.norm_Thorium(T=1e6)], linestyle='-.',label=r'$1\alpha$-Thorium')

    plt.legend(fontsize=12)

    #ax = plt.gca()
    #plt.text(0.05, 0.9, r"$\sigma_p^{\mathrm{SI}}=10^{-45}\,\mathrm{cm}^2$",fontsize=16.0, transform=ax.transAxes)
    
    plt.ylabel("dR/dx [1/nm/Myr]")
    plt.xlabel("x [nm]")
    #plt.ylim(1e-4,1e11)
    plt.xlim(1,400)

    #plt.savefig("plots/" + mineral.name + "_spectra.pdf",bbox_inches="tight")
    #print([x_Th, x_Th], [1e-10, mineral.norm_Thorium(T=1e6)])
    plt.show()

# use mineral data for halite
Hal = Mineral("Halite")
plotSpectrum(Hal)


# range over 0-400 nm
x_bins = np.linspace(10,400,num=len(inplane_filtered))
x = x_bins

# establish theoretical math model
def theoretical_func(xdata, mineral, A, B, sigma, x0, c, d, beta):
    ## sigmoid for efficiency function
    eff_func = 1 / (1 + np.exp(-1 * c * (xdata[:-1] - (d))))
    #(np.exp((x[:-1]- shift1)*c1)/(np.exp((x[:-1]- shift2)*c2) + 1))

    
    ## gaussian for thorium bkg
    gauss = B * np.exp((-1/2) * ((xdata[:-1] - x0) / sigma)**2)
    
    ## import neutron bkg
    neutron_bkg = 1e6 * np.exp(-1 * beta * (xdata[:-1]) )
    #mineral.dRdx_neutrons(xdata) # B = neutron constant/gauss constant
    #
    #
    
    ## combine components
    total_func = eff_func * A * (gauss + neutron_bkg) # A = overall constant/ gauss constant
    
    return total_func

# establish dataframe that will store all data
#chi_test_vals = pd.DataFrame(columns = ['Chi-Squared Test Statistic (overall)', 'A', 'B', 'x_0', 'sigma', 'c', 'd'])

## x = x_bins!! (0-400 nm)

A = 70000
sigma = 3
B = 3e7
x0 = 28
c = 0.05
d = 120
beta = 0.018 #0.02325581 = 1/43

# 43.012952 (chi) 618000.0  2550000.0  28.0    2.5  0.055  200.0  0.0449

### chi of 43.3 !!
# =============================================================================
# A = 622000
# sigma = 2.5
# B = 2.55e6
# x0 = 29
# c = 0.055
# d = 200
# beta = 0.0449 #0.02325581 = 1/43
# =============================================================================


#plt.figure()
#plt.plot(x_bins, B * np.exp((-1/2) * ((x_bins - x0) / sigma)**2))

#plt.figure()
#plt.plot(x, 1 / (1 + np.exp(-1 * c * x + (d))))


## visualize efficiency function
#plt.figure()
#c = 0.025
#plt.plot(x, 1 / (1 + np.exp(-1 * c * x)))

## integrate over theoretical function

#import scipy.integrate as integrate

# range of tracks lengths to integrate over
# x_batch = x[0:2] # or something like that

# Signal spectrum for a given DM cross section and mass
int_func = theoretical_func(x, Hal, A, B, sigma, x0, c, d, beta)

# fit integral function
f = InterpolatedUnivariateSpline(x[:-1], int_func)  # cubic spline default

plt.figure()
# plot both functions to verify fit
#plt.plot(x[:-1],f(x[:-1]), label='spline')


plt.plot([45,45], [0, 2e8], color = 'black', 
         linestyle='-.', linewidth = 0.5)

#plt.ylim(0, 1.5e8)

#plt.yticks([])
plt.title('Theoretical Th-alpha Recoil Tracks in Halite')
plt.xlabel('Track Length (nm)')
plt.ylabel('Measured Tracks (per kg)')


plt.plot(x[:-1], int_func, label='theoretical model', color='black')
#plt.legend()

# estimated exposure time
mineral_age_est = 1

## integral of theoretical function
# f.integral(x_batch[0], x_batch[1]) * mass_wanted * mineral_age_est # mass # Myr

# create empty dataframe to append values to after iteration
integral_vals = pd.DataFrame(columns = ['bin', 'bin_start', 'bin_stop', 'count (# of tracks per Myr)'])

#width of each bin in measured data plot
bin_width = (max(lengths_per_vol) - min(lengths_per_vol)) / len(bins)

# iterate over integrals and plot a histogram over SAME bins!
for i in range(len(bins)-1):
    x_batch = (bins[i], bins[i+1])
    #(bins[i] - (bin_width/2)), (bins[i] + (bin_width/2)) # is not technically grabbing full range!
    
    count = f.integral(x_batch[0], x_batch[1]) * mass * mineral_age_est # mass (1 cm^3 of halite) # Myr estimate
    
    count_per_area = (count **(2./3)) * scanned_area # convert theoretical track length spectra to tracks/area
    
    integral_vals.loc[len(integral_vals.index)] = [bins[i], x_batch[0], x_batch[1], count_per_area] 
    # tabulate integral vals (counts per bin)

## plot histogram of data:
# hh1 = plt.hist(counted_data.keys(), weights=counted_data.values(), bins=range(10), rwidth=.95, label="counted_data")
plt.figure()
plt.ylabel('Frequency')
plt.xlabel('Track Length (nm)')
plt.title('Theoretical Th-alpha Recoil Track Counts in Halite')
plt.plot([45,45], [0, max(exp_count)], color = 'black', 
         linestyle='-.', linewidth = 0.5)
n1, bins1, _ = plt.hist(integral_vals.iloc[:,0], weights = integral_vals.iloc[:,3], bins=bins, 
                        facecolor='plum', edgecolor='black', linewidth=0.5)



# plotting a specific scenario w error bars

lower_error =  []
upper_error =  []
list_cs = []

plt.figure(figsize=(10,4))
#plot histogram inplane track diameter
y, binEdges, _ = plt.hist(inplane_filtered, num_bins, color= 'blue', 
                             edgecolor='black', linewidth=0.75, label='experimental', alpha=0.6)


bincenters = 0.5*(binEdges[1:]+binEdges[:-1])

for p in range(len(exp_count)):
    
    num_sigma = 1
    
    alpha=erfc(num_sigma/np.sqrt(2))
        
    lower = 0.5 * chi2.ppf(1 - alpha/2, 2*(exp_count[p] + 1)) - exp_count[p]
        
    upper = exp_count[p] - 0.5 * chi2.ppf(alpha/2, 2 * exp_count[p])
    
    lower_error.append(lower)
    upper_error.append(upper)
    
    if integral_vals.iloc[p,3] > exp_count[p]:
        uncertainty = 0.5 * chi2.ppf(1 - alpha/2, 2*(exp_count[p] + 1)) - exp_count[p]
        
    if integral_vals.iloc[p,3] < exp_count[p]:
        uncertainty = exp_count[p] - 0.5 * chi2.ppf(alpha/2, 2 * exp_count[p])
    
    
    chi_square = (exp_count[p] - integral_vals.iloc[p,3])**2 / (uncertainty**2)
    list_cs.append(chi_square)
        
    
# error bar values w/ different -/+ errors that
# also vary with the x-position
asymmetric_error = np.array(list(zip(lower_error, upper_error))).T




menStd     = list_cs

width      = 0.05

#plt.bar(bincenters, y, width=width, color='r', yerr=asymmetric_error)

plt.errorbar(bincenters, y, yerr=asymmetric_error, label='error', ecolor='black', fmt='none', capsize=2, capthick=2)
plt.ylim(0,55)
plt.xlim(0,400)


plt.plot([45,45], [0, max(exp_count)], color = 'black', 
         linestyle='-.', linewidth = 0.5)

plt.title('Experimental Data v. Theoretical Model')
plt.xlabel('Track Length (nm)')
plt.ylabel('Count')

ax = plt.gca()
plt.text(0.84, 0.65, r"$\chi^{2} = 43.013$",fontsize=12.0, transform=ax.transAxes)

#plt.plot(x_bins[:-1], theoretical_func(x, Hal, A, B, sigma, x0, c, d), label='theory')
n1, bins1, _ = plt.hist(integral_vals.iloc[:,0], weights = integral_vals.iloc[:,3], bins=bins, edgecolor='black', linewidth=0.75,
                        color='crimson', label='theory', alpha=0.4)

plt.legend()





chi_squared_sum = sum(list_cs)
print("chi-square: " + str(chi_squared_sum))















# 
# # ----------------------------------------------------------------------------
# ## now compare to counts in each bin of measured lengths
# ## chi-square goodness of fit test
# 
# 
# # in every BIN of every INTEGRAL, do the chi-squared test between experimental data
# # in the samebin, and theoretical - following (E - T)^2/Uncertainy^2
# # uncertainty is the poissan's distributions
# 
# # need to create 5 for loops (one per variable)
# # also need to store the data - and make sure there is not a memory error
# # once data is stored THEN create the separate surface plots and such
# # put titles on each plot
# # for loop before integral - and everything should be included in the loop
# # ten values per varb to start
# # but need to know approx where to start the ranges.. mess around with varbs first to see


# =============================================================================
# # =============================================================================
# # range over all variables 
# A_range = [5000,10000] # 2 
# sigma_range = [2,10] # 10
# B_range = [5e5, 14e5] # 10
# x0_range = [5, 50] # 10
# c_range = [0.03, 0.075] # 10
# d_range = [5, 6.8] # 10
# 
# for i in A_range:  #iterate over varbs
#     for j in sigma_range:
#         for k in B_range:
#             for l in x0_range:
#                 for m in c_range:
#                     for n in d_range:
#                         # Signal spectrum for a given DM cross section and mass
#                         int_func = theoretical_func(x, Hal, i, k, j, l, m, n)
# 
#                         # fit integral function
#                         f = InterpolatedUnivariateSpline(x[:-1], int_func)  # cubic spline default
# 
#                         # estimated exposure time
#                         mineral_age_est = 100
# 
#                         # create empty dataframe to append values to after iteration
#                         integral_vals = pd.DataFrame(columns = ['bin', 'bin_start', 'bin_stop', 'count (# of tracks per 100 Myr)'])
# 
#                         #width of each bin in measured data plot
#                         bin_width = (max(lengths_per_vol) - min(lengths_per_vol)) / len(bins)
#                         
# 
#                         # iterate over integrals and plot a histogram over SAME bins!
#                         for o in range(len(bins)-1):
#                             x_batch = (bins[o], bins[o+1])
#                             
#                             count = f.integral(x_batch[0], x_batch[1]) * mass * mineral_age_est # mass (1 cm^3 of halite) # Myr estimate
#                             
#                             count_per_area = (count **(2./3)) * scanned_area # convert theoretical track length spectra to tracks/area
#                             
#                             integral_vals.loc[len(integral_vals.index)] = [bins[o], x_batch[0], x_batch[1], count_per_area] 
#                             # tabulate integral vals (counts per bin)
# 
# 
#                         
#                         # empty list
#                         list_cs = []
#                         
#                         for p in range(len(exp_count)):
#                             
#                             # compute Poisson's uncertainties
#                             num_sigma = 1
#                             alpha=erfc(num_sigma/np.sqrt(2))
#                         
#                             if integral_vals.iloc[p,3] > exp_count[p]:
#                                 #Error Bars
#                                 #two-sided upper limit
#                                 uncertainty = 0.5 * chi2.ppf(1 - alpha/2, 2*(exp_count[p] + 1)) - exp_count[p]
#                             
# 
#                             if integral_vals.iloc[p,3] < exp_count[p]:
#                                 #two-sided lower limit
#                                 uncertainty = exp_count[p] - 0.5 * chi2.ppf(alpha/2, 2 * exp_count[p])
# 
#                             # compute chi-square PER BIN and append to list
#                             chi_square = (exp_count[p] - integral_vals.iloc[p,3])**2 / (uncertainty**2)
#                             list_cs.append(chi_square)
#                         
#                         # sum list together to get test statistic
#                         chi_squared_sum = sum(list_cs)
#                         
#                         # append chi-square test statistic and variable combo to dataframe
#                         chi_test_vals.loc[len(chi_test_vals.index)] = [chi_squared_sum, i, k, l, j, m, n] 
# 
#                    
# chi_test_vals.to_csv('Chi-Squared Test-test1-2 per varb-spy.csv', encoding='utf-8', index=False)                    
# =============================================================================




#print("run time: ", datetime.now() - startTime)
