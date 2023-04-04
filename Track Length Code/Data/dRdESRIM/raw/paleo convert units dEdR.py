# -*- coding: utf-8 -*-
"""
Code to import SRIM data and plot as spectrum.

Created on Tue Oct 11 11:17:36 2022

@author: emili
"""
import numpy as np

infile = "Thorium in Muscovite.txt"
outfile = "Th-Musc.txt"

N_skip = 0

E_list = np.loadtxt(infile, usecols=(0,), skiprows=N_skip)
Eunits_list = np.loadtxt(infile, usecols=(1,), dtype=str, skiprows=N_skip)

for i in range(len(E_list)):
    unit = Eunits_list[i]
    if (unit == "eV"):
        E_list[i] *= 1e-3
    elif (unit == "MeV"):
        E_list[i] *= 1e3
        
dEedx =  np.loadtxt(infile, usecols=(2,), skiprows=N_skip)
dEndx =  np.loadtxt(infile, usecols=(3,), skiprows=N_skip)

x_list = np.loadtxt(infile, usecols=(4,), skiprows=N_skip)
xunits_list = np.loadtxt(infile, usecols=(5,),dtype=str,skiprows=N_skip)

for i in range(len(x_list)):
    unit = xunits_list[i]
    if (unit == "A"):
        x_list[i] *= 1e-4

        
np.savetxt(outfile, np.vstack([E_list, dEedx, dEndx, x_list]).T, 
           header="Energy(keV)    dEe/dx(keV/micro_m)  dEn/dx(keV/micro_m)  x(micro_m)")