# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 08:24:14 2023

Code to automatically detect etch pits from Keyence VK-X1000 images.

started from: 
    https://github.com/rishim9816/Hole-detection-in-sarees-using-Image-Processing/blob/master/GaussImg.py

@author: emilie
"""
from PIL import Image, ImageOps
from PIL import Image, ImageFont, ImageDraw 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pandas import read_csv
from matplotlib import cm
import math
from datetime import datetime
startTime = datetime.now()



# import image and RAW height information in .csv format
File_Name = [file_name.png] #in string format # no scale bars, text, borders # meant for laser images only
Height_File_Name = [file name.csv] # in string format

# change per image! magnification image was captured at. usually 50 or 150
magnification = 50

## This needs to be double checked depending on the Keyence parameters
# resolution depending on user-given magnification
#resolution_150x = 0.4741  # high res. 150x image
resolution_150x = 0.0474 #microns per pixel for 150x image 2048x1536
resolution_50x = 0.1399 # for 2048x1536 pixel image
#resolution_50x = 0.2797 #microns per pixel for 50x image

# open image
img = cv2.imread("images/" + File_Name,1);

# open height csv
height_csv = read_csv(Height_File_Name, 
                      skiprows = 15, header=None)


# read image in pixel format to get pixel size of image
img_for_size = Image.open("images/" + File_Name)

# parameters for image size in pixels
width, height = img_for_size.size

# amount of pixels you want cut off equally around image (possibly reduce noise  esp on stitched images)
pixels_crop = 0

# Setting the points for cropped image
left = pixels_crop
top = pixels_crop
right = width - pixels_crop
bottom = height - pixels_crop
 
# Cropped image of above dimension
# (It will not change original image)
crop_img = img[top:bottom, left:right]

#grab cropped image dimensions to fit height csv to
new_height = crop_img.shape[0]
new_width = crop_img.shape[1]

# display cropped image to check for errors/partial pits
# =============================================================================
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)
# 
# =============================================================================

# crop height .csv (accounting for one less row at end)
# convert to pd df
height_df = pd.DataFrame(data=height_csv)

# =============================================================================
# #cropping height dataframe - 7 pixels around (manual at the moment)
# height_df = height_df.drop(labels=[0,1,2,3,4], axis=0)
# height_df = height_df.drop(labels=[761,762,763,764,765], axis=0)
# 
# height_df = height_df.drop(labels=[0,1,2,3,4], axis=1)
# height_df = height_df.drop(labels=[1017,1018,1019,1020,1021], axis=1)
# =============================================================================

# convert image to greyscale
grey = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)

# take the gaussian blur image
gauss = cv2.GaussianBlur(grey,(13,13),100) # gaussian smoothing
## (5,5) standard gaussian kernel size - should be odd
## 100 is standard deviation


# show gauss blur image - check for errors
gauss_display = gauss
gauss_display = cv2.resize(gauss_display, (1024, 768))
cv2.imshow('gaussian blur', gauss_display)
cv2.waitKey(0)

## next few lines are for binary image creation if needed
#-----------
# use these parameters to create a binary image, if needed
threshold_val = 170 #manipulate this for "sensitivity"
# like the sensitivity of contrast you want to consider. 
# low value = only very dark contrast

#find threshold to convert into pure black white image
ret,thresh = cv2.threshold(gauss,threshold_val,255,0)
## https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
## 127, 255 = black; 0 = white


# Show binary image created w given threshold
# =============================================================================
# cv2.imshow("Thresh", thresh)
# cv2.waitKey(0)
# =============================================================================

# =============================================================================
# # save this image
# cv2.imwrite('test data hole detection/blob map ' + File_Name, thresh)
# =============================================================================
# ---------------



########################### Blob Detection ############################

#detect holes using blob detector
#detector = cv2.SimpleBlobDetector_create()
# https://docs.opencv.org/3.4/d0/d7a/classcv_1_1SimpleBlobDetector.html
# https://learnopencv.com/blob-detection-using-opencv-python-c/
#  Blob is a group of connected pixels in an image that share some common property

""" 
Thresholding : Convert the source images to several binary images by thresholding 
the source image with thresholds starting at minThreshold. 
These thresholds are incremented  by thresholdStep until maxThreshold. 
So the first threshold is minThreshold, the second is minThreshold + thresholdStep, 
the third is minThreshold + 2 x thresholdStep, and so on.

Grouping : In each binary image,  connected white pixels are grouped.  
Let’s call these binary blobs.

Merging  : The centers of the binary blobs in the binary images are computed, 
and blobs located closer than minDistBetweenBlobs are merged.

Center & Radius Calculation:  The centers and radii of the newly merged blobs 
are computed and returned.

####

https://learnopencv.com/blob-detection-using-opencv-python-c/

You can filter blobs by color, circularity, convexity, and intertia ratio
(how elongated a shape is).

These filters will likely need to change PER IMAGE! Future work would include automating/improving this!
"""

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 100
#params.maxThreshold = 500
 
# Filter by Area
params.filterByArea = True
params.minArea = 25 # 5 pixel area = 0.492 == 0.5 microns
params.maxArea = 500
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.01
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.3
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.05
 
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
  detector = cv2.SimpleBlobDetector(params)
else : 
  detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs using gauss filtered image
keypoints = detector.detect(gauss)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(gauss, 
                                      keypoints, 
                                      np.array([]), 
                                      (0,0,255), 
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show image with keypoints detected
im_with_keypoints_display = im_with_keypoints

#resize image so it fits in the window
im_with_keypoints_display = cv2.resize(im_with_keypoints_display, (1024, 768))
cv2.imshow("Keypoints", im_with_keypoints_display)
cv2.waitKey(0)


## ---------------------------------------------------------------------------

# Process height data before plotting
# First minus a reference plane with MSE Linear Regression

# convert gauss image to DF
gauss_df = pd.DataFrame(data=gauss)


# we are going to normalize the data, finding a vector theta that will describe the
# desired flattened plane. 

m = gauss.shape[0] #size of image
n = gauss.shape[1]

# create grid of data in same dimensions of image
X1, X2 = np.mgrid[:m, :n]

## ----------------------------------------------------------------------------
# # comment this section out if you want to take away linear regression and plane subtraction

# ============================================================================= 
# # convert dataframe to numpy array
# height_data = height_df.to_numpy()
# 
# #Regression
# X = np.hstack(( np.reshape(X1, (m*n, 1)), np.reshape(X2, (m*n, 1))))
# X = np.hstack(( np.ones((m*n, 1)), X))
# YY = np.reshape(height_data, (m*n, 1))
# 
# # Normal Equation, to find the angle theta 
# theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
# 
# # plane detected by lin regression
# plane = np.reshape(np.dot(X, theta), (m, n));
# 
# # Subtraction of detected plane from array of height data
# Y_sub = height_data - plane
# 
# 
# # =============================================================================
# # 
# # 
# # 
# # # BEFORE
# # ax = fig.add_subplot(1, 2, 1, projection='3d')
# # ax.plot_surface(X1, X2, height_data, rstride=1, cstride=1,cmap = cm.coolwarm,
# #                 linewidth = 0, antialiased = False)
# # ax.set_title('before');
# # ax.view_init(5, 180)
# # plt.show()
# # =============================================================================
# 
# fig = plt.figure(figsize=plt.figaspect(0.5))
# # AFTER
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# ax.plot_surface(X1, X2, Y_sub, rstride=10, cstride=10, cmap=cm.coolwarm,
#                        linewidth = 0, antialiased = False)
# ax.set_title('after plane subtraction');
# ax.view_init(5, 180) # angle to view 3D plot
# plt.show()
# 
# # switch back to height dataframe but used normalized dats
# height_df = pd.DataFrame(data = Y_sub)
# =============================================================================
## ----------------------------------------------------------------------------


## -----------------------------------------------------------------------------
# Plotting the histogram of blob diameter and depth

# create empty list to start loop
blob_hist = []

#grab pixel coordinates of blobs, size, and convert to microns
for i in range(len(keypoints)):
    x = keypoints[i].pt[0] # x coordinate of blob
    y = keypoints[i].pt[1] # y coordinate of blob
    s = keypoints[i].size # rough pixel diameter of blob
    
    if magnification == 150:
        blob_diameter = resolution_150x * s #convert to microns
    
    if magnification == 50:
        blob_diameter = resolution_50x * s #convert to microns
    
    # append blob diameter to list
    blob_hist.append(blob_diameter)

   
#plot histogram of blob diameter
n, bins, patches = plt.hist(blob_hist, 
                            10, 
                            facecolor='g', 
                            edgecolor='black', 
                            linewidth=1.2)


plt.xlabel('Blob Diameter in Microns')
plt.ylabel('Frequency')
plt.title('Histogram of Blob Diameter')
plt.show()


arr = np.array(crop_img) # cropped image as array

depth_hist = [] # empty list for loop

# outline of code to grab pixel coordinates of blobs, size, and convert to microns
for i in range(len(keypoints)):
    x = keypoints[i].pt[0] # column
    y = keypoints[i].pt[1] # row
    s = keypoints[i].size # rough pixel diameter of blob
    
    row_coor = round(y) # round to nearest integer
    col_coor = round(x)
    
    empty_height = [] # empty list for j loop
    
    # within a given range of pixel area, search for the minimum depth
    for j in range(25): # pixel area around coordinate to search
        height_coor_plus = height_df.iloc[row_coor + j, col_coor + j] 
        empty_height.append(height_coor_plus) #append each value to list
        
        height_coor_plus_minus = height_df.iloc[row_coor + j, col_coor - j]
        empty_height.append(height_coor_plus_minus)
        
        height_coor_minus_plus = height_df.iloc[row_coor - j, col_coor + j]
        empty_height.append(height_coor_minus_plus)
        
        height_coor_minus = height_df.iloc[row_coor - j, col_coor - j]
        empty_height.append(height_coor_minus)
        
    height_real = math.dist((np.average(empty_height),), (min(empty_height),))
    #np.average(empty_height) - min(empty_height) # max - min depth
    #abs(np.average(height_df.mean())) - abs(min(empty_height))
    # average value or level of surface across sample, minus the miniumum (or largest depth)
    # accured in loop
    
    #empty_height_min.append(abs(min(empty_height)))
    depth_hist.append(abs(height_real)*1000) #difference between min and max
    # converted to nm instead of microns
    
    
   
#plot histogram of blob depth
n, bins, patches = plt.hist(depth_hist, 
                            20, # number of bins
                            facecolor='g', 
                            edgecolor='black', 
                            linewidth=1.2)


plt.title('Histogram of Blob Depth')   
plt.xlabel("Depth in Nanometers")
plt.ylabel("Frequency")

## ---------------------------------------------------------------------------
    
# show the cropped image with the associated pinned coordinates/labels from keypoints
image = Image.fromarray(arr, "RGB") #convert array to image
image_draw = ImageDraw.Draw(image) #convert to editable image


for i in range(len(keypoints)):
    x = keypoints[i].pt[0] # column
    y = keypoints[i].pt[1] # row
    
    row_coor = round(y) # round to nearest integer
    col_coor = round(x)
    
    # to color the coordinate pixels on the crop image copy
    # makes sure we are pinning the blob coordinates
    # color is pink
    arr[row_coor, col_coor] = [248, 131, 121] # to make sure we are actually pinning 
    arr[row_coor, col_coor + 1] = [248, 131, 121] # the correct coordinates
    arr[row_coor + 1, col_coor + 1] = [248, 131, 121]
    arr[row_coor + 1, col_coor] = [248, 131, 121]
    
    blob_text = str(i+1) # 0 = 1 etc
    
    #label each blob
    image_draw.text((col_coor + 1, row_coor + 1), blob_text, (0, 0, 0), size=22) 

image.save("blob maps/" + File_Name + " blob map.jpg") #save blob map


##----------------------------------------------------------------------------
##  save info to roughness dataframe

# need to read first lines of height info .csv - this is where a lot of info is stored
height_info = read_csv(Height_File_Name, header = None, nrows=6)
height_info_2 = read_csv(Height_File_Name, header = None, skiprows=6, nrows=1)
height_info_3 = read_csv(Height_File_Name, header = None, skiprows=7, nrows=6)


# grab only first two columns of first 13 rows (where all info is stored)
# will be the same for every file
#height_info = height_info.iloc[:13,:2]

# convert to DF
height_info_df = pd. DataFrame(data = height_info)
height_info_2_df = pd. DataFrame(data = height_info_2)
height_info_3_df = pd. DataFrame(data = height_info_3)



height_info_df = pd.concat([height_info_df, height_info_2_df])
height_info_df = height_info_df.drop(labels=[2], axis=1)
height_info_df = pd.concat([height_info_df, height_info_3_df])



## ------------------- add real image dimensions ------------------------------
# add in real dimensions - same for every image
# for a 50x image:
if magnification == 50:
    mag_50x = pd. DataFrame([['real horizontal - um', 286.458],
                            ['real vertical - um', 214.774],
                            ['pixel resolution - um', 0.2797]])
    
    # add to dataframe
    csv_data = pd.concat([height_info_df, mag_50x])

# for a 150x image
if magnification == 150:
    mag_150x = pd. DataFrame([['real horizontal - um', 97.122],
                             ['real vertical - um', 72.817],
                             ['pixel resolution - um', 0.0474]])
    
    # add to dataframe
    csv_data = pd.concat([height_info_df, mag_150x])


## ------------------------ add track density ---------------------------------

# append density
if magnification == 50:
    horizontal_real = 286.458  # um
    vertical_real = 214.774 #um
    
    density = (len(keypoints)/(horizontal_real * vertical_real))/10e-8 
    # tracks/um^2 converted to tracks/cm^2
    
    # write in sci notation
    density_df = pd.DataFrame([['density - tracks/cm^2', "{:e}".format(density)]])
    
    # append to dataframe
    csv_data = pd.concat([csv_data, density_df])

if magnification == 150:
    horizontal_real = 97.122  # um
    vertical_real = 72.817 #um
    
    density = (len(keypoints)/(horizontal_real * vertical_real))/10e-8 
    # tracks/um^2 converted to tracks/cm^2
    
    # write in sci notation
    density_df = pd.DataFrame([['density - tracks/cm^2', "{:e}".format(density)]])
    
    # append to dataframe
    csv_data = pd.concat([csv_data, density_df])


## --------------------- add roughness parameters -----------------------------

## compute roughness & create dataframe with roughness parameters

# arithmetic average
Ra = abs(np.average(height_df.mean()))


# root mean squared
rq_list = []

for i in range(gauss_df.shape[1]):
    for j in range(gauss_df.shape[0]):
        rq_list.append((height_df.iloc[j,i])**2)
    
Rq = np.sqrt(sum(rq_list)/gauss_df.size)

# skewness
rsk_list = []

for i in range(gauss_df.shape[1]):
    for j in range(gauss_df.shape[0]):
        rsk_list.append((height_df.iloc[j,i])**3)
        
Rsk = sum(rsk_list)/(gauss_df.size * (Rq**3))

# roughness parameters dataframe
roughness_df = pd.DataFrame([['Ra - um', Ra], 
                             ['RMS Rq - um', Rq],
                             ['Skewness Rsk', Rsk]])

# append to dataframe
csv_data = pd.concat([csv_data, roughness_df])

## ----------------------- add file name as first row ------------------------

# include file name in csv
File_Name_df = pd.DataFrame([['File Name', File_Name]])
csv_data = pd.concat([File_Name_df, csv_data]).reset_index(drop = True)

## ----------------------------- min/max height ------------------------------

minimum_height = min(height_df.min())
maximum_height = max(height_df.max())

# create dataframe of values then append
min_max_df = pd.DataFrame([['min z-value AFTER lin reg', minimum_height], 
                             ['max z-value AFTER lin reg', maximum_height]])

csv_data = pd.concat([csv_data, min_max_df])



## ---------------- list of blobs w depths and diameters ----------------------
blob_cols = ['detected blobs', 'diameter (um)', 'depths (nm)'] # column names

blob_list = [] # empty list to append to 

for i in range(len(keypoints)):
    blob_list.append([i + 1, blob_hist[i], depth_hist[i]]) 
    # loop thru keypoints & grab data
    
blob_df = pd.DataFrame(blob_list, columns = blob_cols) # turn into dataframe
        

csv_data = pd.concat([csv_data, blob_df])   #merge into csv data
    

## ------------------------- SAVE TO .CSV FILE! -------------------------------
# save all information to a .csv file
csv_data.to_csv('statistical csv files/' + File_Name[:-4]  + ' statistical profile.csv', index=False)



## ----------------------------------------------------------------------------
# plotting etch pits on 3D space

# for cohensive datatype
X1_df = pd.DataFrame(data = X1)
X2_df = pd.DataFrame(data = X2)

# create loop to plot all keypoints (detected blobs)

for k in range(len(keypoints)):

    x_coor = []
    y_coor = []
    z_coor = []

    # plot 3D pits
    # outline of code to grab pixel coordinates of blobs, size, and convert to microns

    x = keypoints[k].pt[0] # column
    y = keypoints[k].pt[1] # row
    s = keypoints[k].size # rough pixel diameter of blob
    
    row_coor = round(y) # round to nearest integer
    col_coor = round(x)

    origin_x, origin_y = row_coor - 6, col_coor - 6

    for i in range(12):
        for j in range(12):
            z_coor.append(height_df.iloc[origin_x + i, origin_y + j])
            x_coor.append(X1_df.iloc[origin_x + i, origin_y + j])
            y_coor.append(X2_df.iloc[origin_x + i, origin_y + j])



    x_coor = np.reshape(np.array(x_coor), 
                    (len(np.unique(x_coor)),len(np.unique(y_coor))))
    y_coor = np.reshape(np.array(y_coor), 
                    (len(np.unique(x_coor)),len(np.unique(y_coor))))
    z_coor = np.reshape(np.array(z_coor), 
                    (len(np.unique(x_coor)),len(np.unique(y_coor))))

    # plot the data
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x_coor, y_coor, z_coor, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(5, 180)

    plt.show()

## --------------------- plot color map -----------------------------------
# =============================================================================
# for k in range(len(keypoints)):
#     # plot color map
#     # outline of code to grab pixel coordinates of blobs, size, and convert to microns
# 
#     x = keypoints[k].pt[0] # column
#     y = keypoints[k].pt[1] # row
#     s = keypoints[k].size # rough pixel diameter of blob
#     
#     row_coor = round(y) # round to nearest integer
#     col_coor = round(x)
# 
#     z = height_df.iloc[row_coor - 10: row_coor + 10, col_coor - 10:col_coor + 10]
#     
#     x = range(z.shape[1])
#     y = range(z.shape[0])
# 
# 
#     X1, X2 = np.meshgrid(x, y)
#     #X1 = X1.reshape(-1,1)
#     #X2 = X2.reshape(-1, 1)
# 
#     fig, ax = plt.subplots()
#     im = ax.pcolormesh(X1, X2, z)
#     fig.colorbar(im)
#     plt.show()
# =============================================================================


print("run time:", datetime.now() - startTime)
