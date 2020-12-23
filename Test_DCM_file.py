# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:26:39 2020

@author: dhkim
"""


import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file

# fpath = get_testdata_file('CT_small.dcm')
fpath = '1417_103_r8_120cm_Long_Chest_1.dcm'
ds = dcmread(fpath)

# Normal mode:
print()
print(f"File path........: {fpath}")
print(f"SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
print()

pat_name = ds.PatientName
display_name = pat_name.family_name + ", " + pat_name.given_name
print(f"Patient's Name...: {display_name}")
print(f"Patient ID.......: {ds.PatientID}")
print(f"Modality.........: {ds.Modality}")
print(f"Study Date.......: {ds.StudyDate}")
print(f"Image size.......: {ds.Rows} x {ds.Columns}")
print(f"Pixel Spacing....: {ds.PixelSpacing}")

# use .get() if not sure the item exists, and want a default value if missing
print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")

dcmImg = ds.pixel_array
h,w=dcmImg.shape
imgType = str(dcmImg.dtype)

wFileName= fpath[:-4] +"_"+ str(w)+"x"+str(h)+"_"+imgType +".raw"

dcmImg.tofile(wFileName)

print("Write File: " + wFileName)

# plot the image using matplotlib

plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
plt.show()


#%% random get 

import numpy as np 
import cv2

image = dcmImg
image_size = 512

for id in range(5):
    h, w = image.shape[:2]
    i = np.random.randint(h - image_size + 1)
    j = np.random.randint(w - image_size + 1)
    
    cropImage = image[i:i + image_size, j:j + image_size]
    
    plt.imshow(cropImage, cmap=plt.cm.gray)
    # cv2.imshow("crop Image", cropImage)
    key = cv2.waitKey(-1)
    
    
    # "q": quit
    if key == 113:
        break
    
    
    # plt.imshow(cropImage, cmap=plt.cm.gray)
    # plt.show()





#%% test
# import cv2
# inFileName  = "1417_103_r8_120cm_Long_Chest_1.dcm"

# img=cv2.imread(inFileName)

# imgGray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# img = imgGray

# h,w= img.shape

# gw = w
# gh = h