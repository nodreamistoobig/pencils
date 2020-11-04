import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, try_all_threshold, threshold_triangle
from scipy.ndimage import morphology

def toGray (image):
    return (0.2989 * image[:, :, 0] + 0.587 * image[:, :, 1] +  0.114 * image[:, :, 2]).astype("uint8")

def hist(gray):
    H = np.zeros(256)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            val = gray[i,j]
            H[val]+=1
    return H

def binarization (image, limit_min, limit_max):
    B = image.copy()
    B[B<limit_min] = 0
    B[B>=limit_max] = 0
    B[B>0] = 1
    return B

def elongation(region):
    bbox = region.bbox
    elongation =  (bbox[0]-bbox[2])/(bbox[1]-bbox[3])
    area = region.area
    if elongation < 1:
        elongation = 1/elongation
    box_area = (bbox[0]-bbox[2])*(bbox[1]-bbox[3])
    return elongation * (box_area/area)


for i in range(1,13):
    image_name = "images/img (" + str(i) + ").jpg"
    image = plt.imread(image_name)
    print (image_name)
    gray = toGray(image)
    H = hist(gray)
    
    thresh = threshold_triangle(gray)
    binary = binarization(gray, 0, thresh)
    
    binary = morphology.binary_dilation(binary, iterations=1)
    
    labeled = label(binary)
    
    areas = []
    
    for region in regionprops(labeled):
        areas.append(region.area)
    
    for region in regionprops(labeled):
        if region.area < np.mean(areas):
            labeled[labeled == region.label] = 0
        bbox = region.bbox
        if bbox[0] == 0 or bbox[1] == 0:
            labeled[labeled == region.label] = 0
    
    labeled[labeled>0]=1
    labeled = label(labeled)
    
    pencils = 0
    
    for region in regionprops(labeled):
       if elongation(region) > 10:
           pencils+=1
    
    print(pencils)