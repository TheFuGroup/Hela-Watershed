def find_nearest_centroid(centroids, x, y):
    min_dist = float('inf')
    nearest_center = ()
    for center in centroids:
        distance = (center[0] - x)**2 + (center[1] - y)**2
        if distance < min_dist:
            min_dist = distance 
            nearest_center = center
    return nearest_center



import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import color, measure

IMG_PATH = "09162021complete.tif"

_, masks = cv2.imreadmulti(IMG_PATH, [], cv2.IMREAD_ANYDEPTH)

# print(len(masks))

img = masks[0]
print(img.shape)

cells = img[img.shape[0]-1050:img.shape[0]-550, img.shape[1]-500:img.shape[1]] 

img_temp = cv2.merge((cells,cells,cells))

ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

from skimage.segmentation import clear_border
opening = clear_border(opening) #Remove edge touching grains

sure_bg = cv2.dilate(opening,kernel,iterations=10)
# cv2.imshow("Sure background", sure_bg)
# cv2.waitKey(0)


dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2, 3)


ret2, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)
# cv2.imshow("Sure background", sure_fg)
# cv2.waitKey(0)
unknown = cv2.subtract(sure_bg,sure_fg)

ret3, markers = cv2.connectedComponents(sure_fg)

markers = markers+10

markers[unknown==255] = 0
# plt.imshow(markers, cmap='jet')   #Look at the 3 distinct regions.

markers = cv2.watershed(img_temp,markers)

img_temp[markers == -1] = [0,255,255]  

img2 = color.label2rgb(markers, bg_label=0)

# print(markers.shape)

cv2.imshow('Overlay on original image', img_temp)
cv2.imshow('Colored Grains', img2)
cv2.waitKey(0)

# regions = measure.regionprops(markers, intensity_image=cells)


# centriods = []
# intensities = []

# #Can print various parameters for all objects
# for prop in regions:
#     centriods.append(prop.centroid)
#     intensities.append(prop.perimeter)
#     print(prop.mean_intensity)