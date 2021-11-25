from os import set_inheritable
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import color, measure
import imageio

class WatershedProcessor:

    ## Init function to intialize the WatershedProcessor
    def __init__(self, img_path):
        self.img_path = img_path

        ret, masks = cv2.imreadmulti(self.img_path, [], cv2.IMREAD_ANYDEPTH)
        self.frames = []
        for mask in masks:
            self.frames.append(mask[mask.shape[0]-1050:mask.shape[0]-550, mask.shape[1]-500:mask.shape[1]])
        self.centroid_dist_threshold = 1000
        self.area_threshold = 0.5
        self.current_centroids = []
        self.current_intensities = []
        self.cells_intensities = []
        self.current_area = []
        self.cells_bounding_boxes = []

    def find_nearest_centroid(self, centroids, areas, prop):
        min_dist = float('inf')
        second_min_dist = float('inf')
        nearest_center_idx = 0
        second_nearest_center_idx = 0
        x, y = prop.centroid[0], prop.centroid[1]
        for idx in range(len(centroids)):
            distance = (centroids[idx][0] - x)**2 + (centroids[idx][1] - y)**2
            if distance < min_dist:
                min_dist = distance 
                nearest_center_idx = idx
            elif distance < second_min_dist:
                second_min_dist = distance
                second_nearest_center_idx = idx
        if min_dist < self.centroid_dist_threshold and prop.area > self.area_threshold * areas[nearest_center_idx] and prop.area < (1/self.area_threshold)*areas[nearest_center_idx]:
            return nearest_center_idx
        if second_min_dist < self.centroid_dist_threshold and prop.area > self.area_threshold * areas[second_nearest_center_idx] and prop.area < (1/self.area_threshold)*areas[second_nearest_center_idx]:
            return nearest_center_idx
        return -1
    
        
    ## Process a single frame of tif image, return all cell regions in this frame
    def process_frame(self, index):
        img = self.frames[index]
        cells = img
        img_temp = cv2.merge((cells,cells,cells))
        ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
        from skimage.segmentation import clear_border
        opening = clear_border(opening) #Remove edge touching grains
        sure_bg = cv2.dilate(opening,kernel,iterations=10)
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        ret2, sure_fg = cv2.threshold(dist_transform,0.28*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        ret3, markers = cv2.connectedComponents(sure_fg)
        markers = markers+10
        markers[unknown==255] = 0
        markers = cv2.watershed(img_temp,markers)
        img_temp[markers == -1] = [0,255,255]  
        regions = measure.regionprops(markers, intensity_image=cells)
        return regions

    def process_tif(self):
        self.process_firstframe()
        for idx in range(1, len(self.frames)):
            regions = self.process_frame(idx)
            for prop in regions:
                tracking_index = self.find_nearest_centroid(self.current_centroids, self.current_area, prop)
                if tracking_index >= 0:
                    self.cells_intensities[tracking_index].append(prop.area * prop.mean_intensity)
                    self.current_centroids[tracking_index] = prop.centroid
                    self.current_area[tracking_index] = prop.area
                    self.cells_bounding_boxes[tracking_index].append(prop.bbox)
                # self.current_intensities.append(prop.area * prop.mean_intensity)


        
    def process_firstframe(self):
        regions = self.process_frame(0)
        for prop in regions:
            self.current_centroids.append(prop.centroid)
            self.current_area.append(prop.area)
            # self.current_intensities.append(prop.area * prop.mean_intensity)
            self.cells_intensities.append([prop.area * prop.mean_intensity])
            self.cells_bounding_boxes.append([prop.bbox])

    def plot_curve(self):
        self.process_tif()
        for cell_id in range(1, len(self.cells_intensities)):
            plt.plot(self.cells_intensities[cell_id])
            plt.savefig("testing/Plots/cell" + str(cell_id) + ".png")
            plt.clf()
            idx = 0
            bounding_boxes = []        
            for box in self.cells_bounding_boxes[cell_id]:
                bounding_boxes.append(self.frames[idx][box[0]:box[2], box[1]:box[3]])
                idx = idx + 1
            imageio.mimsave("testing/GIF/cell" + str(cell_id) + ".gif", bounding_boxes, fps=5)
            # box = self.cells_bounding_boxes[cell_id][0]
            # cv2.imwrite("testing/GIF/cell" + str(cell_id) + ".png", self.frames[0][box[0]-5:box[2]+5, box[1]-5:box[3]+5])
        

if __name__ == "__main__":
    processor = WatershedProcessor(img_path="09162021complete.tif")
    # print(len(processor.frames))
    processor.plot_curve()
    # print(processor.cells_intensities[0])