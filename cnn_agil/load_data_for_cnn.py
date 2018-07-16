# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 15:04:38 2018

@author: david
"""
import os
import cv2
import numpy as np
import heatmap_generation

#%%

class read_data:
    def __init__(self, dir_to_save_frames_in_use, dir_to_save_heatmap, height, width):
        self.dir_to_save_frames_in_use = dir_to_save_frames_in_use
        self.dir_ground_truth_heatmap = dir_to_save_heatmap
        self.width = width
        self.height = height
        
    def load_data(self):   
        # Load frames as training data
        frame_sets = [x for x in os.listdir(self.dir_to_save_frames_in_use) if x.endswith('.jpg')]
        training_data = np.empty((100, self.height, self.width, 3))
        count = 0
        for frame in frame_sets:
            index = int(frame.strip('frame').strip('_y.jpg'))   # Get loaded frame index
            if index <= 4339:
                img = cv2.imread(self.dir_to_save_frames_in_use + frame)   # Load the frame to visualize
                training_data[count] = img
                count += 1
                
        # Load ground truth heatmaps as labels
        truth_heatmap_sets = [x for x in os.listdir(self.dir_ground_truth_heatmap) if x.endswith('.jpg')]  
        training_labels = np.empty((100, self.height, self.width, 3))
        count = 0
        for heatmap in truth_heatmap_sets:
            index = int(heatmap.strip('frame').strip('.jpg'))   # Get loaded frame index
            if index <= 4339:
                img = cv2.imread(self.dir_ground_truth_heatmap + frame)   # Load the frame to visualize
                training_labels[count, :, :, :] = img
                count += 1
        # Whitening
        training_data = (training_data - np.mean(training_data)) / np.std(training_data)
        heatmap_object = heatmap_generation.heatmap(1024, 576)
        training_labels = heatmap_object.normalize(training_labels)
        
        return training_data, training_labels


