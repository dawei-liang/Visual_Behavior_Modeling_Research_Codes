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
        for i in range(len(frame_sets)):
            # Load 100 frames
            if i < 100:
                img = cv2.imread(self.dir_to_save_frames_in_use + frame_sets[i])   
                training_data[count] = img
                count += 1
                
        # Load ground truth heatmaps as labels
        truth_heatmap_sets = [x for x in os.listdir(self.dir_ground_truth_heatmap) if x.endswith('.npz')]  
        training_labels = np.empty((100, self.height, self.width))
        heatmap_object = heatmap_generation.heatmap(1024, 576)
        count = 0
        for j in range(len(truth_heatmap_sets)):
            # Load 100 labels
            if j < 100:
                img = np.load(self.dir_ground_truth_heatmap + truth_heatmap_sets[j])['heatmap']   
                img = heatmap_object.normalize(img)   # heatmap normalization
                training_labels[count] = img
                count += 1
                
        # Reshape input labels (heatmaps)
        training_labels = np.reshape(training_labels, (training_labels.shape[0], 576, 1024, 1))
        # Whitening
        training_data = (training_data - np.mean(training_data)) / np.std(training_data)
        
        return training_data, training_labels
    

