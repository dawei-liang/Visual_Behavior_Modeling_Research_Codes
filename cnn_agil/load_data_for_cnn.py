# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 15:04:38 2018

@author: david
"""
import os
import cv2
import numpy as np
import heatmap_generation
import copy as cp
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
        # Sort file name by frame index
        for i in range(len(frame_sets)):
             frame_sets[i] = int(frame_sets[i].strip('frame').strip('.jpg'))
        frame_sets.sort()
        for i in range(len(frame_sets)):
             frame_sets[i] = 'frame' + str(frame_sets[i]) + '.jpg'

        framesSize = len(frame_sets)
        training_data = np.empty((framesSize, self.height, self.width, 3))
        frame_idx_list = []
        for i in range(len(frame_sets)):
             frame_idx = int(frame_sets[i].strip('frame').strip('.jpg'))
             frame_idx_list.append(frame_idx)
             print('frame:', frame_idx)   # For index check
             img = cv2.imread(self.dir_to_save_frames_in_use + frame_sets[i])
             training_data[i] = cp.deepcopy(img)
                
        # Load ground truth heatmaps as labels
        truth_heatmap_sets = [x for x in os.listdir(self.dir_ground_truth_heatmap) if x.endswith('.npz')]
        # Sort file name by frame index
        for j in range(len(truth_heatmap_sets)):
             truth_heatmap_sets[j] = int(truth_heatmap_sets[j].strip('heatmap').strip('.npz'))
        truth_heatmap_sets.sort()
        for j in range(len(truth_heatmap_sets)):
             truth_heatmap_sets[j] = 'heatmap' + str(truth_heatmap_sets[j]) + '.npz'

        assert len(frame_sets) == len(truth_heatmap_sets), 'Loaded frame sets and heatmap sets are of different length.'
        training_labels = np.empty((framesSize, self.height, self.width))
        heatmap_object = heatmap_generation.heatmap(self.width, self.height)
        heatmap_idx_list = []
        for j in range(len(truth_heatmap_sets)):
            heatmap_idx = int(truth_heatmap_sets[j].strip('heatmap').strip('.npz'))
            heatmap_idx_list.append(heatmap_idx)
            print('heatmap:', heatmap_idx)   # For index check
            img2 = np.load(self.dir_ground_truth_heatmap + truth_heatmap_sets[j])['heatmap']   
            img2 = heatmap_object.normalize(img2)   # heatmap normalization
            training_labels[j] = cp.deepcopy(img2)
                
        # Reshape input labels (heatmaps)
        training_labels = np.reshape(training_labels, (training_labels.shape[0], self.height, self.width, 1))
        
        return training_data, training_labels, frame_idx_list, heatmap_idx_list
    
    '''
    def load_data_sub2(self, framesSize):   
        # Load frames as validation data
        frame_sets = [x for x in os.listdir(self.dir_to_save_frames_in_use) if x.endswith('.jpg')]
        # Sort file name by frame index
        for i in range(len(frame_sets)):
             frame_sets[i] = int(frame_sets[i].strip('frame').strip('.jpg'))
        frame_sets.sort()
        for i in range(len(frame_sets)):
             frame_sets[i] = 'frame' + str(frame_sets[i]) + '.jpg'

        validation_data = np.empty((framesSize, self.height, self.width, 3))
        count = 0
        frame_idx_list = []
        for i in range(len(frame_sets)):
             if i < framesSize:
                 frame_idx = int(frame_sets[i].strip('frame').strip('.jpg'))
                 frame_idx_list.append(frame_idx)
                 print('frame:', frame_idx)   # For index check
                 img_val = cv2.imread(self.dir_to_save_frames_in_use + frame_sets[i])                  
                 validation_data[count] = cp.deepcopy(img_val)
                 #validation_data[count] = (validation_data[count] - np.mean(validation_data[count])) / np.std(validation_data[count])
                 count += 1
                
        # Load ground truth heatmaps as labels
        truth_heatmap_sets = [x for x in os.listdir(self.dir_ground_truth_heatmap) if x.endswith('.npz')]
        # Sort file name by frame index
        for j in range(len(truth_heatmap_sets)):
             truth_heatmap_sets[j] = int(truth_heatmap_sets[j].strip('heatmap').strip('.npz'))
        truth_heatmap_sets.sort()
        for j in range(len(truth_heatmap_sets)):
             truth_heatmap_sets[j] = 'heatmap' + str(truth_heatmap_sets[j]) + '.npz'

  
        validation_labels = np.empty((5400, self.height, self.width))
        heatmap_object = heatmap_generation.heatmap(self.width, self.height)
        count = 0
        heatmap_idx_list = []
        for j in range(len(truth_heatmap_sets)):
             if j < 5400:
             	heatmap_idx = int(truth_heatmap_sets[j].strip('heatmap').strip('.npz'))
                heatmap_idx_list.append(heatmap_idx)
             	print('heatmap:', heatmap_idx)   # For index check
             	img2_val = np.load(self.dir_ground_truth_heatmap + truth_heatmap_sets[j])['heatmap']   
             	img2_val = heatmap_object.normalize(img2_val)   # heatmap normalization
             	validation_labels[count] = cp.deepcopy(img2_val)
             	count += 1
                
        # Reshape input labels (heatmaps)
        validation_labels = np.reshape(validation_labels, (validation_labels.shape[0], self.height, self.width, 1))
        
        return validation_data, validation_labels, frame_idx_list, heatmap_idx_list


    def load_data_sub3(self):   
        # Load frames as validation data
        frame_sets = [x for x in os.listdir(self.dir_to_save_frames_in_use) if x.endswith('.jpg')]
        # Sort file name by frame index
        for i in range(len(frame_sets)):
             frame_sets[i] = int(frame_sets[i].strip('frame').strip('.jpg'))
        frame_sets.sort()
        for i in range(len(frame_sets)):
             frame_sets[i] = 'frame' + str(frame_sets[i]) + '.jpg'

        test_data = np.empty((3700, self.height, self.width, 3))
        count = 0
        frame_idx_list = []
        for i in range(len(frame_sets)):
             if i >= 0 and i < 3700:
                 frame_idx = int(frame_sets[i].strip('frame').strip('.jpg'))
                 frame_idx_list.append(frame_idx)
                 print('frame:', frame_idx)   # For index check
                 img_test = cv2.imread(self.dir_to_save_frames_in_use + frame_sets[i])
                 test_data[count] = cp.deepcopy(img_test)
                 #test_data[count] = (test_data[count] - np.mean(test_data[count])) / np.std(test_data[count])
                 count += 1
                
        # Load ground truth heatmaps as labels
        truth_heatmap_sets = [x for x in os.listdir(self.dir_ground_truth_heatmap) if x.endswith('.npz')]
        # Sort file name by frame index
        for j in range(len(truth_heatmap_sets)):
             truth_heatmap_sets[j] = int(truth_heatmap_sets[j].strip('heatmap').strip('.npz'))
        truth_heatmap_sets.sort()
        for j in range(len(truth_heatmap_sets)):
             truth_heatmap_sets[j] = 'heatmap' + str(truth_heatmap_sets[j]) + '.npz'

  
        test_labels = np.empty((3700, self.height, self.width))
        heatmap_object = heatmap_generation.heatmap(self.width, self.height)
        count = 0
        heatmap_idx_list = []
        for j in range(len(truth_heatmap_sets)):
             if j >= 0 and j < 3700:
             	heatmap_idx = int(truth_heatmap_sets[j].strip('heatmap').strip('.npz'))
                heatmap_idx_list.append(heatmap_idx)
             	print('heatmap:', heatmap_idx)   # For index check
             	img3_test = np.load(self.dir_ground_truth_heatmap + truth_heatmap_sets[j])['heatmap']   
             	img3_test = heatmap_object.normalize(img3_test)   # heatmap normalization
             	test_labels[count] = cp.deepcopy(img3_test)
             	count += 1
                
        # Reshape input labels (heatmaps)
        test_labels = np.reshape(test_labels, (test_labels.shape[0], self.height, self.width, 1))
        
        return test_data, test_labels, frame_idx_list, heatmap_idx_list   '''
    

