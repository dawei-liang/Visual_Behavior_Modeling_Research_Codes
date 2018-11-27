# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 22:57:30 2018

@author: david
"""
import numpy as np
import config
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from numpy import unravel_index
import cv2

import heatmap
import load_data_for_cnn

def centralization(dir_load_frame, dir_load_heatmap, height, width):
    centralized_map = np.zeros((config.pixel['y']*2, config.pixel['x']*2, 3))
    count = 1
    
    data_object = load_data_for_cnn.read_data(dir_load_frame, dir_load_heatmap, height, width)
    img_set, npz_map_set, frame_idx, heatmap_idx = data_object.load_data()   # Attention which sub being loaded
    print('total img size:', img_set.shape[0])
    
    for i in range(img_set.shape[0]):
        # Load heatmaps
        npz_map = npz_map_set[i]
        pos = unravel_index(npz_map.argmax(), npz_map.shape)
        pos = np.asarray(pos)   # vertical(y), horizontal(x)
        print('Predicted gaze position on the frame [y,x]:', pos)
    
        delta_y = config.pixel['y'] - pos[0]
        delta_x = config.pixel['x'] - pos[1]
    
        # Load img
        img = cv2.imread(img_set[i])
    
        print('adding img:', i)
        for color in range(0, 3):
            for y in range(delta_y, delta_y + img.shape[0]):
                for x in range(delta_x, delta_x + img.shape[1]):
                    y_npz = y - delta_y
                    x_npz = x - delta_x
                    centralized_map[y, x, color] += img[y_npz, x_npz, color]
        count += 1   # Total # of img
    
    centralized_map /= count
    
    # Min-max scaling => (0,1)
    centralized_map = (centralized_map - np.min(centralized_map)) / np.max(centralized_map)
    
    '''
    # Visualize centralized map
    plt.figure()
    plt.imshow(centralized_map[:,:,:])
    
    # Visualize heatmap for refrence
    m = cm.ScalarMappable(cmap='jet')
    rgb_img = m.to_rgba(npz_map)[:,:,:3]
    plt.figure()
    plt.imshow(rgb_img)'''
    
    return centralized_map

centralized_map = centralization('', '', 120, 160)
plt.imsave('./centralized_map.jpg', centralized_map)