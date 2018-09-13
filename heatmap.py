# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 23:13:26 2018

@author: david
"""
import numpy as np

#%%
'''Generate heatmap'''

class heatmap:
    def __init__(self, img_x, img_y):
        self.img_x = img_x
        self.img_y = img_y
        
    def generate_Gaussian_map(self, gaze_x, gaze_y, variance_x, variance_y):
        self.gaze_x = gaze_x
        self.gaze_y = gaze_y
        self.variance_x = variance_x
	self.variance_y = variance_y
        
        gaussian_map = np.zeros((self.img_y, self.img_x))
        for x_p in range(self.img_x):
            for y_p in range(self.img_y):
                dist_sq = (x_p - self.gaze_x) * (x_p - self.gaze_x) + \
                            (y_p - self.gaze_y) * (y_p - self.gaze_y)   # Distance from gaze point
                exponent = dist_sq / 2.0 / self.variance_x / self.variance_y
                gaussian_map[y_p, x_p] = np.exp(-exponent)   # Gaussian distribution
        return gaussian_map
        
    def normalize(self, raw_heat_map):   
        sum_of_hm = raw_heat_map.sum()
        if sum_of_hm != 0:
            heatmap = raw_heat_map / sum_of_hm
            return heatmap
        else:
            return raw_heat_map
