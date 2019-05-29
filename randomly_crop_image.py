# -*- coding: utf-8 -*-
"""
Created on Mon Apr 8 22:45:33 2019

@author: david
"""
#%%

import numpy as np

def random_crop(frame, width, height, patch_width, patch_height, gaze_horizontal, gaze_vertical):
    # Select the top-left corner for cropping
    left_point = np.random.randint(0, width - patch_width)
    up_point = np.random.randint(0, height - patch_height)
    # If not including a gaze in the crop, select again; set threshold: 30 loops
    while left_point + patch_width < gaze_horizontal or left_point > gaze_horizontal or up_point + patch_height < gaze_vertical or up_point > gaze_vertical:
        left_point = np.random.randint(0, width - patch_width)
        up_point = np.random.randint(0, height - patch_height) 
        # remove gaze outliers
        if gaze_horizontal == 0 or gaze_horizontal == width or gaze_vertical == 0 or gaze_vertical == height:
            return None, gaze_horizontal, gaze_vertical          
    # A well-satisfied image boundary:
    # Bound as an image patch
    down_point, right_point = up_point + patch_height, left_point + patch_width
    patch = frame[up_point:down_point, left_point:right_point, :]
    # calculate new gaze positions with respect to new image patch
    gaze_horizontal_new = gaze_horizontal - left_point
    gaze_vertical_new = gaze_vertical - up_point

    return patch, gaze_horizontal_new, gaze_vertical_new

