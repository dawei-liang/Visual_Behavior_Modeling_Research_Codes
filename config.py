# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 22:45:33 2018

@author: david
"""
#%%

dir_to_load_frames = './frames1/'
dir_to_save_groundtruth_log = './'
dir_to_save_log = './itti_related_results/'
dir_to_save_groundtruth = './frames_groundtruth_1/'
dir_to_save_frames_in_use = './frames_in_use_1/'
dir_to_save_heatmap = './groundtruth_heatmap_1/'
dir_to_save_saliency_map = './predicted_saliency_map_intensity_only/'
dir_to_save_antisaliency_map = './predicted_antisaliency_map_intensity_only/'
dir_to_save_chance_map = './predicted_chance_map_intensity_only/'

dir_to_load_frames_for_video = dir_to_save_groundtruth
dir_write_video = dir_to_save_groundtruth

pixel = {'x': 320, 'y': 240}
variance_x = 3.055
variance_y = 4.073

groundtruth_file = './gaze_positions.csv'
groundtruth_file_sub2 = './por.mat'


