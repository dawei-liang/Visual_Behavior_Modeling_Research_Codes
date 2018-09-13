# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:29:34 2018

@author: david
"""
import cv2
import pandas as pd
import os
import numpy as np
import config
import video
import heatmap
import check_dirs
    
#%%
'''Visualize groundtruth coordinates and generate groundtruth videos/heatmaps'''

if __name__ == '__main__':
    check_dirs.check_dir(config.dir_to_save_groundtruth)
    check_dirs.check_dir(config.dir_to_save_heatmap)
    check_dirs.check_dir(config.dir_write_video)
    groundtruth_log = open(config.dir_to_save_groundtruth_log +"/log1.txt", 'w')
    
    '''Import groundtruth coordinates and frame indeces from csv'''
    truth_df = pd.read_csv(config.groudtruth_file)
    norm_pos_x = truth_df['norm_pos_x'].values
    norm_pos_y = truth_df['norm_pos_y'].values
    frame_idx = truth_df['index'].values
    heatmap_object = heatmap.heatmap(config.pixel['x'], config.pixel['y'])
    
    '''Generate and save groundtruth frames/heatmap/txt'''
    frame_sets = [x for x in os.listdir(config.dir_to_load_frames) if x.endswith('.jpg')]   # Load frames   
    print('Number of frames:', len(frame_sets))

    # Sort file name by frame index
    for i in range(len(frame_sets)):
        frame_sets[i] = int(frame_sets[i].strip('frame').strip('.jpg'))
    frame_sets.sort()
    for i in range(len(frame_sets)):
	frame_sets[i] = 'frame' + str(frame_sets[i]) + '.jpg' 
    
    for frame in frame_sets:
        index = int(frame.strip('frame').strip('.jpg'))   # Get loaded frame index
        # Frames to test
        if (index >= 4240 and index < 5308) or \
        (index >= 5332 and index < 6363) or \
        (index >= 6701 and index < 7778) or \
        (index >= 7814 and index < 8917) or \
        (index >= 9329 and index < 11444) or \
        (index >= 11678 and index < 13857):
            groundtruth_log.write('\n' + str(index))
            print('frame:', index)               
            img = cv2.imread(config.dir_to_load_frames + frame)   # Load the frame to visualize 
            raw_Gaussian_map = np.zeros((config.pixel['y'], config.pixel['x']))   # Set heatmap size
            
            # Loop for matched gaze
            for i in range(len(frame_idx)): 
                # Exist a gaze pos for the frame
                if index == frame_idx[i]:     
                    center = (int(norm_pos_x[i] * config.pixel['x']), 
                              int(config.pixel['y'] - norm_pos_y[i] * config.pixel['y']))   # Gaze points, x:right;y:down
                    # Write groundtruth to txt
                    groundtruth_log.write(' ' + str(center[0]) + ' ' + str(center[1]))   
                    # Update Gaussian map
                    raw_Gaussian_map = np.dstack((raw_Gaussian_map,
                                                  heatmap_object.generate_Gaussian_map(center[0], center[1], 
											config.variance_x, config.variance_y)))
                    # Plot a red circle on frame
                    cv2.circle(img, center, radius=5, color=(0,0,255), 
                                             thickness=2, lineType=8, shift=0)
            # Save plotted frames
            cv2.imwrite(config.dir_to_save_groundtruth + 'frame%s.jpg' % index, img)   
            # Convert Gaussian map to heatmap using 'maximum pixels'
            heatmap = np.amax(raw_Gaussian_map, axis=2)
            # Save as npz            
            np.savez(config.dir_to_save_heatmap + 'heatmap%s' % index, heatmap = heatmap)
            
    groundtruth_log.close()
    
    ''' Generate a groundtruth video '''
    video_object = video.video(config.dir_to_load_frames_for_video, config.dir_write_video)
    video_object.to_video()
