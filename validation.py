# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:29:34 2018

@author: david
"""
import cv2
import pandas as pd
import scipy.io as sio
import os
import numpy as np
import copy as cp
import config
import video
import heatmap
import check_dirs
import randomly_crop_image
import read_frameInfo
import resize_patch_for_cnn
    
#%%
'''Visualize groundtruth coordinates and generate groundtruth videos/heatmaps'''

if __name__ == '__main__':
    subject = 3
    check_dirs.check_dir(config.dir_to_save_groundtruth)
    check_dirs.check_dir(config.dir_to_save_heatmap)
    check_dirs.check_dir(config.dir_write_video)
    check_dirs.check_dir(config.dir_to_save_crop_images)
    groundtruth_log = open(config.dir_to_save_groundtruth_log +"/log"+str(subject)+".txt", 'w')
    patch_width, patch_height = 105, 80   # Target cropped patch size
    resize_for_cnn = [160, 120]    # Width, height, resize patches for CNN so that the proposed CNN can fit all training patches

    ''' Import frame idx '''
    subFrames, _, _ = read_frameInfo.frameInfo('./frameInfo.mat', subject=subject)
    
    '''Import groundtruth coordinates and frame indeces from csv'''
    matfn = config.groundtruth_file
    data=sio.loadmat(matfn)
    norm_pos_x = data['new_porX'] * 1.0 / 1920
    norm_pos_y = data['new_porY'] * 1.0 / 1080
    fix_index = data['fixation_frames']
    norm_pos_x = norm_pos_x
    norm_pos_y = norm_pos_y
    fix_index = fix_index.astype(np.int32)
    heatmap_object = heatmap.heatmap(resize_for_cnn[0], resize_for_cnn[1])
    
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
        # Frames to use
        if index in subFrames and fix_index[index] == 1:
            groundtruth_log.write('\n' + str(index))
            print('frame:', index)               
            img = cv2.imread(config.dir_to_load_frames + frame)   # Load the frame to visualize 
            raw_Gaussian_map = np.zeros((resize_for_cnn[1], resize_for_cnn[0]))   # Set heatmap size
            
            # Loop for matched gaze
            center = (int(norm_pos_x[index] * config.pixel['x']), 
                      int(norm_pos_y[index] * config.pixel['y'])) # Gaze points, x:right;y:down
            # crop image and return new gaze positions
            patch, gaze_horizontal_new, gaze_vertical_new = randomly_crop_image.random_crop(img, config.pixel['x'], config.pixel['y'], patch_width, patch_height, center[0], center[1])
            if patch is not None:
                # Rescale patches for cnn training
                patch = resize_patch_for_cnn.resize(patch, resize_for_cnn[0], resize_for_cnn[1])
                gaze_horizontal_new = int(resize_for_cnn[0] / float(patch_width) * gaze_horizontal_new)
                gaze_vertical_new = int(resize_for_cnn[1] / float(patch_height) * gaze_vertical_new)
                # Write groundtruth to txt
                groundtruth_log.write(' ' + str(gaze_horizontal_new) + ' ' + str(gaze_vertical_new))   
                # Update Gaussian map
                raw_Gaussian_map = np.dstack((raw_Gaussian_map,
                                                  heatmap_object.generate_Gaussian_map(gaze_horizontal_new, gaze_vertical_new, 
											config.variance_x, config.variance_y)))
                # Plot a cross line on frame
                img2 = cp.deepcopy(patch)
                cv2.line(img2,(gaze_horizontal_new-5, gaze_vertical_new), (gaze_horizontal_new+5, gaze_vertical_new), (0,0,255), 3)
                cv2.line(img2,(gaze_horizontal_new, gaze_vertical_new-5), (gaze_horizontal_new, gaze_vertical_new+5), (0,0,255), 3)
                #cv2.circle(img, center, radius=10, color=(,0,255), 
                     #                thickness=2, lineType=8, shift=0)

                # Save plotted frames
                cv2.imwrite(config.dir_to_save_groundtruth + 'frame%s.jpg' % index, img2)   
                # Convert Gaussian map to heatmap using 'maximum pixels'
                heatmap = np.amax(raw_Gaussian_map, axis=2)
                # Save as npz            
                np.savez(config.dir_to_save_heatmap + 'heatmap%s' % index, heatmap = heatmap)
                # Save new gaze patch
                cv2.imwrite(config.dir_to_save_crop_images + 'frame%s.jpg' % index, patch)     # save cropped frames as JPEG files
            
    groundtruth_log.close()
    
    ''' Generate a groundtruth video '''
    video_object = video.video(config.dir_to_load_frames_for_video, config.dir_write_video)
    video_object.to_video()
