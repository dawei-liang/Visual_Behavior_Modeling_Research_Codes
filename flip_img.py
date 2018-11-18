# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 17:54:19 2018

@author: david
"""

import config
import check_dirs
import os
import numpy as np
import cv2

# Check and create necessary dir
dir_groundtruth = config.dir_to_save_groundtruth
check_dirs.check_dir(dir_groundtruth[:-1]+'_flip/')

dir_heatmap = config.dir_to_save_heatmap
check_dirs.check_dir(dir_heatmap[:-1]+'_flip/')

dir_img = config.dir_to_save_frames_in_use
check_dirs.check_dir(dir_img[:-1]+'_flip/')

frame_sets = [x for x in os.listdir(dir_img) if x.endswith('.jpg')]   # Load frames
print('Number of frames:', len(frame_sets))

heatmap_sets = [x for x in os.listdir(dir_heatmap) if x.endswith('.npz')]   # Load heatmaps
print('Number of heatmaps:', len(heatmap_sets))

groundtruth_sets = [x for x in os.listdir(dir_groundtruth) if x.endswith('.jpg')]   # Load groundtruth
print('Number of groundtruth:', len(groundtruth_sets))

for i in range(len(frame_sets)):
    index = int(frame_sets[i].strip('frame').strip('.jpg'))   # Get loaded frame index
    frame = np.fliplr(cv2.imread(dir_img + frame_sets[i]))
    # Save plotted frames
    cv2.imwrite(dir_img[:-1] + '_flip/' + 'frame%s.jpg' % index, frame)
    print('saving frame %d' %index)
    
for i in range(len(heatmap_sets)):
    index = int(heatmap_sets[i].strip('heatmap').strip('.npz'))   # Get loaded frame index
    heatmap = np.fliplr(np.load(dir_heatmap + heatmap_sets[i])['heatmap'])
    # Save plotted frames
    np.savez(dir_heatmap[:-1] + '_flip/' + 'heatmap%s.npz' % index, heatmap = heatmap)
    print('saving heatmap %d' %index)
    
for i in range(len(groundtruth_sets)):
    index = int(groundtruth_sets[i].strip('frame').strip('.jpg'))   # Get loaded frame index
    groundtruth = np.fliplr(cv2.imread(dir_groundtruth + groundtruth_sets[i]))
    # Save plotted frames
    cv2.imwrite(dir_groundtruth[:-1] + '_flip/' + 'frame%s.jpg' % index, groundtruth)
    print('saving groundtruth %d' %index)
    