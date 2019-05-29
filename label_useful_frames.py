# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:31:06 2018

@author: david
"""

import os
import config
import shutil
import check_dirs
import scipy.io as sio
import numpy as np

import read_frameInfo

matfn = config.groundtruth_file
data=sio.loadmat(matfn)
fix_index = data['fixation_frames']
fix_index = fix_index.astype(np.int32)
subject = 1   # Can be 1,2,3

check_dirs.check_dir(config.dir_to_save_frames_in_use)    
frame_sets = [x for x in os.listdir(config.dir_to_load_frames) if x.endswith('.jpg')]   # Load frames
print('Number of frames:', len(frame_sets))

# Import frame idx
subFrames, _, _ = read_frameInfo.frameInfo('./frameInfo.mat', subject=subject)   

for frame in frame_sets:
    index = int(frame.strip('frame').strip('.jpg'))   # Get loaded frame index
    # Range of frames in use
    if index in subFrames and fix_index[index] == 1:
        shutil.copy(config.dir_to_load_frames + frame, config.dir_to_save_frames_in_use) 
        try:
            os.rename(config.dir_to_save_frames_in_use + frame, 
                          config.dir_to_save_frames_in_use + frame.strip('.jpg') + '.jpg')   # Rename
            print('# of renamed frame index:', index)
        except:
            continue
  
