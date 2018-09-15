# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:31:06 2018

@author: david
"""

import os
import config
import shutil
import check_dirs


check_dirs.check_dir(config.dir_to_save_frames_in_use)    
frame_sets = [x for x in os.listdir(config.dir_to_load_frames) if x.endswith('.jpg')]   # Load frames
print('Number of frames:', len(frame_sets))

for frame in frame_sets:
    index = int(frame.strip('frame').strip('.jpg'))   # Get loaded frame index
    # Range of frames in use
    #'''Sub 1'''
    #if (index >= 4240 and index < 5308) or \
    #    (index >= 5332 and index < 6363) or \
    #    (index >= 6701 and index < 7778) or \
    #    (index >= 7814 and index < 8917) or \
    #    (index >= 9329 and index < 11444) or \
    #    (index >= 11678 and index < 13857):
    #    shutil.copy(config.dir_to_load_frames + frame, config.dir_to_save_frames_in_use) 
    #    try:
    #        os.rename(config.dir_to_save_frames_in_use + frame, 
    #                      config.dir_to_save_frames_in_use + frame.strip('.jpg') + '.jpg')   # Rename
    #        print('# of renamed frame index:', index)
    #    except:
    #        continue

    '''Sub 2'''
    if (index >= 2530 and index < 4361) or \
        (index >= 4481 and index < 6191) or \
        (index >= 6445 and index < 8075) or \
        (index >= 8195 and index < 9960) or \
        (index >= 10174 and index < 11896) or \
        (index >= 11950 and index < 13712):
        shutil.copy(config.dir_to_load_frames + frame, config.dir_to_save_frames_in_use) 
        try:
            os.rename(config.dir_to_save_frames_in_use + frame, 
                          config.dir_to_save_frames_in_use + frame.strip('.jpg') + '.jpg')   # Rename
            print('# of renamed frame index:', index)
        except:
            continue
  
