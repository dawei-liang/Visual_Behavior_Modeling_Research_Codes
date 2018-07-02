# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 00:18:58 2018

@author: david
"""

import metrics
import numpy as np
import config
import os
import itti_model

#%%
dir_to_load_heatmap = config.dir_to_save_heatmap
dir_to_load_log = config.dir_to_save_log
dir_to_load_frames = config.dir_to_load_frames
y_pos, x_pos = config.pixel['y'], config.pixel['x']


if __name__ == '__main__':    
    '''Load frames'''
    frame_sets = [x for x in os.listdir(dir_to_load_frames) if x.endswith('.jpg')]   # Load frames   
    print('Number of frames:', len(frame_sets))
    with open(dir_to_load_log + 'log.txt') as f:   # Load gaze pos
        gaze = f.readlines()
    
    NSS_score, chance_NSS_score = 0, 0
    AUC_score, chance_AUC_score = 0, 0
    KL_score, chance_KL_score = 0, 0
    CC_score, chance_CC_score = 0, 0
    count = 0
    
    for frame in frame_sets:
        index = int(frame.strip('frame').strip('.jpg'))   # Get loaded frame index
        # Frames to test
        if index >= config.frame_range['lower'] and index < config.frame_range['upper']:   
            print('frame index:', index)            
            # Generate saliency map
            itti_model_object = itti_model.itti_model(dir_to_load_frames + frame)
            itti_model_object.saliency_map()
            saliency_map = itti_model_object.saliency_map
            anti_saliency_map = np.ones((saliency_map.shape[0],saliency_map.shape[1])) - saliency_map
            # Create a gaze position list and gaze map
            gaze_line = gaze[index-4500+1].split(' ')[1:]   # Extract lines of gaze from the txt
            pos = []   # Gaze position list
            i=0
            # Convert txt gaze lines to pairs of coordinates
            while i < len(gaze_line) - 1:
                pos.append(list((gaze_line[i], gaze_line[i+1])))   # gaze_line[i], gaze_line[i+1]= x,y
                i += 2
            gaze_map = np.zeros((y_pos, x_pos))
            for i in range(len(pos)):
                pos[i][0] = int(pos[i][0])    
                pos[i][1] = int(pos[i][1])   # Convert to int
                gaze_map[pos[i][1], pos[i][0]] = 1
            
            # Compute      
            # NSS    
            NSS_score += metrics.computeNSS(saliency_map, pos)
            # AUC
            AUC_score += metrics.computeAUC(saliency_map, gaze_map)
            # CC
            heatmap = np.load(dir_to_load_heatmap + 'heatmap' + str(index) + '.npz')['heatmap']
            CC_score += metrics.computeCC(saliency_map, heatmap)
            # KL
            KL_score = metrics.computeKL(saliency_map, heatmap)
            count += 1
            
    NSS_score /= count
    print('NSS:', NSS_score)
    AUC_score/= count
    print('AUC:', AUC_score)
    CC_score/= count
    print('CC:', CC_score)
    KL_score/= count
    print('KL:', KL_score)
    
    # Reference
    chance_prediction = np.random.rand(y_pos, x_pos)
    # NSS
    chance_NSS_score = metrics.computeNSS(chance_prediction, pos)
    # AUC
    chance_AUC_score = metrics.computeAUC(chance_prediction, gaze_map)
    # CC
    chance_CC_score = metrics.computeCC(chance_prediction, heatmap)
    # KL
    chance_KL_score = metrics.computeKL(chance_prediction, heatmap)
    print('chance NSS, AUC, CC, KL: %f %f %f %f' 
          %(chance_NSS_score, chance_AUC_score, chance_CC_score, chance_KL_score))
    
    # Save computed results
    results_log = open(config.dir_to_save_log +"/results.txt", 'w')
    results_log.write('frame range' + \
                      str(config.frame_range['lower']) + ' ' + str(config.frame_range['upper']) + \
                      '\n' + 'NSS,AUC,CC,KL' + '\n'+ \
                      str(NSS_score) + ' ' + \
                      str(AUC_score) + ' ' + \
                      str(CC_score) + ' ' + \
                      str(KL_score))
    results_log.close()
    
