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
import heatmap
import check_dirs

#%%
dir_to_load_heatmap = config.dir_to_save_heatmap
dir_to_load_log = config.dir_to_save_log
dir_to_load_frames = config.dir_to_load_frames
y_pos, x_pos = config.pixel['y'], config.pixel['x']


if __name__ == '__main__':
    check_dirs.check_dir(config.dir_to_save_log)
    NSS_itti = open(config.dir_to_save_log +"/NSS_itti.txt", 'w')
    AUC_itti = open(config.dir_to_save_log +"/AUC_itti.txt", 'w')
    CC_itti = open(config.dir_to_save_log +"/CC_itti.txt", 'w')
    KL_itti = open(config.dir_to_save_log +"/KL_itti.txt", 'w') 
    NSS_itti.write('index' + ' ' + 'saliency' + ' ' + 'chance' + ' ' + 'anti_saliency' + '\n')
    AUC_itti.write('index' + ' ' + 'saliency' + ' ' + 'chance' + ' ' + 'anti_saliency' + '\n')
    CC_itti.write('index' + ' ' + 'saliency' + ' ' + 'chance' + ' ' + 'anti_saliency' + '\n')
    KL_itti.write('index' + ' ' + 'saliency' + ' ' + 'chance' + ' ' + 'anti_saliency' + '\n')
    
    '''Load frames'''
    frame_sets = [x for x in os.listdir(dir_to_load_frames) if x.endswith('.jpg')]   # Load frames   
    print('Number of frames:', len(frame_sets))
    with open(dir_to_load_log + 'log.txt') as f:   # Load gaze pos
        gaze = f.readlines()
    
    NSS_score, chance_NSS_score, anti_NSS_score = 0, 0, 0
    AUC_score, chance_AUC_score, anti_AUC_score = 0, 0, 0
    KL_score, chance_KL_score, anti_KL_score = 0, 0, 0
    CC_score, chance_CC_score, anti_CC_score = 0, 0, 0
    count = 0
    
    for frame in frame_sets:
        index = int(frame.strip('frame').strip('.jpg'))   # Get loaded frame index
        # Frames to test
        if (index >= 4240 and index < 5308) or \
        (index >= 5332 and index < 6363) or \
        (index >= 6701 and index < 7778) or \
        (index >= 7814 and index < 8917) or \
        (index >= 9329 and index < 11444) or \
        (index >= 11678 and index < 13857):   
            print('frame index:', index)            
            # Generate saliency/anti_saliency map
            itti_model_object = itti_model.itti_model(dir_to_load_frames + frame)                       
            itti_model_object.saliency_map()         
            saliency_map = itti_model_object.saliency_map
            anti_saliency_map = itti_model_object.anti_saliency_map
            chance_prediction = np.random.rand(y_pos, x_pos)
            # Normalize as distribution
            heatmap_object = heatmap.heatmap(x_pos, y_pos)
            saliency_map = heatmap_object.normalize(saliency_map)
            anti_saliency_map = heatmap_object.normalize(anti_saliency_map)
            chance_prediction = heatmap_object.normalize(chance_prediction)
            # Create a gaze position list and gaze map
            gaze_line = gaze[1].split(' ')[1:]   # Extract lines of gaze from the txt
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
            
            # Compute and save 
            # Groundtruth heatmaps
            heatmap_truth = np.load(dir_to_load_heatmap + 'heatmap' + str(index) + '.npz')['heatmap']  
            # NSS
            temp_NSS = metrics.computeNSS(saliency_map, pos)
            temp_chance_NSS = metrics.computeNSS(chance_prediction, pos)
            temp_anti_NSS = metrics.computeNSS(anti_saliency_map, pos)
            NSS_itti.write(str(index) + ' ' + 
                           str(temp_NSS) + ' ' + 
                           str(temp_chance_NSS) + ' ' + 
                           str(temp_anti_NSS) + '\n')
            NSS_score += temp_NSS
            chance_NSS_score +=  temp_chance_NSS
            anti_NSS_score += temp_anti_NSS
            # AUC
            temp_AUC = metrics.computeAUC(saliency_map, gaze_map)
            temp_chance_AUC = metrics.computeAUC(chance_prediction, gaze_map)
            temp_anti_AUC = metrics.computeAUC(anti_saliency_map, gaze_map)
            AUC_itti.write(str(index) + ' ' + 
                           str(temp_AUC) + ' ' + 
                           str(temp_chance_AUC) + ' ' + 
                           str(temp_anti_AUC) + '\n')
            AUC_score += temp_AUC
            chance_AUC_score +=  temp_chance_AUC
            anti_AUC_score += temp_anti_AUC
            # CC
            temp_CC = metrics.computeCC(saliency_map, heatmap_truth)
            temp_chance_CC = metrics.computeCC(chance_prediction, heatmap_truth)
            temp_anti_CC = metrics.computeCC(anti_saliency_map, heatmap_truth)
            CC_itti.write(str(index) + ' ' + 
                           str(temp_CC) + ' ' + 
                           str(temp_chance_CC) + ' ' + 
                           str(temp_anti_CC) + '\n')
            CC_score += temp_CC
            chance_CC_score +=  temp_chance_CC
            anti_CC_score += temp_anti_CC
            # KL
            temp_KL = metrics.computeKL(saliency_map, heatmap_truth)
            temp_chance_KL = metrics.computeKL(chance_prediction, heatmap_truth)
            temp_anti_KL = metrics.computeKL(anti_saliency_map, heatmap_truth)
            KL_itti.write(str(index) + ' ' + 
                           str(temp_KL) + ' ' + 
                           str(temp_chance_KL) + ' ' + 
                           str(temp_anti_KL) + '\n')
            KL_score += temp_KL
            chance_KL_score +=  temp_chance_KL
            anti_KL_score += temp_anti_KL
            
            count += 1
            
    NSS_score /= count
    chance_NSS_score /=  count
    anti_NSS_score /= count
    NSS_itti.write('Overall' + ' ' + 
                   str(NSS_score) + ' ' + 
                   str(chance_NSS_score) + ' ' + 
                   str(anti_NSS_score))
    print('NSS,chance,anti_NSS: %f %f %f' %(NSS_score,chance_NSS_score,anti_NSS_score))
    
    AUC_score /= count
    chance_AUC_score /=  count
    anti_AUC_score /= count
    AUC_itti.write('Overall' + ' ' + 
                   str(AUC_score) + ' ' + 
                   str(chance_AUC_score) + ' ' + 
                   str(anti_AUC_score))
    print('AUC,chance,anti_AUC: %f %f %f' %(AUC_score,chance_AUC_score,anti_AUC_score))
    
    CC_score /= count
    chance_CC_score /=  count
    anti_CC_score /= count
    CC_itti.write('Overall' + ' ' + 
                   str(CC_score) + ' ' + 
                   str(chance_CC_score) + ' ' + 
                   str(anti_CC_score))
    print('CC,chance,anti_CC: %f %f %f' %(CC_score,chance_CC_score,anti_CC_score))
    
    KL_score/= count
    chance_KL_score /=  count
    anti_KL_score /= count
    KL_itti.write('Overall' + ' ' + 
                   str(KL_score) + ' ' + 
                   str(chance_KL_score) + ' ' + 
                   str(anti_KL_score))
    print('KL,chance,anti_KL: %f %f %f' %(KL_score,chance_KL_score,anti_KL_score))
    
    NSS_itti.close()
    AUC_itti.close()
    CC_itti.close()
    KL_itti.close()
    

    
    