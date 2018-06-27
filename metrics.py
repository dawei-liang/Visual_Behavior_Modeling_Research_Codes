# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 22:58:26 2018

@author: david
"""
import numpy as np
from scipy.stats import entropy
from sklearn import metrics


def computeNSS(saliency_map, gt_interest_points):
    if len(gt_interest_points) == 0:
        print ("Warning: No gaze data for this frame!")
        return 2.0

    stddev = np.std(saliency_map)
    if (stddev > 0):
        sal = (saliency_map - np.mean(saliency_map)) / stddev
    else:
        sal = saliency_map

    score = np.mean([ sal[y][x] for x,y in gt_interest_points ])
    return score

def computeCC(saliency_map, gt_saliency_map):
    saliency_map = saliency_map.flatten()
    gt_saliency_map = gt_saliency_map.flatten()

    if len(gt_saliency_map) == 0:
        return 1.0
    gt_sal = (gt_saliency_map - np.mean(gt_saliency_map)) / np.std(gt_saliency_map)
    stddev = np.std(saliency_map)
    if (stddev > 0):
        sal = (saliency_map - np.mean(saliency_map)) / stddev
        score = np.corrcoef([gt_sal, sal])[0][1]
    else:
        sal = saliency_map
        score = np.cov([gt_sal, sal])[0][1]

    return score

def computeKL(saliency_map, gt_saliency_map):
    epsilon = 1e-10
    saliency_map = np.clip(saliency_map.flatten(), epsilon, 1)
    gt_saliency_map = np.clip(gt_saliency_map.flatten(), epsilon, 1)

    return entropy(gt_saliency_map, saliency_map)

def computeAUC(saliency_map, fixationmap_gt):
    fixationmap_gt = np.clip(fixationmap_gt, 0, 1)
    fpr, tpr, thresholds = metrics.roc_curve(fixationmap_gt.flatten(), saliency_map.flatten())
    return metrics.auc(fpr, tpr)
    
#%%
#a = np.random.randint(0,2,(1080,1920))
#c=[[500,500],[1000,1000]]   # gt gaze list
#score = computeNSS(a, pos)
#score = computeNSS(binarized_map, c)
#print(score)

#a = np.random.randint(0,2,(1920,1080))
##score = computeAUC(heatmap, binarized_map)
#temp=np.zeros((1080,1920))
#for i in range(len(gaze_pos)):
#    temp[gaze_pos[i][1], gaze_pos[i][0]] = 1
#score = computeAUC(saliency_map, temp)
#print(score)

#b=np.random.rand(1920,1080)
##score = computeCC(b, heatmap)
#score = computeCC(saliency_map, heatmap)
#print(score)

#score = computeKL(b, heatmap)
##score = computeKL(saliency_map, heatmap)
#print(score)

#%%
#with open('./frames_validation/log.txt') as f:
#    a = f.readlines()
#a=a[101].split(' ')[1:]
#pos = []
#i=0
#while i < len(a) - 1:
#    pos.append(list((a[i],a[i+1])))
#    i += 2
#
#b=np.load('./new/heatmap2.npz')['a']
#print(b.shape)
#
#temp=np.zeros((1080,1920))
#for i in range(len(pos)):
#    pos[i][0] = int(float(pos[i][0]) * pixel['x'])    # x
#    pos[i][1] = int(pixel['y'] - float(pos[i][1]) * pixel['y'])   # y
#    temp[pos[i][1], pos[i][0]] = 1
#score = computeAUC(saliency_map, temp)
#print(score)


