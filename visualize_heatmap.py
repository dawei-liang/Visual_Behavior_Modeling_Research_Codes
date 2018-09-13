# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 23:55:30 2018

@author: david
"""
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

dir_load_npz = 'G:/Research2/sem2 w1/groundtruth_heatmap_1/heatmap4243.npz'
key = 'heatmap'

print('Converting npz to jpgs..')  
# npz => jpg Conversion
m = cm.ScalarMappable(cmap='jet')
npz_img = np.load(dir_load_npz)[key]
rgb_img = m.to_rgba(npz_img)[:,:,:3]
plt.imshow(rgb_img)