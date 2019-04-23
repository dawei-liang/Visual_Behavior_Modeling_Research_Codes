# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:05:53 2019

@author: david
"""

import scipy.io as sio
import numpy as np

def frameInfo(matfile, subject):
    '''Return an array of frame index in use for specified subject.'''
    
    frame_info = sio.loadmat(matfile)
    frameNumbers = frame_info['frameNumbers']
    subjKey = frame_info['subjKey']
    walkKey = frame_info['walkKey']
    frameNumbers, subjKey, walkKey = frameNumbers.astype(np.int), subjKey.astype(np.int), walkKey.astype(np.int)
    
    subFrames = []
    test_idx = []
    assert len(frameNumbers) == len(subjKey), print('subject key size is different from frame size, please check mat file!')
    assert len(walkKey) == len(subjKey), print('subject key size is different from walkkey size, please check mat file!')
    
    for key in range(len(subjKey)):
        if subjKey[key] == subject:
            subFrames.append(frameNumbers[key])
            if walkKey[key] == 3 or walkKey[key] == 4:   # walkKey 3 and 4 are used for testing
                test_idx.append(key)
    
    # Frame idx for specified subject
    subFrames = np.asarray(subFrames, dtype=int)
    subFrames = np.reshape(subFrames, (subFrames.shape[0], subFrames.shape[1]))
    # test idx
    test_idx_start, test_idx_end = test_idx[0], test_idx[-1]
    return subFrames, test_idx_start, test_idx_end

