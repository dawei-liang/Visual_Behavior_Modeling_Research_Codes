# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:41:27 2018

@author: david
"""

#%%

import cv2

def resize(image, width, height):
    new_frame = cv2.resize(image, (width, height), interpolation =cv2.INTER_AREA)
    return new_frame
