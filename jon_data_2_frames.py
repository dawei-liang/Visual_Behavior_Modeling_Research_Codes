# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 17:41:27 2018

@author: david
"""
'''To use ffmpeg: http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/'''

#%%

import cv2
import check_dirs
print('OpenCV version: ', cv2.__version__)

dir_to_load_video = './'
dir_to_save_frames = './frames1/'
check_dirs.check_dir(dir_to_save_frames)
pixel = {'x': 320, 'y': 240}   # Target size

cap = cv2.VideoCapture(dir_to_load_video + 'world1.mp4')   #Obtain video
ret, frame = cap.read()   #Capture each frame

count = 0
ret = True
while ret:
    frame = cv2.resize(frame, (pixel['x'], pixel['y']), interpolation =cv2.INTER_AREA)
    cv2.imwrite(dir_to_save_frames + "frame%d.jpg" % count, frame)     # save frames as JPEG files
    ret, frame = cap.read()
    print ('Read a new frame: ', count)
    count += 1
    
    
