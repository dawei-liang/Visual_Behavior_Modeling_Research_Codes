# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 23:10:12 2018

@author: david
"""
import cv2
import os


#%%
'''Form frames as a video'''

class video:
    def __init__(self, dir_to_load_frames_for_video, dir_write_video):
        self.dir_to_load_frames_for_video = dir_to_load_frames_for_video
        self.dir_write_video = dir_write_video
        
    def to_video(self):
        print('OpenCV version: ', cv2.__version__)
        
        frame_video_sets = [x for x in os.listdir(self.dir_to_load_frames_for_video) 
                            if x.endswith('.jpg')]   # Load frames
        print('Number of frames to video:', len(frame_video_sets))
        
        img_video = cv2.imread(self.dir_to_load_frames_for_video + frame_video_sets[0])   # Set video parameters
        height, width, layers =  img_video.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # (Be sure to use lower case)
        video = cv2.VideoWriter(self.dir_write_video + 'validation_video.mp4', fourcc, 30.0, (width, height))       
        print('height, width, layers:', height, width, layers)
        
        for frame_video in frame_video_sets:
            img_video = cv2.imread(self.dir_to_load_frames_for_video + frame_video)
            video.write(img_video)     
        cv2.destroyAllWindows()
        video.release()
        print('Video written')