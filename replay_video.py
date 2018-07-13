#!/usr/bin/env python
# Author: Zhuode Liu
# Modified by: Dawei Liang

# Visualize prediction using saved ground truth frames and converted heatmap JPG files 
# (JPG heatmaps converted from npz)

import sys, pygame, time, os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import check_dirs

inital_sample_img = './frames_groundtruth/frame4240.jpg'
w, h = 1024, 576

dir_ground_truth = './frames_groundtruth/'   # dir of ground truth frames

convert_npz = True   # Whether to convert npz heatmaps to jpg files first
dir_load_npz_saliency_map = 'G:/Research2/w5/predicted_antisaliency_map/'   # For npz => jpg conversion
key = 'antisaliency'   # key of npz files: 'saliency', 'antisaliency', 'chance'
dir_save_png_map = "G:/Research2/w5/rgb_saliency_maps/"   # dir of jpg heatmaps

#%%


class drawgc_wrapper:
    def __init__(self):
        self.cursor = pygame.image.load(inital_sample_img)
        self.cursorsize = (self.cursor.get_width(), self.cursor.get_height())

    def draw_gc(self, screen, gaze_position):
        '''draw the gaze-contingent window on screen '''
        region_topleft = (gaze_position[0] - self.cursorsize[0] // 2, gaze_position[1] - self.cursorsize[1] // 2)
        screen.blit(self.cursor, region_topleft) # Draws and shows the cursor content;

class DrawStatus:
    draw_many_gazes = True
    cur_frame_id = 1
    total_frame = None
    target_fps = 30
    pause = False
ds = DrawStatus()

def event_handler_func():
    global ds

    for event in pygame.event.get() :
      if event.type == pygame.KEYDOWN :   
        if event.key == pygame.constants.K_UP:   #up & down
            print ("Fast-backward 5 seconds")
            ds.cur_frame_id -= 5 * ds.target_fps
        elif event.key == pygame.constants.K_DOWN:  
            print ("Fast-forward 5 seconds")
            ds.cur_frame_id += 5 * ds.target_fps
        if event.key == pygame.constants.K_LEFT:   #left & right
            print ("Moving to previous frame")
            ds.cur_frame_id -= 1
        elif event.key == pygame.constants.K_RIGHT:
            print ("Moving to next frame")
            ds.cur_frame_id += 1
        elif event.key == pygame.constants.K_F3:   #F3
            p = float(pygame.constants.raw_input("Seeking through the video. Enter a percentage in float: "))
            ds.cur_frame_id = int(p/100*ds.total_frame)
        elif event.key == pygame.constants.K_SPACE:   #Space
            ds.pause = not ds.pause
        elif event.key == pygame.constants.K_F9:   #F9
            ds.draw_many_gazes = not ds.draw_many_gazes
            print ("draw all gazes belonging to a frame: %s" % ("ON" if ds.draw_many_gazes else "OFF"))
        elif event.key == pygame.constants.K_F11:   #F11
            ds.target_fps -= 2
            print ("Setting target FPS to %d" % ds.target_fps)
        elif event.key == pygame.constants.K_F12:   #F12
            ds.target_fps += 2
            print ("Setting target FPS to %d" % ds.target_fps)
    ds.cur_frame_id = max(0,min(ds.cur_frame_id, ds.total_frame))
    ds.target_fps = max(1, ds.target_fps)

#%%
    
if __name__ == "__main__":
    # Loop over ground truth frames, check overall info
    truth_files = [x for x in os.listdir(dir_ground_truth) if x.endswith('.jpg')]   
    # Sort file name by frame index
    for i in range(len(truth_files)):
        truth_files[i] = int(truth_files[i].strip('frame').strip('.jpg'))
    truth_files.sort()
    for i in range(len(truth_files)):
        truth_files[i] = 'frame' + str(truth_files[i]) + '.jpg'
        
    print('Number of frames:', len(truth_files))
    print ("\nYou can control the replay using keyboard. Try pressing space/up/down/left/right.") 
    print ("For all available keys, see event_handler_func() code.\n")
    

    # init pygame and other stuffs
    pygame.init()
    pygame.display.set_mode((w, h), pygame.constants.RESIZABLE | pygame.constants.DOUBLEBUF | pygame.constants.RLEACCEL, 32)
    screen = pygame.display.get_surface()
    print ("Reading gaze data...")
    dw = drawgc_wrapper()
    ds.target_fps = 30
    ds.total_frame = len(truth_files)

    last_time = time.time()
    clock = pygame.time.Clock()
    
    # Whether to convert npz saliency maps first
    if convert_npz:
        print('Converting npz to jpgs..')
        # Check if target dir exits
        check_dirs.check_dir(dir_save_png_map)   
        # npz => jpg Conversion
        temp_id = ds.cur_frame_id - 1       
        while temp_id < ds.total_frame:
            print('converting frame:', truth_files[temp_id-1])
            m = cm.ScalarMappable(cmap='jet')
            npz_img = np.load(dir_load_npz_saliency_map + key + 
                          truth_files[temp_id-1].strip('frame').strip('.jpg') + '.npz')[key]
            rgb_img = m.to_rgba(npz_img)[:,:,:3]
            plt.imsave(dir_save_png_map + truth_files[temp_id-1], rgb_img)   # Save jpg
            temp_id += 1
        
    # Load saliency maps and play  
    print('\n loading ground truth and predicted saliency maps..')      
    while ds.cur_frame_id-1 < ds.total_frame:
        print('frame:', truth_files[ds.cur_frame_id-1])            
        clock.tick(ds.target_fps)  # control FPS 
        # Display FPS
        diff_time = time.time()-last_time
        if diff_time > 1.0:
            print ('FPS: %.1f Duration: %ds(%.1f%%)' % (clock.get_fps(), 
                ds.total_frame/ds.target_fps, 100.0*ds.cur_frame_id/ds.total_frame))
            last_time=time.time()
        # Set controller
        event_handler_func()
        
        # Load ground truth frames
        s = pygame.image.load(dir_ground_truth + truth_files[ds.cur_frame_id-1])
        s = pygame.transform.scale(s, (w,h))
        pygame.Surface.convert_alpha(s)
        s.set_alpha(100)
        # Load predicted heatmaps
        try:
            heatmap = pygame.image.load(dir_save_png_map + 
                                        truth_files[ds.cur_frame_id-1])
            heatmap = pygame.transform.smoothscale(heatmap, (w,h))
        except pygame.error:
            heatmap = s
            print ("Warning: no predicted heatmap for frame ID %d" % ds.cur_frame_id)
        screen.blit(heatmap, (0,0))   # Add heatmaps
        screen.blit(s, (0,0))   # Add ground truth frames
        pygame.display.flip()
        # Check if pause
        if not ds.pause:
            ds.cur_frame_id += 1

    print ("Replay ended.")
    pygame.quit()
    sys.exit()
