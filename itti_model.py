#-------------------------------------------------------------------------------
# Name:        main
# Purpose:     Testing the package pySaliencyMap
#
# Author:      Akisato Kimura <akisato@ieee.org>
#
# Created:     May 4, 2014
# Copyright:   (c) Akisato Kimura 2014-
# Licence:     All rights reserved
#-------------------------------------------------------------------------------

import cv2
import matplotlib.pyplot as plt
import pySaliencyMap


# main
class itti_model:
    def __init__(self, fname):
        self.fname = fname
    def saliency_map(self):
        # read
        img = cv2.imread(self.fname)
        # initialize
        imgsize = img.shape
        img_width  = imgsize[1]
        img_height = imgsize[0]
        sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
        # computation
        self.saliency_map = sm.SMGetSM(img)
        self.binarized_map = sm.SMGetBinarizedSM(img)
        self.salient_region = sm.SMGetSalientRegion(img)
        # visualize
#    #    plt.subplot(2,2,1), plt.imshow(img, 'gray')
#        plt.subplot(2,2,1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#        plt.title('Input image')
#    #    cv2.imshow("input",  img)
#        plt.subplot(2,2,2), plt.imshow(self.saliency_map, 'gray')
#        plt.title('Saliency map')
#    #    cv2.imshow("output", map)
#        plt.subplot(2,2,3), plt.imshow(self.binarized_map)
#        plt.title('Binarilized saliency map')
#    #    cv2.imshow("Binarized", binarized_map)
#        plt.subplot(2,2,4), plt.imshow(cv2.cvtColor(self.salient_region, cv2.COLOR_BGR2RGB))
#        plt.title('Salient region')
#    #    cv2.imshow("Segmented", segmented_map)
#    
#        plt.show()
#    #    cv2.waitKey(0)
#        cv2.destroyAllWindows()
    
  
#%%
#for i in range(1920):
#    for j in range(1080):
#        if gaze_map[j,i]!=0:
#            print(j,i)
#
#plt.figure(2)
#plt.imshow(heatmap)
        
