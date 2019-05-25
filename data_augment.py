# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:21:29 2019

@author: hxq
"""

import cv2
import math
import os
import ipdb

def rotate(img, angle):
    height = img.shape[0]
    width = img.shape[1]
    
    if angle % 180 == 0:
        scale = 1
    elif angle % 90 == 0:
        scale = max(height, width) / min(height, width)
    else:
        scale = math.sqrt(math.pow(height,2) + math.pow(width,2)) / min(height, width)
        
    RotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
    RotateImg = cv2.warpAffine(img, RotateMat, (width, height))
    return RotateImg

data_path = './data/faces'
imgs = os.listdir(data_path)

for i, img in enumerate(imgs, 1):
#    ipdb.set_trace()
    img_path = os.path.join(data_path, img)
    image = cv2.imread(img_path)
    img_rotate45 = rotate(image, 45)
    img_rotate315 = rotate(image, -45)
    img_flip = cv2.flip(image, -1)
    
    fix_name = './data/face/{}'.format(i)
    cv2.imwrite(fix_name+'_origin.jpg', image)
    cv2.imwrite(fix_name+'_rotate45.jpg', img_rotate45)
    cv2.imwrite(fix_name+'_rotate315.jpg', img_rotate315)
    cv2.imwrite(fix_name+'_flip.jpg', img_flip)
    

    

    
    
    
    
    
