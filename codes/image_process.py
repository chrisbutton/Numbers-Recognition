# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 19:37:44 2018

@author: Boyu
"""

import cv2
import time

def image_process(in_path,out_path):

    image = cv2.imread(in_path,0)
    blurred = cv2.GaussianBlur(image,(5,5),0) 
    ret,thresh = cv2.threshold(blurred , 160 ,255 ,cv2.THRESH_BINARY)

    time.sleep(3)

    print('图片二值化处理完成')

    cv2.imwrite(out_path,thresh)