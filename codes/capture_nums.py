# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:29:13 2018

@author: Boyu
"""

import cv2
import numpy as np


def image_cap_and_process():
	cap = cv2.VideoCapture(0)   

	while(True):
		ret , frame = cap.read()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
		blur = cv2.blur(gray, (5,5))
		equ = cv2.equalizeHist(blur)
		cv2.imshow("capture", frame)
		cv2.imshow('equ',equ)
		if cv2.waitKey(1) & 0xFF == ord('s'):
			cv2.imwrite('cap_nums_gray.png',gray)
			cv2.imwrite('cap_nums_blur.png',blur)
			cv2.imwrite('cap_nums_equ.png',equ)
			cv2.imwrite('cap_nums.png',frame)
			break


	ret,thresh = cv2.threshold(equ,15,255,cv2.THRESH_BINARY)
	cv2.imwrite('binary.png',thresh)
	#cv2.imshow('thresh',thresh)
	cv2.waitKey(0)
    
	cap.release()
	cv2.destroyAllWindows()
    

    
    
        
        
        
    
