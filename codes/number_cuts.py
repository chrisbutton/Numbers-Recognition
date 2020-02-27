# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:17:08 2018

@author: Boyu
"""

from PIL import Image

def cut_to_nums(image,xx,yy):    

    img_size = image.size
    x = img_size[0] // xx
    y = img_size[1] // yy
    for i in range(xx):
	    left = i*x
	    right = left + x
	    up = 0
	    low = y
			
	    region = image.crop((left,up,right,low))
	    region = region.resize((28,28))
	    temp = str(i)
	    region.save("../single_nums/single_nums"+temp+".png")  


    
