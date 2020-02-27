# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 22:55:06 2018

@author: Boyu
"""

from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
from number_cuts import *
from image_process import *
from capture_nums import *
import cv2

tf.reset_default_graph()

image_cap_and_process()

a = eval(input('请输入数字的位数：'))

#Picture captured by camera in function cap_numbers()
path1 = 'cap_nums.png'

#Picture after function :image_cap_and_process()
path2 = 'binary.png'

img1 = Image.open(path1)
img2 = Image.open(path2)

"""print('原始图片如下：')
plt.imshow(img0)
plt.imshow(img1)
plt.show()"""


#print('二值化图片如下：')
plt.imshow(img2)
plt.show()

cut_to_nums(img2,a,1)

#def get_images(): 
#   images = []
#   for i in range(1,9):
#       locals()['im%i'% i] =[]
#       locals()['im%i'% i].append(Image.open('../single_nums/single_nums' + str(i-1)+'.png'))       
   
def imageprepare():  
    
    global a 
    im1 = Image.open('../single_nums/single_nums0.png')
    im2 = Image.open('../single_nums/single_nums1.png')
    im3 = Image.open('../single_nums/single_nums2.png')
    im4 = Image.open('../single_nums/single_nums3.png')
    im5 = Image.open('../single_nums/single_nums4.png')
    im6 = Image.open('../single_nums/single_nums5.png')
    im7 = Image.open('../single_nums/single_nums6.png')
    im8 = Image.open('../single_nums/single_nums7.png')
      
    tv1 = list(im1.convert('L').getdata())
    tv2 = list(im2.convert('L').getdata())
    tv3 = list(im3.convert('L').getdata())
    tv4 = list(im4.convert('L').getdata())
    tv5 = list(im5.convert('L').getdata())
    tv6 = list(im6.convert('L').getdata())
    tv7 = list(im7.convert('L').getdata())
    tv8 = list(im8.convert('L').getdata())

     
    tva1 = [(255-x)*1.0/255.0 for x in tv1] 
    tva2 = [(255-x)*1.0/255.0 for x in tv2] 
    tva3 = [(255-x)*1.0/255.0 for x in tv3] 
    tva4 = [(255-x)*1.0/255.0 for x in tv4] 
    tva5 = [(255-x)*1.0/255.0 for x in tv5]
    tva6 = [(255-x)*1.0/255.0 for x in tv6]
    tva7 = [(255-x)*1.0/255.0 for x in tv7]
    tva8 = [(255-x)*1.0/255.0 for x in tv8]

    return tva1,tva2,tva3,tva4,tva5,tva6,tva7,tva8
    
    """images = [im1,im2,im3,im4,im5,im6,im7,im8]
    
    for i in range(a):
        plt.subplot(1,a,i+1),plt.imshow(images[i])
        plt.xticks([]),plt.yticks([])
        plt.show()"""

result=imageprepare()
x = tf.placeholder(tf.float32, [None, 784])

y_ = tf.placeholder(tf.float32, [None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])

#generate 28*28*32
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
#generate 14*14*32
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#generate 14*14*64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#generate 7*7*64
h_pool2 = max_pool_2x2(h_conv2)

#input_nodes*output_nodes parameters
W_fc1 = weight_variable([7 * 7 * 64, 1024])
#1024 bias
b_fc1 = bias_variable([1024])

#flatten
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#work out full_connection2's all parameters
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "../MNIST_MODEL/model.ckpt") #使用模型，参数和之前的代码保持一致

    prediction=tf.argmax(y_conv,1)
    predint1=prediction.eval(feed_dict={x: [result[0]],keep_prob: 1.0}, session=sess)
    predint2=prediction.eval(feed_dict={x: [result[1]],keep_prob: 1.0}, session=sess)
    predint3=prediction.eval(feed_dict={x: [result[2]],keep_prob: 1.0}, session=sess)
    predint4=prediction.eval(feed_dict={x: [result[3]],keep_prob: 1.0}, session=sess)
    predint5=prediction.eval(feed_dict={x: [result[4]],keep_prob: 1.0}, session=sess)
    predint6=prediction.eval(feed_dict={x: [result[5]],keep_prob: 1.0}, session=sess)
    predint7=prediction.eval(feed_dict={x: [result[6]],keep_prob: 1.0}, session=sess)
    predint8=prediction.eval(feed_dict={x: [result[7]],keep_prob: 1.0}, session=sess)


    print('识别结果:')
    if a == 1:
        print(predint1[0])
        
    if a == 2:
        print(predint1[0],predint2[0])
        
    if a == 3 :
        print(predint1[0],predint2[0],predint3[0])

    if a == 4:
        print(predint1[0],predint2[0],predint3[0],predint4[0])
        
    if a == 5:
        print(predint1[0],predint2[0],predint3[0],predint4[0],predint5[0])
        
    if a == 6 :
        print(predint1[0],predint2[0],predint3[0],predint4[0],predint5[0],predint6[0])
        
    if a == 7:
        print(predint1[0],predint2[0],predint3[0],predint4[0],predint5[0],predint6[0],predint7[0])
        
    if a == 8 :
        print(predint1[0],predint2[0],predint3[0],predint4[0],predint5[0],predint6[0],predint7[0],predint8[0])
    
    
    

    
    
    
    #../code_after.png
