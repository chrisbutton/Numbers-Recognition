# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 22:14:55 2018

@author: Boyu
"""

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

mnist = input_data.read_data_sets("../MNIST_data", one_hot=True) #MNIST数据集所在路径

x = tf.placeholder(tf.float32, [None, 784])

y_ = tf.placeholder(tf.float32, [None, 10])

#定义产生权重的函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

#定义产生偏置量的函数
def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

#定义产生卷积核的函数
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

#定义产生最大池化层的函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#将数据集整理成卷积神经网络能处理的形式
x_image = tf.reshape(x,[-1,28,28,1])

#声明卷积层一（第一层）的变量
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#实现卷积层一的前向传播过程
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)

#实现池化层一（第二层）的前向传播过程
h_pool1 = max_pool_2x2(h_conv1)

#声明卷积层二（第三层）的变量
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#实现卷积层一的前向传播过程
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

#实现池化层二（第四层）的前向传播过程
h_pool2 = max_pool_2x2(h_conv2)


#生成全连接层一（第五层）
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

#生成全连接层二（第六层）
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

"""定义dropout，dropout可以进一步提升模型的可靠性并防止过拟合。keep_prob范围为（0，1]，
当keep_prob = 1 时，不进行dropout操作"""
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#生成全连接层二（第六层）
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#训练得到的结果经过softmax层，将神经网络的输出转化为一个概率分布
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#定义交叉熵作为神经网络的损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

#使用Adam优化算法，在0.0001基础学习率的基础上动态调整对于每个参数的学习率
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#使用前向传播的结果预测训练的正确率
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

#正确率以浮点数的形式返回
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#初始化TensorFlow持久化类
saver = tf.train.Saver() #定义saver

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('After %d training steps , training accuracy  is %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    saver.save(sess, "../MNIST_MODEL2/model.ckpt") #模型储存位置

    print('Training is over, final test accuracy is %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
    
    
    
    
    