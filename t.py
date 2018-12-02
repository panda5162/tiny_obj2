# import os
# from PIL import Image
# # import cv2
# import tensorlayer as tl
# import config
# import tensorflow as tf
# import numpy as np
#
# # img = cv2.imread('VOCROOT/VOC2007/val_img/1.jpg')
# # size = img.shape
# # # print(img)
# # # print(size)
# # # print([img])
# #
# # train_hr_img_list = sorted(tl.files.load_file_list(path='VOCROOT/VOC2007/val_img', regx='.*.jpg', printable=False))
# #
# # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path='VOCROOT/VOC2007/val_img', n_threads=32)
# # print(train_hr_imgs)
# # print(train_hr_imgs[0].shape)
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
# # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# print(sess.run(c))
#


# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Define add layer function.
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs

# Define palceholder for inputs to network.
# Use [with] including xs & ys:
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_in') # Add name
    ys = tf.placeholder(tf.float32, [None, 1], name='y_in')


# Add hidden layer
with tf.name_scope('hidden_layer'):
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# Add output layer
with tf.name_scope('output_layer'):
    prediction = add_layer(l1, 10, 1, activation_function=None)

# The error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
        reduction_indices=[1]))

with tf.name_scope('train'):
    train_setp = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
# ** Add frame to file
writer = tf.summary.FileWriter('./graph/', sess.graph)

# Important step
sess.run(tf.initialize_all_variables())