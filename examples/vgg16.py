"""
Created by Alex Wang on 2017-12-13

tensorlayer:https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_vgg16.py
pretrained model and descriptor: http://www.cs.toronto.edu/~frossard/post/vgg16/
"""
import os
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

from scipy.misc import imread, imresize
from imagenet_classes import class_names

def conv_layers_simple_api(net_in):
    with tf.name_scope('preprocess') as scope:
        """
        Notice that we include a preprocessing layer that takes the RGB image
        with pixels values in the range of 0-255 and subtracts the mean image
        values (calculated over the entire ImageNet training set).
        """
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in.outputs = net_in.outputs - mean
    """ conv1 """
    network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
                     strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
    network = Conv2d(network, n_filter=64, filter_size=(3, 3),
                     strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                        padding='SAME', name='pool1')
    """ conv2 """
    network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                     strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
    network = Conv2d(network,n_filter=128, filter_size=(3, 3),
                     strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                        padding='SAME', name='pool2')
    """ conv3 """
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                     strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                     strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                     strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                        padding='SAME', name='pool3')
    """ conv4 """
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                     strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                     strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                     strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                        padding='SAME', name='pool4')
    """ conv5 """
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                     strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                     strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                     strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                        padding='SAME', name='pool5')
    return network

def fc_layers(net):
    network = FlattenLayer(net, name='flatten')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc1_relu')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc2_relu')
    network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc3_relu')
    return network

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
# y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

net_in = InputLayer(x, name='input')
# net_cnn = conv_layers(net_in)               # professional CNN APIs
net_cnn = conv_layers_simple_api(net_in)  # simplified CNN APIs
network = fc_layers(net_cnn)

y = network.outputs
probs = tf.nn.softmax(y)
# y_op = tf.argmax(tf.nn.softmax(y), 1)
# cost = tl.cost.cross_entropy(y, y_, name='cost')
# correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.float32), tf.cast(y_, tf.float32))
# acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tl.layers.initialize_global_variables(sess)
network.print_params()
network.print_layers()

if not os.path.isfile("E://data/vgg16_weights.npz"):
    print("Please download vgg16_weights.npz from : http://www.cs.toronto.edu/~frossard/post/vgg16/")
    exit()
npz = np.load('E://data/vgg16_weights.npz')

params = []
for val in sorted( npz.items() ):
    print("  Loading %s" % str(val[1].shape))
    params.append(val[1])

tl.files.assign_params(sess, params, network)

img1 = imread('E://data/laska.png', mode='RGB') # test data in github
img1 = imresize(img1, (224, 224))

start_time = time.time()
prob = sess.run(probs, feed_dict={x: [img1]})[0]
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])