"""
Created by Alex Wang on 2017-12-13

tensorlayer:https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_vgg16.py
pretrained model and descriptor: http://www.cs.toronto.edu/~frossard/post/vgg16/
  layer   0: conv1_1/Relu:0       (?, 224, 224, 64)    float32
  layer   1: conv1_2/Relu:0       (?, 224, 224, 64)    float32
  layer   2: pool1:0              (?, 112, 112, 64)    float32
  layer   3: conv2_1/Relu:0       (?, 112, 112, 128)    float32
  layer   4: conv2_2/Relu:0       (?, 112, 112, 128)    float32
  layer   5: pool2:0              (?, 56, 56, 128)    float32
  layer   6: conv3_1/Relu:0       (?, 56, 56, 256)    float32
  layer   7: conv3_2/Relu:0       (?, 56, 56, 256)    float32
  layer   8: conv3_3/Relu:0       (?, 56, 56, 256)    float32
  layer   9: pool3:0              (?, 28, 28, 256)    float32
  layer  10: conv4_1/Relu:0       (?, 28, 28, 512)    float32
  layer  11: conv4_2/Relu:0       (?, 28, 28, 512)    float32
  layer  12: conv4_3/Relu:0       (?, 28, 28, 512)    float32
  layer  13: pool4:0              (?, 14, 14, 512)    float32
  layer  14: conv5_1/Relu:0       (?, 14, 14, 512)    float32
  layer  15: conv5_2/Relu:0       (?, 14, 14, 512)    float32
  layer  16: conv5_3/Relu:0       (?, 14, 14, 512)    float32
  layer  17: pool5:0              (?, 7, 7, 512)     float32
  layer  18: flatten:0            (?, 25088)         float32
  layer  19: fc1_relu/Relu:0      (?, 4096)          float32
  layer  20: fc2_relu/Relu:0      (?, 4096)          float32
  layer  21: fc3_relu/Identity:0  (?, 1000)          float32


  param   0: conv1_1/W_conv2d:0   (3, 3, 3, 64)      float32_ref (mean: -0.0006835053791292012, median: -0.0006735174683853984, std: 0.01788090169429779)
  param   1: conv1_1/b_conv2d:0   (64,)              float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param   2: conv1_2/W_conv2d:0   (3, 3, 64, 64)     float32_ref (mean: -6.518061127280816e-05, median: -9.442680311622098e-05, std: 0.017496472224593163)
  param   3: conv1_2/b_conv2d:0   (64,)              float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param   4: conv2_1/W_conv2d:0   (3, 3, 64, 128)    float32_ref (mean: 0.0001446532114641741, median: 0.00011040597746614367, std: 0.017627066001296043)
  param   5: conv2_1/b_conv2d:0   (128,)             float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param   6: conv2_2/W_conv2d:0   (3, 3, 128, 128)    float32_ref (mean: -3.7152728964429116e-06, median: 6.197370112204226e-06, std: 0.017558325082063675)
  param   7: conv2_2/b_conv2d:0   (128,)             float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param   8: conv3_1/W_conv2d:0   (3, 3, 128, 256)    float32_ref (mean: -7.893554538895842e-06, median: -6.899346772115678e-05, std: 0.01756639964878559)
  param   9: conv3_1/b_conv2d:0   (256,)             float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param  10: conv3_2/W_conv2d:0   (3, 3, 256, 256)    float32_ref (mean: 2.424410922685638e-05, median: 2.583660534583032e-05, std: 0.017574215307831764)
  param  11: conv3_2/b_conv2d:0   (256,)             float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param  12: conv3_3/W_conv2d:0   (3, 3, 256, 256)    float32_ref (mean: -1.372377391817281e-05, median: -1.982607864192687e-05, std: 0.017588535323739052)
  param  13: conv3_3/b_conv2d:0   (256,)             float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param  14: conv4_1/W_conv2d:0   (3, 3, 256, 512)    float32_ref (mean: -3.1803258480067598e-06, median: -3.6449901017476805e-06, std: 0.017595777288079262)
  param  15: conv4_1/b_conv2d:0   (512,)             float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param  16: conv4_2/W_conv2d:0   (3, 3, 512, 512)    float32_ref (mean: 1.1407656529627275e-05, median: 3.367540102772182e-06, std: 0.017596272751688957)
  param  17: conv4_2/b_conv2d:0   (512,)             float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param  18: conv4_3/W_conv2d:0   (3, 3, 512, 512)    float32_ref (mean: -9.763101616044878e-07, median: 9.682942618383095e-06, std: 0.01759697124361992)
  param  19: conv4_3/b_conv2d:0   (512,)             float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param  20: conv5_1/W_conv2d:0   (3, 3, 512, 512)    float32_ref (mean: 1.7139340343419462e-05, median: 1.1453739716671407e-05, std: 0.01759149506688118)
  param  21: conv5_1/b_conv2d:0   (512,)             float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param  22: conv5_2/W_conv2d:0   (3, 3, 512, 512)    float32_ref (mean: 1.104799139284296e-05, median: 6.277702595980372e-06, std: 0.017592385411262512)
  param  23: conv5_2/b_conv2d:0   (512,)             float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param  24: conv5_3/W_conv2d:0   (3, 3, 512, 512)    float32_ref (mean: -4.533968422038015e-06, median: -8.610464647063054e-06, std: 0.017594775184988976)
  param  25: conv5_3/b_conv2d:0   (512,)             float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param  26: fc1_relu/W:0         (25088, 4096)      float32_ref (mean: 1.2362265806586947e-05, median: 1.3758284694631584e-05, std: 0.08796705305576324)
  param  27: fc1_relu/b:0         (4096,)            float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param  28: fc2_relu/W:0         (4096, 4096)       float32_ref (mean: -1.6828264051582664e-05, median: -8.5651399786002e-06, std: 0.08796363323926926)
  param  29: fc2_relu/b:0         (4096,)            float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
  param  30: fc3_relu/W:0         (4096, 1000)       float32_ref (mean: -1.8854416339308955e-05, median: -5.936184243182652e-05, std: 0.08796814829111099)
  param  31: fc3_relu/b:0         (1000,)            float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )
"""
import os
import time
import cv2

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

# if not os.path.isfile("E://data/vgg16_weights.npz"):
#     print("Please download vgg16_weights.npz from : http://www.cs.toronto.edu/~frossard/post/vgg16/")
#     exit()
npz = np.load('/Users/alexwang/Downloads/vgg16_weights.npz')

params = []
for val in sorted( npz.items() ):
    print("  Loading %s" % str(val[1].shape))
    params.append(val[1])

tl.files.assign_params(sess, params, network)

# img1 = imread('laska.png', mode='RGB') # test data in github
# img1 = imresize(img1, (224, 224))

img1 = cv2.imread('laska.png') # test data in github
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32)
img1 = cv2.resize(img1, (224, 224))

start_time = time.time()
prob = sess.run(probs, feed_dict={x: [img1]})[0]
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])