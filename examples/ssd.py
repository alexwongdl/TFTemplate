"""
Created by Alex Wang
On 2017-10-27

Fast-RCNN:https://github.com/CharlesShang/TFFRCNN
CTPN：detecting text in natural image with connectionist text proposal network
    https://github.com/eragonruan/text-detection-ctpn
EAST:An Efficient and Accurate Scene Text Detector, 2017
SSD：Single Shot MultiBox Detector 2016
"""

import collections
import os
import time
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from PIL import Image
import random

from examples.objectdetect import ssd_data_prepare
from examples.objectdetect.ssd_anchors import ssd_anchors, ssd_anchors_step_num

# anchors 层数
ssd_anchors_layers_num = len(ssd_anchors)

from examples.objectdetect.class_info import voc_classes, voc_classes_num


def ssd_predict_layer(network, anchor_set_size, layer_num):
    pred_cls = tl.layers.Conv2d(network, n_filter=2 * anchor_set_size, filter_size=(1, 1), strides=(1, 1),
                                padding='SAME', name='cls_pred_' + str(layer_num))  # [batch_size, W, H , 2k]
    pred_box = tl.layers.Conv2d(network, n_filter=4 * anchor_set_size, filter_size=(1, 1), strides=(1, 1),
                                padding='SAME', name='box_pred_' + str(layer_num))  # [batch_size, W, H , 4k]
    pred_obj_cls = tl.layers.Conv2d(network, n_filter=voc_classes_num * anchor_set_size, filter_size=(1, 1),
                                    strides=(1, 1), padding='SAME', name='obj_cls_pred_' + str(layer_num))

    shape_ind = tf.shape(pred_cls.outputs)  # [image_ind, fea_map_w, fea_map_h, anchors * 2]
    # 1. cls reshape
    shape_cls = tf.concat([shape_ind[0:3], tf.convert_to_tensor([anchor_set_size, 2])], axis=0)
    pred_cls_reshape = tf.reshape(pred_cls.outputs, shape_cls)

    # 2. box_reg reshape
    shape_box = tf.concat([shape_ind[0:3], tf.convert_to_tensor([anchor_set_size, 4])], axis=0)
    pred_box_reshape = tf.reshape(pred_box.outputs, shape_box)

    # 3. object classfication reshape
    shape_obj_cls = tf.concat([shape_ind[0:3], tf.convert_to_tensor([anchor_set_size, voc_classes_num])], axis=0)
    pred_obj_cls_reshape = tf.reshape(pred_obj_cls.outputs, shape_obj_cls)

    # return pred_cls, pred_box, pred_obj_cls
    return pred_cls, pred_box, pred_obj_cls, pred_cls_reshape, pred_box_reshape, pred_obj_cls_reshape


def ssd_model(x_input, reuse, is_training, FLAGS, anchor_set_size=3,
              fea_map_inds_1=None,
              fea_map_inds_2=None,
              fea_map_inds_3=None,
              fea_map_inds_4=None,
              box_reg_1=None,
              box_reg_2=None,
              box_reg_3=None,
              box_reg_4=None,
              cls_1=None,
              cls_2=None,
              cls_3=None,
              cls_4=None,
              object_cls_1=None,
              object_cls_2=None,
              object_cls_3=None,
              object_cls_4=None,
              cal_loss=True):
    """
    :param x_input: [batch, None, None, 3]
    :param reuse:
    :param is_training:
    :param FLAGS:
    :param anchor_set_size:每一组anchors的大小
    :param fea_map_inds:[batch_size_ind, W_ind, H_ind, k_ind] should be int32
    :param box_reg:[n, 4]
    :param cls:[n, 2]   background:[1,0], some object:[0,1]  box_class_batch
    :param object_cls:[n, cls_num]
    :param cal_loss：如果cal_loss为true，计算并返回损失函数
    :return:
    """
    fea_map_inds = []
    fea_map_inds.append(fea_map_inds_1)
    fea_map_inds.append(fea_map_inds_2)
    fea_map_inds.append(fea_map_inds_3)
    fea_map_inds.append(fea_map_inds_4)
    object_cls = []
    object_cls.append(object_cls_1)
    object_cls.append(object_cls_2)
    object_cls.append(object_cls_3)
    object_cls.append(object_cls_4)
    cls = []
    cls.append(cls_1)
    cls.append(cls_2)
    cls.append(cls_3)
    cls.append(cls_4)
    box_reg = []
    box_reg.append(box_reg_1)
    box_reg.append(box_reg_2)
    box_reg.append(box_reg_3)
    box_reg.append(box_reg_4)

    print('construct faster-rcnn model')
    # construct graph
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
    with tf.variable_scope('faster_rcnn_model', reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        x_input = tl.layers.InputLayer(x_input, name='input_layer')
        with tf.name_scope('preprocess') as scope:
            """
            Notice that we include a preprocessing layer that takes the RGB image
            with pixels values in the range of 0-255 and subtracts the mean image
            values (calculated over the entire ImageNet training set).
            """
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            x_input.outputs = x_input.outputs - mean

        with tf.name_scope('vgg16') as scope:
            """ conv1 """
            network = tl.layers.Conv2d(x_input, n_filter=64, filter_size=(3, 3),
                                       strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
            network = tl.layers.Conv2d(network, n_filter=64, filter_size=(3, 3),
                                       strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
            network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                          padding='SAME', name='pool1')
            """ conv2 """
            network = tl.layers.Conv2d(network, n_filter=128, filter_size=(3, 3),
                                       strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
            network = tl.layers.Conv2d(network, n_filter=128, filter_size=(3, 3),
                                       strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
            network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                          padding='SAME', name='pool2')
            """ conv3 """
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3),
                                       strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3),
                                       strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3),
                                       strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
            network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                          padding='SAME', name='pool3')
            """ conv4 """
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3),
                                       strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3),
                                       strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3),
                                       strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
            network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                          padding='SAME', name='pool4')
            """ conv5 """
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3),
                                       strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3),
                                       strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3),
                                       strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
            conv5_3 = network
            # cnn_network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            #                     padding='SAME', name='pool5')
            #
            # network = tl.layers.FlattenLayer(cnn_network, name='flatten')
            # network = tl.layers.DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc1_relu')
            # network = tl.layers.DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc2_relu')
            # network = tl.layers.DenseLayer(network, n_units=1000, act=tf.identity, name='fc3_relu')
            # vgg16_net = network

        with tf.name_scope('rpn') as scope:
            pred_cls_outputs = []
            pred_box_outputs = []
            pred_obj_cls_outputs = []

            pred_cls = []
            pred_box = []
            pred_obj_cls = []

            """ ssd_conv_0 """
            network = tl.layers.MaxPool2d(conv5_3, filter_size=(2, 2), strides=(2, 2),
                                          padding='SAME', name='pool5')
            # network = tl.layers.DropoutLayer(layer=network, keep=FLAGS.keep_prob, is_fix=True, is_train=is_training,
            #                                  name='dropout_ssd_conv_0')
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(1, 1), strides=(1, 1), act=tf.nn.relu,
                                       padding='SAME', name='ssd_conv0_1')
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                                       padding='SAME', name='ssd_conv0_2')
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                                       padding='SAME', name='ssd_conv0_3')
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(1, 1), strides=(1, 1), act=tf.nn.relu,
                                       padding='SAME', name='ssd_conv0_4')
            """ ssd_conv_1 """
            network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                          padding='SAME', name='pool6')
            # network = tl.layers.DropoutLayer(layer=network, keep=FLAGS.keep_prob, is_fix=True, is_train=is_training,
            #                                  name='dropout_ssd_conv_1')
            network = tl.layers.Conv2d(network, n_filter=128, filter_size=(1, 1), strides=(1, 1), act=tf.nn.relu,
                                       padding='SAME', name='ssd_conv1_1')
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                                       padding='SAME', name='ssd_conv1_2')
            """ ssd__pred_conv_1 """
            pred_cls_1, pred_box_1, pred_obj_cls_1, pred_cls_1_outputs, pred_box_1_outputs, pred_obj_cls_1_outputs = ssd_predict_layer(network, anchor_set_size, 1)
            pred_cls_outputs.append(pred_cls_1_outputs)
            pred_box_outputs.append(pred_box_1_outputs)
            pred_obj_cls_outputs.append(pred_obj_cls_1_outputs)

            pred_cls.append(pred_cls_1)
            pred_box.append(pred_box_1)
            pred_obj_cls.append(pred_obj_cls_1)

            """ ssd_conv_2 """
            network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                          padding='SAME', name='pool7')
            # network = tl.layers.DropoutLayer(layer=network, keep=FLAGS.keep_prob, is_fix=True, is_train=is_training,
            #                                  name='dropout_ssd_conv_2')
            network = tl.layers.Conv2d(network, n_filter=128, filter_size=(1, 1), strides=(1, 1), act=tf.nn.relu,
                                       padding='SAME', name='ssd_conv2_1')
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                                       padding='SAME', name='ssd_conv2_2')
            """ ssd__pred_conv_2 """
            pred_cls_2, pred_box_2, pred_obj_cls_2, pred_cls_2_outputs, pred_box_2_outputs, pred_obj_cls_2_outputs = ssd_predict_layer(network, anchor_set_size, 2)
            pred_cls_outputs.append(pred_cls_2_outputs)
            pred_box_outputs.append(pred_box_2_outputs)
            pred_obj_cls_outputs.append(pred_obj_cls_2_outputs)

            pred_cls.append(pred_cls_2)
            pred_box.append(pred_box_2)
            pred_obj_cls.append(pred_obj_cls_2)

            """ ssd_conv_3 """
            network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                          padding='SAME', name='pool8')
            # network = tl.layers.DropoutLayer(layer=network, keep=FLAGS.keep_prob, is_fix=True, is_train=is_training,
            #                                  name='dropout_ssd_conv_3')
            network = tl.layers.Conv2d(network, n_filter=128, filter_size=(1, 1), strides=(1, 1), act=tf.nn.relu,
                                       padding='SAME', name='ssd_conv3_1')
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                                       padding='SAME', name='ssd_conv3_2')
            """ ssd__pred_conv_3 """
            pred_cls_3, pred_box_3, pred_obj_cls_3, pred_cls_3_outputs, pred_box_3_outputs, pred_obj_cls_3_outputs = ssd_predict_layer(network, anchor_set_size, 3)
            pred_cls_outputs.append(pred_cls_3_outputs)
            pred_box_outputs.append(pred_box_3_outputs)
            pred_obj_cls_outputs.append(pred_obj_cls_3_outputs)

            pred_cls.append(pred_cls_3)
            pred_box.append(pred_box_3)
            pred_obj_cls.append(pred_obj_cls_3)

            """ ssd_conv_4 """
            network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                          padding='SAME', name='pool9')
            # network = tl.layers.DropoutLayer(layer=network, keep=FLAGS.keep_prob, is_fix=True, is_train=is_training,
            #                                  name='dropout_ssd_conv_4')
            network = tl.layers.Conv2d(network, n_filter=128, filter_size=(1, 1), strides=(1, 1), act=tf.nn.relu,
                                       padding='SAME', name='ssd_conv4_1')
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                                       padding='SAME', name='ssd_conv4_2')
            """ ssd__pred_conv_4 """
            pred_cls_4, pred_box_4, pred_obj_cls_4, pred_cls_4_outputs, pred_box_4_outputs, pred_obj_cls_4_outputs = ssd_predict_layer(network, anchor_set_size, 4)
            pred_cls_outputs.append(pred_cls_4_outputs)
            pred_box_outputs.append(pred_box_4_outputs)
            pred_obj_cls_outputs.append(pred_obj_cls_4_outputs)

            pred_cls.append(pred_cls_4)
            pred_box.append(pred_box_4)
            pred_obj_cls.append(pred_obj_cls_4)

            # pred_cls = tl.layers.Conv2d(network, n_filter=2 * anchor_set_size, filter_size=(1, 1), strides=(1, 1),
            #                             padding='SAME', name='cls_pred')  # [batch_size, W, H , 2k]
            #
            # pred_box = tl.layers.Conv2d(network, n_filter=4 * anchor_set_size, filter_size=(1, 1), strides=(1, 1),
            #                             padding='SAME', name='box_pred')  # [batch_size, W, H , 4k]
            #
            # pred_obj_cls = tl.layers.Conv2d(network, n_filter=voc_classes_num * anchor_set_size, filter_size=(1, 1),
            #                                 strides=(1, 1), padding='SAME', name='obj_cls_pred')

        if not cal_loss:  ##测试阶段，不需要计算损失函数
            return pred_cls_outputs, pred_box_outputs, pred_obj_cls_outputs, conv5_3

        with tf.name_scope('loss') as scope:
            """ 损失函数 """
            loss_cls = []
            loss_box = []
            loss_obj_cls = []
            obj_cls_label = []

            assert ssd_anchors_layers_num == len(pred_cls_outputs)
            for i in range(ssd_anchors_layers_num):
                pred_cls_i = pred_cls_outputs[i]
                pred_box_i = pred_box_outputs[i]
                pred_obj_cls_i = pred_obj_cls_outputs[i]

                if fea_map_inds[i] == None:
                    continue

                # 1. cls reshape and calculate loss [image_ind, fea_map_w, fea_map_h, anchors, 2]
                pred_cls_use = tf.gather_nd(pred_cls_i, fea_map_inds[i])

                box_label = tf.argmax(cls[i], axis=1)
                loss_cls_i = tl.cost.cross_entropy(pred_cls_use, box_label, name='loss_cls_' + str(i))
                loss_cls_i = tf.reduce_sum(loss_cls_i) / FLAGS.batch_size  # about 0.01
                loss_cls.append(loss_cls_i)

                # TODO:查看第一层的前景背景预测情况
                if i==0:
                    box_class_gather_0 = pred_cls_use
                    fea_map_ind_0 = fea_map_inds[i]

                # 2. box_reg reshape and calculate loss [image_ind, fea_map_w, fea_map_h, anchors, 4]
                pred_box_use = tf.gather_nd(pred_box_i, fea_map_inds[i])

                box_diff = tf.subtract(pred_box_use, box_reg[i])
                box_diff_abs = tf.abs(box_diff)
                y1 = 0.5 * box_diff_abs ** 2  # smooth L1 loss
                y2 = box_diff_abs - 0.5
                loss_box_i = tf.where(tf.less(box_diff_abs, tf.ones_like(box_diff_abs)), y1, y2, name='loss_box_' + str(i))

                ## 负样本不计算box回归
                loss_box_i = tf.where(tf.less(box_label, tf.ones_like(box_label)), tf.zeros_like(loss_box_i), loss_box_i,
                                'loss_box_' + str(i))
                loss_box_i = tf.reduce_sum(loss_box_i) / FLAGS.batch_size  # about 3
                loss_box.append(loss_box_i)

                # 3. object classfication and calculate loss  [image_ind, fea_map_w, fea_map_h, anchors, voc_classes_num]
                pred_obj_cls_used = tf.gather_nd(pred_obj_cls_i, fea_map_inds[i])

                obj_cls_label_i = tf.argmax(object_cls[i], axis=1)
                # loss_obj_cls_i = tl.cost.cross_entropy( pred_obj_cls_used, obj_cls_label_i, name='loss_obj_cls_' + str(i))
                loss_obj_cls_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=obj_cls_label_i, logits=pred_obj_cls_used, name='loss_obj_cls_' + str(i))

                zeros = tf.cast(tf.zeros_like(obj_cls_label_i),dtype=tf.bool)
                ones = tf.cast(tf.ones_like(obj_cls_label_i),dtype=tf.bool)
                loss_obj_cls_i_mask = tf.where(tf.less(obj_cls_label_i, tf.ones_like(obj_cls_label_i)), zeros, ones,'loss_obj_cls_pos_' + str(i))  # 正样本的损失
                loss_obj_cls_i_pos = tf.reduce_mean(tf.boolean_mask(loss_obj_cls_i, loss_obj_cls_i_mask))

                loss_obj_cls_i = tf.reduce_mean(loss_obj_cls_i) # about 0.07
                # loss_obj_cls_i = tf.reduce_mean(loss_obj_cls_i) + loss_obj_cls_i_pos # about 0.07
                # loss_obj_cls_i = loss_obj_cls_i_pos # about 0.07
                loss_obj_cls.append(loss_obj_cls_i)
                obj_cls_label.append(obj_cls_label_i)

            # TODO: loss 添加 box reg loss和类别预测loss
            # loss = 300 * loss_cls + loss_box + 100 * loss_obj_cls
            sum_loss_cls = sum(loss_cls)
            sum_loss_box = sum(loss_box)
            sum_loss_obj_cls = sum(loss_obj_cls)
            # loss = sum_loss_cls + 0.1 * sum_loss_box +  sum_loss_obj_cls
            loss = sum_loss_obj_cls
            cost = loss

            return pred_cls, pred_box, pred_obj_cls, pred_cls_outputs, pred_box_outputs, pred_obj_cls_outputs, loss, cost, \
                   sum_loss_cls, sum_loss_box, sum_loss_obj_cls, conv5_3, obj_cls_label, box_class_gather_0, fea_map_ind_0, loss_obj_cls_i_pos


def my_cross_entropy(output, target, name=None):
    """It is a softmax cross-entropy operation, returns the TensorFlow expression of cross-entropy of two distributions, implement
    softmax internally. See ``tf.nn.sparse_softmax_cross_entropy_with_logits``.

    Parameters
    ----------
    output : Tensorflow variable
        A distribution with shape: [batch_size, n_feature].
    target : Tensorflow variable
        A batch of index with shape: [batch_size, ].
    name : string
        Name of this loss.

    Examples
    --------
    >>> ce = tl.cost.cross_entropy(y_logits, y_target_logits, 'my_loss')

    References
    -----------
    - About cross-entropy: `wiki <https://en.wikipedia.org/wiki/Cross_entropy>`_.\n
    - The code is borrowed from: `here <https://en.wikipedia.org/wiki/Cross_entropy>`_.
    """
    # try: # old
    #     return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, targets=target))
    # except: # TF 1.0
    assert name is not None, "Please give a unique name to tl.cost.cross_entropy for TF1.0+"
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output, name=name))


negative_label = [0] * voc_classes_num
negative_label[0] = 1

def process_one_image(data):
    result = {}
    index = data['index']
    image = Image.open(data['image_path'])
    result['image'] = np.array(image)

    fea_map_inds_batch = [[],[],[],[]]
    box_reg_batch = [[],[],[],[]]
    box_class_batch = [[],[],[],[]]
    object_class_batch = [[],[],[],[]]

    for i in range(ssd_anchors_layers_num):
        positive_anchors_one_layer = data['positive_anchors'][i]
        for anchor in positive_anchors_one_layer:
            obj_cls_ind = np.argmax(anchor['object_cls'])
            obj_cls = voc_classes[obj_cls_ind]
            # if obj_cls == 'person' and random.random() > 0.2:
            if obj_cls == 'person':
                continue
            # if obj_cls == 'car' and random.random() > 0.5:
            if obj_cls == 'car':
                continue

            # test gather_nd
            # anchor['fea_map_ind'] = [ 1000000000000, 1000000000000, anchor['fea_map_ind'][2]]  ##get 0

            fea_map_inds_batch[i].append([index] + anchor['fea_map_ind'])
            box_reg_batch[i].append(anchor['box_reg'])
            box_class_batch[i].append(anchor['cls'])
            object_class_batch[i].append(anchor['object_cls'])

    for i in range(ssd_anchors_layers_num):
        negative_anchors_one_layer = data['negative_anchors'][i]
        for anchor in negative_anchors_one_layer:
            if random.random() > 0.1:
                continue

            fea_map_inds_batch[i].append([index] + anchor['fea_map_ind'])
            box_reg_batch[i].append(anchor['box_reg'])
            box_class_batch[i].append(anchor['cls'])
            object_class_batch[i].append(anchor['object_cls'])

    for i in range(ssd_anchors_layers_num):
        negative_anchors_one_layer = data['negative_anchors'][i]
        if len(fea_map_inds_batch[i]) == 0 and len(negative_anchors_one_layer)>0:
            permutation_ind = np.random.permutation(len(negative_anchors_one_layer))
            anchor = negative_anchors_one_layer[permutation_ind[0]]
            fea_map_inds_batch[i].append([index] + anchor['fea_map_ind'])
            box_reg_batch[i].append(anchor['box_reg'])
            box_class_batch[i].append(anchor['cls'])
            object_class_batch[i].append(anchor['object_cls'])

    for i in range(ssd_anchors_layers_num):
        positive_anchors_one_layer = data['positive_anchors'][i]
        if len(fea_map_inds_batch[i]) == 0 and len(positive_anchors_one_layer) > 0:
            permutation_ind = np.random.permutation(len(positive_anchors_one_layer))
            anchor = positive_anchors_one_layer[permutation_ind[0]]
            fea_map_inds_batch[i].append([index] + anchor['fea_map_ind'])
            box_reg_batch[i].append(anchor['box_reg'])
            box_class_batch[i].append(anchor['cls'])
            object_class_batch[i].append(anchor['object_cls'])

    for i in range(ssd_anchors_layers_num):
        if len(fea_map_inds_batch[i]) == 0:
            fea_map_inds_batch[i].append([index] + [0,0,0])
            box_reg_batch[i].append([0,0,0,0])
            box_class_batch[i].append([1,0])
            object_class_batch[i].append(negative_label)

    result['fea_map_inds'] = fea_map_inds_batch
    result['box_reg'] = box_reg_batch
    result['box_class'] = box_class_batch
    result['object_class'] = object_class_batch
    return result


def batch_preprocess(train_data_batch):
    """

    :param train_data_batch:
    :return:
    """
    x_train_batch = []  # [FLAGS.batch_size, None, None, 3]
    fea_map_inds_batch = [[],[],[],[]] # [ssd_anchors_layers_num, None, 4]
    box_reg_batch = [[],[],[],[]]  # [ssd_anchors_layers_num, None, 4]
    box_class_batch = [[],[],[],[]]  # [ssd_anchors_layers_num, None, 2]
    object_class_batch = [[],[],[],[]]  # [ssd_anchors_layers_num, None, voc_classes_num]

    # print(train_data_batch[0]['image_path'])
    index = 0
    for data in train_data_batch:
        data['index'] = index
        index += 1
    pool = Pool(8)
    result = pool.map(process_one_image, train_data_batch)
    pool.close()
    pool.join()

    for data in result:
        x_train_batch.append(data['image'])
        for i in range(ssd_anchors_layers_num):
            fea_map_inds_batch[i].extend(data['fea_map_inds'][i])
            box_reg_batch[i].extend(data['box_reg'][i])
            box_class_batch[i].extend(data['box_class'][i])
            object_class_batch[i].extend(data['object_class'][i])

    return x_train_batch, fea_map_inds_batch, box_reg_batch, box_class_batch, object_class_batch


def train_faster_rcnn(FLAGS):
    print("start train faster rcnn model")
    # 2.load data
    train_data_info = ssd_data_prepare.generate_train_pathes(pickle_save_path=FLAGS.input_dir)
    ## TODO:修改训练样本数量
    # train_data_info = train_data_info[1:500]
    train_data_info = np.asarray(train_data_info)
    total_data_num = len(train_data_info)

    valid_set_num = 100
    train_data_set = train_data_info[:(total_data_num - valid_set_num)]  # 训练集
    valid_data_set = train_data_info[(total_data_num - valid_set_num):]  # 验证集
    print('size of train_data_set:{}, size of valid_data_set:{}'.format(len(train_data_set), len(valid_data_set)))

    # 3.build graph：including loss function，learning rate decay，optimization operation
    print('start build graph...')
    x_train = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 512, 512, 3], name='x_train')
    fea_map_inds_1 = tf.placeholder(tf.int32, shape=[None, 4],name='fea_map_inds_1')  # [Image_ind, W_ind, H_ind, k_ind]
    fea_map_inds_2 = tf.placeholder(tf.int32, shape=[None, 4],name='fea_map_inds_2')  # [Image_ind, W_ind, H_ind, k_ind]
    fea_map_inds_3 = tf.placeholder(tf.int32, shape=[None, 4],name='fea_map_inds_3')  # [Image_ind, W_ind, H_ind, k_ind]
    fea_map_inds_4 = tf.placeholder(tf.int32, shape=[None, 4],name='fea_map_inds_4')  # [Image_ind, W_ind, H_ind, k_ind]

    box_reg_1 = tf.placeholder(tf.float32, shape=[None, 4], name='box_reg_1')
    box_reg_2 = tf.placeholder(tf.float32, shape=[None, 4], name='box_reg_2')
    box_reg_3 = tf.placeholder(tf.float32, shape=[None, 4], name='box_reg_3')
    box_reg_4 = tf.placeholder(tf.float32, shape=[None, 4], name='box_reg_4')

    box_class_1 = tf.placeholder(tf.int16, shape=[None, 2], name='box_class_1')
    box_class_2 = tf.placeholder(tf.int16, shape=[None, 2], name='box_class_2')
    box_class_3 = tf.placeholder(tf.int16, shape=[None, 2], name='box_class_3')
    box_class_4 = tf.placeholder(tf.int16, shape=[None, 2], name='box_class_4')

    object_class_1 = tf.placeholder(tf.int16, shape=[None, voc_classes_num], name='object_class_1')
    object_class_2 = tf.placeholder(tf.int16, shape=[None, voc_classes_num], name='object_class_2')
    object_class_3 = tf.placeholder(tf.int16, shape=[None, voc_classes_num], name='object_class_3')
    object_class_4 = tf.placeholder(tf.int16, shape=[None, voc_classes_num], name='object_class_4')

    pred_cls_train, pred_box_train, pred_obj_cls_train, pred_cls_outputs_train, pred_box_outputs_train, pred_obj_cls_outputs_train,\
    loss_train, cost_train, loss_cls_train, loss_box_train, loss_obj_cls_train, conv5_3, obj_cls_label_train, box_class_gather_0, \
    fea_map_ind_0, loss_obj_cls_i_pos = ssd_model(
            x_input=x_train, reuse=False, is_training=True, FLAGS=FLAGS,
            fea_map_inds_1=fea_map_inds_1,
            fea_map_inds_2=fea_map_inds_2,
            fea_map_inds_3=fea_map_inds_3,
            fea_map_inds_4=fea_map_inds_4,
            box_reg_1=box_reg_1,
            box_reg_2=box_reg_2,
            box_reg_3=box_reg_3,
            box_reg_4=box_reg_4,
            cls_1=box_class_1,
            cls_2=box_class_2,
            cls_3=box_class_3,
            cls_4=box_class_4,
            object_cls_1=object_class_1,
            object_cls_2=object_class_2,
            object_cls_3=object_class_3,
            object_cls_4=object_class_4,
            cal_loss=True)  # train

    pred_cls_pred, pred_box_pred, pred_obj_cls_pred, pred_cls_outputs_pred, pred_box_outputs_pred, pred_obj_cls_outputs_pred,\
    loss_pred, cost_pred, loss_cls_pred, loss_box_pred, loss_obj_cls_pred, _, obj_cls_label_pred, _, _, _ = ssd_model(
            x_input=x_train, reuse=True, is_training=False, FLAGS=FLAGS,
            fea_map_inds_1=fea_map_inds_1,
            fea_map_inds_2=fea_map_inds_2,
            fea_map_inds_3=fea_map_inds_3,
            fea_map_inds_4=fea_map_inds_4,
            box_reg_1=box_reg_1,
            box_reg_2=box_reg_2,
            box_reg_3=box_reg_3,
            box_reg_4=box_reg_4,
            cls_1=box_class_1,
            cls_2=box_class_2,
            cls_3=box_class_3,
            cls_4=box_class_4,
            object_cls_1=object_class_1,
            object_cls_2=object_class_2,
            object_cls_3=object_class_3,
            object_cls_4=object_class_4,
            cal_loss=True)  # train)  # validate/test

    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                               staircase=True)
    # learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step,
    #                                           end_learning_rate=0.00001)
    incr_global_step = tf.assign(global_step, global_step + 1)

    # TODO:修改需要训练的模型参数，冻结VGG16变量
    # train_params = net.all_params
    train_params = []
    for i in range(ssd_anchors_layers_num):
        pred_cls_params = pred_cls_train[i].all_params
        pred_box_params = pred_box_train[i].all_params
        pred_obj_cls_params = pred_obj_cls_train[i].all_params
        # train_params.extend(pred_cls_params[20:])
        # train_params.extend(pred_box_params[20:])
        train_params.extend(pred_obj_cls_params[20:])
        # train_params.extend(pred_obj_cls_params[18:])
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False). \
        minimize(cost_train, var_list=train_params)

    init_op = tf.global_variables_initializer()

    # 4.summary
    tf.summary.scalar('cost', cost_train)
    tf.summary.scalar('learning_rate', learning_rate)

    # 5.training、valid、save check point in loops
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
        print('start optimization...')
        with sess.as_default():
            # tl.layers.initialize_global_variables(sess)
            init_ = sess.run(init_op)
            # 加载VGG16预训练模型
            npz = np.load(FLAGS.vgg16_path)
            vgg16_params = []
            for val in sorted(npz.items()):
                print("  Loading %s" % str(val[1].shape))
                vgg16_params.append(val[1])
            tl.files.assign_params(sess, vgg16_params[0:26], conv5_3)
            pred_cls_train[ssd_anchors_layers_num - 1].print_params()
            pred_cls_train[ssd_anchors_layers_num - 1].print_layers()
            # pred_box_train.print_params()
            # pred_box_train.print_layers()
            print('tl.layers.print_all_variables()..................')
            tl.layers.print_all_variables()

        # load check point if FLAGS.checkpoint is not None
        if FLAGS.checkpoint is not None:
            saver.restore(sess, FLAGS.checkpoint)

        for step in range(FLAGS.max_iter):
            permutation_ind = np.random.permutation(len(train_data_set))
            x_train_batch, fea_map_inds_batch, box_reg_batch, box_class_batch, object_class_batch = \
                batch_preprocess(train_data_set[permutation_ind[:FLAGS.batch_size]])
            # x_train = tf.placeholder(tf.float32, [FLAGS.batch_size, None, None, 3])
            # fea_map_inds = tf.placeholder(tf.int16, [None, 4])  #[Image_ind, W_ind, H_ind, k_ind]
            # box_reg = tf.placeholder(tf.float32, [None, 4])
            # box_class = tf.placeholder(tf.int16, [None, 2])
            # object_class = tf.placeholder(tf.int16, [None, voc_classes_num])
            feed_dict = {x_train: x_train_batch,
                         fea_map_inds_1: fea_map_inds_batch[0],
                         fea_map_inds_2: fea_map_inds_batch[1],
                         fea_map_inds_3: fea_map_inds_batch[2],
                         fea_map_inds_4: fea_map_inds_batch[3],
                         box_reg_1: box_reg_batch[0],
                         box_reg_2: box_reg_batch[1],
                         box_reg_3: box_reg_batch[2],
                         box_reg_4: box_reg_batch[3],
                         box_class_1: box_class_batch[0],
                         box_class_2: box_class_batch[1],
                         box_class_3: box_class_batch[2],
                         box_class_4: box_class_batch[3],
                         object_class_1: object_class_batch[0],
                         object_class_2: object_class_batch[1],
                         object_class_3: object_class_batch[2],
                         object_class_4: object_class_batch[3]
                         }
            start_time = time.time()
            fetches = {'train_op': train_op, 'global_step': global_step, 'inc_global_step': incr_global_step}

            if (step + 1) % FLAGS.print_info_freq == 0:
                fetches['cost'] = cost_train
                fetches['learning_rate'] = learning_rate
                # for i in range(ssd_anchors_layers_num):
                #     fetches['pred_cls_train_' + str(i)] = pred_cls_train[i]
                fetches['loss_cls_train'] = loss_cls_train
                fetches['loss_box_train'] = loss_box_train
                fetches['loss_obj_cls_train'] = loss_obj_cls_train
                fetches['obj_cls_label_train'] = obj_cls_label_train # label
                fetches['pred_obj_cls_outputs_train'] = pred_obj_cls_outputs_train # predict result
                fetches['box_class_gather_0'] = box_class_gather_0
                fetches['fea_map_ind_0'] = fea_map_ind_0
                fetches['loss_obj_cls_i_pos'] = loss_obj_cls_i_pos

            if (step + 1) % FLAGS.summary_freq == 0:
                # fetches['summary_op'] = sv.summary_op  # sv.summary_op = summary.merge_all()
                fetches['summary_op'] = tf.summary.merge_all()

            result = sess.run(fetches, feed_dict=feed_dict)

            if (step + 1) % FLAGS.summary_freq == 0:
                # sv.summary_computed(sess, result['summary_op'], global_step=result['global_step'])
                summary_writer.add_summary(result['summary_op'], result['global_step'])

            if (step + 1) % FLAGS.print_info_freq == 0:
                rate = FLAGS.batch_size / (time.time() - start_time)
                print("step:{}\t, rate:{:.2f} images/sec".format(step + 1, rate))
                print("global step:{}".format(result['global_step']))
                print("cost:{:.4f}".format(result['cost']))
                print("loss_cls_train:{:.4f}, loss_box_train:{:.4f}, loss_obj_cls_train:{:.4f}, loss_obj_cls_i_pos:{:.4f}".
                      format(result['loss_cls_train'], result['loss_box_train'], result['loss_obj_cls_train'], result['loss_obj_cls_i_pos']))
                print("learning rate:{:.6f}".format(result['learning_rate']))
                obj_set = set()
                for label_set in result['obj_cls_label_train']:
                    for i in label_set:
                        obj_set.add(voc_classes[i])
                print("obj_cls_label:{}".format(",".join(obj_set)))
                # pred_value = result['pred_cls_train']
                # print(pred_value[0:50])
                predict_cls_set = predict_to_class_name_set(result['pred_obj_cls_outputs_train'])
                print("obj_cls_predict:{}".format(",".join(predict_cls_set)))
                # print(result['pred_obj_cls_outputs_train'])
                for layer in result['pred_obj_cls_outputs_train']:
                    print(np.array(layer).shape)

                print("object_class_batch:")
                print(object_class_batch)
                print('fea_map_inds_batch:')
                fea_map_inds_temp = fea_map_inds_batch[2][0]
                print(fea_map_inds_batch)
                print('box_class_batch')
                print(box_class_batch)
                # print(result['pred_obj_cls_outputs_train'][2][fea_map_inds_temp[0]][fea_map_inds_temp[1]][fea_map_inds_temp[2]][fea_map_inds_temp[3]])
                print('box_class_gather_0')
                print(result['box_class_gather_0'])
                print('fea_map_ind_0')
                print(result['fea_map_ind_0'])

            if (result['global_step'] + 1) % FLAGS.save_model_freq == 0:
                print("save model")
                if not os.path.exists(FLAGS.save_model_dir):
                    os.mkdir(FLAGS.save_model_dir)
                saver.save(sess, os.path.join(FLAGS.save_model_dir, 'model'), global_step=global_step)

            if (result['global_step'] + 1) % FLAGS.valid_freq == 0 or step == 0:
                print("validate model...")
                permutation_ind = np.random.permutation(len(valid_data_set))
                x_train_batch, fea_map_inds_batch, box_reg_batch, box_class_batch, object_class_batch = \
                    batch_preprocess(valid_data_set[permutation_ind[:FLAGS.batch_size]])
                # feed_dict = {x_train: x_train_batch, fea_map_inds: fea_map_inds_batch, box_reg: box_reg_batch,
                #              box_class: box_class_batch, object_class: object_class_batch}
                feed_dict = {x_train: x_train_batch,
                             fea_map_inds_1: fea_map_inds_batch[0],
                             fea_map_inds_2: fea_map_inds_batch[1],
                             fea_map_inds_3: fea_map_inds_batch[2],
                             fea_map_inds_4: fea_map_inds_batch[3],
                             box_reg_1: box_reg_batch[0],
                             box_reg_2: box_reg_batch[1],
                             box_reg_3: box_reg_batch[2],
                             box_reg_4: box_reg_batch[3],
                             box_class_1: box_class_batch[0],
                             box_class_2: box_class_batch[1],
                             box_class_3: box_class_batch[2],
                             box_class_4: box_class_batch[3],
                             object_class_1: object_class_batch[0],
                             object_class_2: object_class_batch[1],
                             object_class_3: object_class_batch[2],
                             object_class_4: object_class_batch[3]
                             }
                fetches['cost_pred'] = cost_pred
                fetches['loss_cls_pred'] = loss_cls_pred
                fetches['loss_box_pred'] = loss_box_pred
                fetches['loss_obj_cls_pred'] = loss_obj_cls_pred
                result = sess.run(fetches, feed_dict=feed_dict)
                print("cost_pred:{:.4f}".format(result['cost_pred']))
                print("loss_cls_pred:{:.4f}, loss_box_pred:{:.4f}, loss_obj_cls_pred:{:.4f}".
                      format(result['loss_cls_pred'], result['loss_box_pred'], result['loss_obj_cls_pred']))
                print()

        print('optimization finished!')
        summary_writer.close()


# 6.testing
def test_one_image(model_dir, image_path):
    print("start test faster-rcnn model")
    # load pretrained model and run test
    F = collections.namedtuple('F', {'init_scale', 'keep_prob', 'batch_size'})
    FLAGS = F(init_scale=0.01, keep_prob=1.0, batch_size=1)

    x_test = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='x_test')
    pred_cls, pred_box, pred_obj_cls, conv5_3 = ssd_model(
            x_input=x_test, reuse=False, is_training=False, FLAGS=FLAGS, fea_map_inds=None,
            box_reg=None, cls=None, object_cls=None, cal_loss=False)  # train
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_dir)

        image = np.array(Image.open(image_path))
        anchor_set_size = 3
        for i in range(ssd_anchors_layers_num):
            pred_cls_i = pred_cls[i].outputs
            pred_box_i = pred_box[i].outputs
            pred_obj_cls_i = pred_obj_cls[i].outputs
            pred_cls_shape = tf.shape(pred_cls_i)


            pred_cls_size = tf.concat([pred_cls_shape[0:3], tf.convert_to_tensor([anchor_set_size, 2])], axis=0)
            pred_cls_reshape = tf.reshape(pred_cls_i, pred_cls_size)

            pred_box_size = tf.concat([pred_cls_shape[0:3], tf.convert_to_tensor([anchor_set_size, 4])], axis=0)
            pred_box_reshape = tf.reshape(pred_box_i, pred_box_size)

            pred_obj_cls_size = tf.concat([pred_cls_shape[0:3], tf.convert_to_tensor([anchor_set_size, voc_classes_num])],
                                          axis=0)
            pred_obj_cls_reshape = tf.reshape(pred_obj_cls_i, [-1, voc_classes_num])

            pred_cls_reshape_arg_max = tf.argmax(pred_obj_cls_reshape, axis=1)
            pred_cls_foreground_ind = tf.greater(pred_cls_reshape_arg_max,
                                                 tf.zeros_like(pred_cls_reshape_arg_max, dtype=tf.int64))
            pred_cls_foreground_ind = tf.where(pred_cls_foreground_ind)
            pred_cls_foreground = tf.gather_nd(pred_cls_reshape_arg_max, pred_cls_foreground_ind)


            fetches = {}
            fetches['pred_cls'] = pred_cls_reshape
            fetches['pred_box'] = pred_box_reshape
            fetches['pred_obj_cls'] = pred_obj_cls_reshape
            fetches['pred_cls_reshape_arg_max'] = pred_cls_reshape_arg_max
            feed_dict = {x_test: [image]}

            result = sess.run(fetches, feed_dict=feed_dict)
            print(result['pred_box'].shape)
            print(result['pred_cls_reshape_arg_max'])
            # print(result['pred_cls'])
            obj_set = set()
            for i in result['pred_cls_reshape_arg_max']:
                if i > 0:
                    obj_set.add(voc_classes[i])
            print(','.join(obj_set))
            print(result['pred_cls'])

def test_batch_process():
    train_data_info = ssd_data_prepare.generate_train_pathes(pickle_save_path='E://data/voc_train_data.pkl')
    train_data_info = np.asarray(train_data_info)
    total_data_num = len(train_data_info)

    valid_set_num = 100
    train_data_set = train_data_info[:(total_data_num - valid_set_num)]  # 训练集
    permutation_ind = np.random.permutation(len(train_data_set))
    x_train_batch, fea_map_inds_batch, box_reg_batch, box_class_batch, object_class_batch = \
        batch_preprocess(train_data_set[permutation_ind[0]])
    print(box_class_batch)

def predict_to_class_name_set(predict):
    """
    预测结果转化为类名set集合
    :param predict:
    :return:
    """
    # predict = np.array(predict)
    # predict_shape = predict.shape
    # print(predict_shape)
    # predict = np.reshape(predict, newshape=(predict_shape[0] * predict_shape[1] * predict_shape[2] * predict_shape[3], predict_shape[4]))
    # class_name_set = set()
    # for predict_temp in predict:
    #     index = np.argmax(predict_temp)
    #     class_name_set.add(voc_classes[index])
    # return class_name_set
    class_name_set = set()
    for image in predict:
        for wimage in image:
            for himage in wimage:
                for anchors in himage:
                    for anchor in anchors:
                        if( len(anchor) != voc_classes_num):
                            print('len anchor != voc_classes_num:' + anchor)
                        index = np.argmax(anchor)
                        class_name_set.add(voc_classes[index])
    return class_name_set

if __name__ == '__main__':
    # test_one_image('E://data/faster_rcnn_model/model-19999', 'E://data/VOCdevkit/VOC2007/JPEGImages/000044.jpg')
    test_one_image('E://data/faster_rcnn_model/model-19999','E://data/VOC_data/000036_0.jpg')
