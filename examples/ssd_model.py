"""
Created by Alex Wang
On 2018-06-22
"""
import tensorflow as tf
import tensorlayer as tl

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
              fea_map_inds_list=None,
              box_reg_list=None,
              cls_list=None,
              object_cls_list=None,
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
    :return:end_points
    """
    end_points = dict()

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

        with tf.name_scope('ssd_layer') as scope:
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
            pred_cls_1, pred_box_1, pred_obj_cls_1, pred_cls_1_outputs, pred_box_1_outputs, pred_obj_cls_1_outputs = ssd_predict_layer(
                network, anchor_set_size, 1)
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
            pred_cls_2, pred_box_2, pred_obj_cls_2, pred_cls_2_outputs, pred_box_2_outputs, pred_obj_cls_2_outputs = ssd_predict_layer(
                network, anchor_set_size, 2)
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
            pred_cls_3, pred_box_3, pred_obj_cls_3, pred_cls_3_outputs, pred_box_3_outputs, pred_obj_cls_3_outputs = ssd_predict_layer(
                network, anchor_set_size, 3)
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
            pred_cls_4, pred_box_4, pred_obj_cls_4, pred_cls_4_outputs, pred_box_4_outputs, pred_obj_cls_4_outputs = ssd_predict_layer(
                network, anchor_set_size, 4)
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

                if fea_map_inds_list[i] == None:
                    continue

                # 1. cls reshape and calculate loss [image_ind, fea_map_w, fea_map_h, anchors, 2]
                pred_cls_use = tf.gather_nd(pred_cls_i, fea_map_inds_list[i])

                box_label = tf.argmax(cls_list[i], axis=1)
                loss_cls_i = tl.cost.cross_entropy(pred_cls_use, box_label, name='loss_cls_' + str(i))
                loss_cls_i = tf.reduce_sum(loss_cls_i) / FLAGS.batch_size  # about 0.01
                loss_cls.append(loss_cls_i)

                # TODO:查看第一层的前景背景预测情况
                if i == 0:
                    box_class_gather_0 = pred_cls_use
                    fea_map_ind_0 = fea_map_inds_list[i]

                # 2. box_reg reshape and calculate loss [image_ind, fea_map_w, fea_map_h, anchors, 4]
                pred_box_use = tf.gather_nd(pred_box_i, fea_map_inds_list[i])

                box_diff = tf.subtract(pred_box_use, box_reg_list[i])
                box_diff_abs = tf.abs(box_diff)
                y1 = 0.5 * box_diff_abs ** 2  # smooth L1 loss
                y2 = box_diff_abs - 0.5
                loss_box_i = tf.where(tf.less(box_diff_abs, tf.ones_like(box_diff_abs)), y1, y2,
                                      name='loss_box_' + str(i))

                ## 负样本不计算box回归
                loss_box_i = tf.where(tf.less(box_label, tf.ones_like(box_label)), tf.zeros_like(loss_box_i),
                                      loss_box_i,
                                      'loss_box_' + str(i))
                loss_box_i = tf.reduce_sum(loss_box_i) / FLAGS.batch_size  # about 3
                loss_box.append(loss_box_i)

                # 3. object classfication and calculate loss  [image_ind, fea_map_w, fea_map_h, anchors, voc_classes_num]
                pred_obj_cls_used = tf.gather_nd(pred_obj_cls_i, fea_map_inds_list[i])

                obj_cls_label_i = tf.argmax(object_cls_list[i], axis=1)
                # loss_obj_cls_i = tl.cost.cross_entropy( pred_obj_cls_used, obj_cls_label_i, name='loss_obj_cls_' + str(i))
                loss_obj_cls_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=obj_cls_label_i,
                                                                                logits=pred_obj_cls_used,
                                                                                name='loss_obj_cls_' + str(i))

                zeros = tf.cast(tf.zeros_like(obj_cls_label_i), dtype=tf.bool)
                ones = tf.cast(tf.ones_like(obj_cls_label_i), dtype=tf.bool)
                loss_obj_cls_i_mask = tf.where(tf.less(obj_cls_label_i, tf.ones_like(obj_cls_label_i)), zeros, ones,
                                               'loss_obj_cls_pos_' + str(i))  # 正样本的损失
                loss_obj_cls_i_pos = tf.reduce_mean(tf.boolean_mask(loss_obj_cls_i, loss_obj_cls_i_mask))

                loss_obj_cls_i = tf.reduce_mean(loss_obj_cls_i)  # about 0.07
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

            end_points['pred_cls'] = pred_cls
            end_points['pred_box'] = pred_box
            end_points['pred_obj_cls'] = pred_obj_cls
            end_points['pred_cls_outputs'] = pred_cls_outputs
            end_points['pred_box_outputs'] = pred_box_outputs
            end_points['pred_obj_cls_outputs'] = pred_obj_cls_outputs
            end_points['loss'] = loss
            end_points['cost'] = cost
            end_points['sum_loss_cls'] = sum_loss_cls
            end_points['sum_loss_box'] = sum_loss_box
            end_points['sum_loss_obj_cls'] = sum_loss_obj_cls
            end_points['conv5_3'] = conv5_3
            end_points['obj_cls_label'] = obj_cls_label
            end_points['box_class_gather_0'] = box_class_gather_0
            end_points['fea_map_ind_0'] = fea_map_ind_0
            end_points['loss_obj_cls_i_pos'] = loss_obj_cls_i_pos

            return end_points
