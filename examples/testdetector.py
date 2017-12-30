"""
Created by Alex Wang
On 2017-10-27

Fast-RCNN:https://github.com/CharlesShang/TFFRCNN
CTPN：detecting text in natural image with connectionist text proposal network
    https://github.com/eragonruan/text-detection-ctpn
EAST:An Efficient and Accurate Scene Text Detector, 2017
"""

import time
import os
import json
import tensorflow as tf
import tensorlayer as tl
from PIL import Image
import numpy as np
from multiprocessing import Pool

from examples.objectdetect import detect_data_prepare
from myutil import printutil
from examples.objectdetect.class_info import voc_classes, voc_classes_num


def faster_rcnn_model(x_input, reuse, is_training, FLAGS, anchor_set_size=9, fea_map_inds=None, box_reg=None, cls=None,
                      object_cls=None, cal_loss=True):
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
            network = tl.layers.DropoutLayer(layer=conv5_3, keep=FLAGS.keep_prob, is_fix=True, is_train=is_training,
                                             name='dropout_conv5_3')
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                                       padding='SAME', name='rpn_512')

            pred_cls = tl.layers.Conv2d(network, n_filter=2 * anchor_set_size, filter_size=(1, 1), strides=(1, 1),
                                        padding='SAME', name='cls_pred')  # [batch_size, W, H , 2k]

            pred_box = tl.layers.Conv2d(network, n_filter=4 * anchor_set_size, filter_size=(1, 1), strides=(1, 1),
                                        padding='SAME', name='box_pred')  # [batch_size, W, H , 4k]

            pred_obj_cls = tl.layers.Conv2d(network, n_filter=voc_classes_num * anchor_set_size, filter_size=(1, 1),
                                            strides=(1, 1), padding='SAME', name='obj_cls_pred')

        if not cal_loss:  ##测试阶段，不需要计算损失函数
            return pred_cls, pred_box, pred_obj_cls, conv5_3

        # 损失函数
        shape_ind = tf.shape(pred_cls.outputs)  # [image_ind, fea_map_w, fea_map_h, anchors * 2]
        shape_ind = tf.expand_dims(shape_ind, 0)
        shape_ind = tf.slice(shape_ind, [0, 0], [1, 3])  # [image_ind, fea_map_w, fea_map_h]

        # 1. cls reshape and calculate loss
        shape_cls = tf.concat([shape_ind, tf.convert_to_tensor([[anchor_set_size, 2]])], 1)
        shape_cls = tf.squeeze(shape_cls)
        pred_cls_reshape = tf.reshape(pred_cls.outputs, shape_cls)
        pred_cls_use = tf.gather_nd(pred_cls_reshape, fea_map_inds)

        box_label = tf.argmax(cls, axis=1)
        loss_cls = tl.cost.cross_entropy(pred_cls_use, box_label, name='class_cost_entropy')
        loss_cls = tf.reduce_sum(loss_cls) / FLAGS.batch_size  # about 0.01

        # 2. box_reg reshape and calculate loss
        shape_box = tf.concat([shape_ind, tf.convert_to_tensor([[anchor_set_size, 4]])], 1)
        shape_box = tf.squeeze(shape_box)
        pred_box_reshape = tf.reshape(pred_box.outputs, shape_box)
        pred_box_use = tf.gather_nd(pred_box_reshape, fea_map_inds)

        box_diff = tf.subtract(pred_box_use, box_reg)
        box_diff_abs = tf.abs(box_diff)
        y1 = 0.5 * box_diff_abs ** 2
        y2 = box_diff_abs - 0.5
        loss_box = tf.where(tf.less(box_diff_abs, tf.ones_like(box_diff_abs)), y1, y2)
        loss_box = tf.reduce_sum(loss_box) / FLAGS.batch_size  # about 3

        # 3. object classfication and calculate loss
        shape_obj_cls = tf.concat([shape_ind, tf.convert_to_tensor([[anchor_set_size, voc_classes_num]])], 1)
        shape_obj_cls = tf.squeeze(shape_obj_cls)
        pred_obj_cls_reshape = tf.reshape(pred_obj_cls.outputs, shape_obj_cls)
        pred_obj_cls_used = tf.gather_nd(pred_obj_cls_reshape, fea_map_inds)

        obj_cls_label = tf.argmax(object_cls, axis=1)
        loss_obj_cls = tl.cost.cross_entropy(pred_obj_cls_used, obj_cls_label, name='class_cost_entropy')
        loss_obj_cls = tf.reduce_sum(loss_obj_cls) / FLAGS.batch_size  # about

        # TODO: loss 添加 box reg loss和类别预测loss
        loss = 300 * loss_cls + loss_box + 100 * loss_obj_cls
        cost = loss

        return pred_cls, pred_box, pred_obj_cls, loss, cost, loss_cls, loss_box, loss_obj_cls, conv5_3


def process_one_image(data):
    result = {}
    index = data['index']
    image = Image.open(data['image_path'])
    result['image'] = np.array(image)

    fea_map_inds_batch = []
    box_reg_batch = []
    box_class_batch = []
    object_class_batch = []

    for anchor in data['positive_anchors']:
        fea_map_inds_batch.append([index] + anchor['fea_map_ind'])
        box_reg_batch.append(anchor['box_reg'])
        box_class_batch.append(anchor['cls'])
        object_class_batch.append(anchor['object_cls'])

    for anchor in data['negative_anchors']:
        fea_map_inds_batch.append([index] + anchor['fea_map_ind'])
        box_reg_batch.append(anchor['box_reg'])
        box_class_batch.append(anchor['cls'])
        object_class_batch.append(anchor['object_cls'])

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
    x_train_batch = []
    fea_map_inds_batch = []
    box_reg_batch = []
    box_class_batch = []
    object_class_batch = []
    fea_map_shape_batch = []

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
        fea_map_inds_batch.extend(data['fea_map_inds'])
        box_reg_batch.extend(data['box_reg'])
        box_class_batch.extend(data['box_class'])
        object_class_batch.extend(data['object_class'])

    return x_train_batch, fea_map_inds_batch, box_reg_batch, box_class_batch, object_class_batch, fea_map_shape_batch


def train_faster_rcnn(FLAGS):
    print("start train faster rcnn model")
    # 2.load data
    train_data_info = detect_data_prepare.generate_train_pathes(pickle_save_path=FLAGS.input_dir)
    train_data_info = np.asarray(train_data_info)
    total_data_num = len(train_data_info)

    valid_set_num = 100
    train_data_set = train_data_info[:(total_data_num - valid_set_num)]  # 训练集
    valid_data_set = train_data_info[(total_data_num - valid_set_num):]  # 验证集
    print('size of train_data_set:{}, size of valid_data_set:{}'.format(len(train_data_set), len(valid_data_set)))

    # train_data_info to batch
    # train_input_queue = tf.train.slice_input_producer([train_data_set])
    # train_batch = tf.train.shuffle_batch(train_input_queue, FLAGS.batch_size, capacity=32, min_after_dequeue=10, num_threads=12)
    # print('batch preprocess...')
    # x_train_batch, fea_map_inds_batch, box_reg_batch, box_class_batch, object_class_batch, fea_map_shape_batch = batch_preprocess(train_batch)

    # 3.build graph：including loss function，learning rate decay，optimization operation
    print('start build graph...')
    x_train = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, None, None, 3], name='x_train')
    fea_map_inds = tf.placeholder(tf.int32, shape=[None, 4], name='fea_map_inds')  # [Image_ind, W_ind, H_ind, k_ind]
    box_reg = tf.placeholder(tf.float32, shape=[None, 4], name='box_reg')
    box_class = tf.placeholder(tf.int16, shape=[None, 2], name='box_class')
    object_class = tf.placeholder(tf.int16, shape=[None, voc_classes_num], name='object_class')

    # def faster_rcnn_model(x_input, reuse, is_training, FLAGS, anchor_set_size=9,
    #                       fea_map_inds=None, box_reg=None, cls=None, object_cls=None, cal_loss = True):
    # pred_cls_train, pred_box_train, loss_train, cost_train = faster_rcnn_model(
    #         x_train_batch, reuse=False, is_training=True, FLAGS=FLAGS, fea_map_inds=fea_map_inds_batch,
    #         box_reg=box_reg_batch, cls=box_class_batch, object_cls=object_class_batch, cal_loss=True)  # train
    pred_cls_train, pred_box_train, pred_obj_cls_train, loss_train, cost_train, loss_cls_train, loss_box_train, loss_obj_cls_train, conv5_3 = faster_rcnn_model(
            x_input=x_train, reuse=False, is_training=True, FLAGS=FLAGS, fea_map_inds=fea_map_inds,
            box_reg=box_reg, cls=box_class, object_cls=object_class, cal_loss=True)  # train

    pred_cls_pred, pred_box_pred, pred_obj_cls_pred, loss_pred, cost_pred, loss_cls_pred, loss_box_pred, loss_obj_cls_pred, _ = faster_rcnn_model(
            x_input=x_train, reuse=True, is_training=False, FLAGS=FLAGS, fea_map_inds=fea_map_inds,
            box_reg=box_reg, cls=box_class, object_cls=object_class, cal_loss=True)  # train)  # validate/test

    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                               staircase=True)
    # learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step,
    #                                           end_learning_rate=0.00001)
    incr_global_step = tf.assign(global_step, global_step + 1)

    # TODO:修改需要训练的模型参数，冻结VGG16变量
    # train_params = net.all_params
    train_params = []
    pred_cls_params = pred_cls_train.all_params
    pred_box_params = pred_box_train.all_params
    pred_obj_cls_params = pred_obj_cls_train.all_params
    train_params.append(pred_cls_params[20:])
    train_params.append(pred_box_params[20:])
    train_params.append(pred_obj_cls_params[20:])
    # train_params = tf.trainable_variables()
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).\
                        minimize(cost_train, var_list=train_params)

    # train_vars = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(cost_train, train_vars),
    #                                   name='clip_grads')
    # train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).apply_gradients(zip(grads, train_vars))

    init_op = tf.global_variables_initializer()

    # 4.summary
    tf.summary.scalar('cost', cost_train)
    tf.summary.scalar('learning_rate', learning_rate)

    # 5.training、valid、save check point in loops
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # sv = tf.train.Supervisor(logdir=FLAGS.summary_dir, save_summaries_secs=10, saver=None)
    # with sv.managed_session(config=config) as sess:
    with tf.Session() as sess:
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
            pred_cls_train.print_params()
            pred_cls_train.print_layers()
            # pred_box_train.print_params()
            # pred_box_train.print_layers()
            print('tl.layers.print_all_variables()..................')
            tl.layers.print_all_variables()

        # load check point if FLAGS.checkpoint is not None
        if FLAGS.checkpoint is not None:
            saver.restore(sess, FLAGS.checkpoint)

        for step in range(FLAGS.max_iter):
            permutation_ind = np.random.permutation(len(train_data_set))
            x_train_batch, fea_map_inds_batch, box_reg_batch, box_class_batch, object_class_batch, fea_map_shape_batch = \
                batch_preprocess(train_data_set[permutation_ind[:FLAGS.batch_size]])
            # x_train = tf.placeholder(tf.float32, [FLAGS.batch_size, None, None, 3])
            # fea_map_inds = tf.placeholder(tf.int16, [None, 4])  #[Image_ind, W_ind, H_ind, k_ind]
            # box_reg = tf.placeholder(tf.float32, [None, 4])
            # box_class = tf.placeholder(tf.int16, [None, 2])
            # object_class = tf.placeholder(tf.int16, [None, voc_classes_num])
            feed_dict = {x_train: x_train_batch, fea_map_inds: fea_map_inds_batch, box_reg: box_reg_batch,
                         box_class: box_class_batch, object_class: object_class_batch}
            start_time = time.time()
            fetches = {'train_op': train_op, 'global_step': global_step, 'inc_global_step': incr_global_step}

            if (step + 1) % FLAGS.print_info_freq == 0:
                fetches['cost'] = cost_train
                fetches['learning_rate'] = learning_rate
                fetches['pred_cls_train'] = pred_cls_train.outputs
                fetches['loss_cls_train'] = loss_cls_train
                fetches['loss_box_train'] = loss_box_train
                fetches['loss_obj_cls_train'] = loss_obj_cls_train

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
                print("loss_cls_train:{:.4f}, loss_box_train:{:.4f}, loss_obj_cls_train:{:.4f}".
                      format(result['loss_cls_train'], result['loss_box_train'], result['loss_obj_cls_train']))
                print("learning rate:{:.6f}".format(result['learning_rate']))
                # pred_value = result['pred_cls_train']
                # print(pred_value[0:50])

            if (result['global_step'] + 1) % FLAGS.save_model_freq == 0:
                print("save model")
                if not os.path.exists(FLAGS.save_model_dir):
                    os.mkdir(FLAGS.save_model_dir)
                saver.save(sess, os.path.join(FLAGS.save_model_dir, 'model'), global_step=global_step)

            if (result['global_step'] + 1) % FLAGS.valid_freq == 0 or step == 0:
                print("validate model...")
                permutation_ind = np.random.permutation(len(valid_data_set))
                x_train_batch, fea_map_inds_batch, box_reg_batch, box_class_batch, object_class_batch, fea_map_shape_batch = \
                        batch_preprocess(train_data_set[permutation_ind[:FLAGS.batch_size]])
                feed_dict = {x_train:x_train_batch, fea_map_inds:fea_map_inds_batch, box_reg:box_reg_batch,
                             box_class:box_class_batch, object_class:object_class_batch}
                fetches['cost_pred'] = cost_pred
                fetches['loss_cls_pred'] = loss_cls_pred
                fetches['loss_box_pred'] = loss_box_pred
                fetches['loss_obj_cls_pred'] = loss_obj_cls_pred
                result = sess.run(fetches, feed_dict = feed_dict)
                print("cost_pred:{:.4f}".format(result['cost_pred']))
                print("loss_cls_pred:{:.4f}, loss_box_pred:{:.4f}, loss_obj_cls_pred:{:.4f}".
                      format(result['loss_cls_pred'], result['loss_box_pred'], result['loss_obj_cls_pred']))

        print('optimization finished!')
        summary_writer.close()


# 6.testing
def test(FLAGS):
    print("start test faster-rcnn model")
    # TODO: load pretrained model and run test
