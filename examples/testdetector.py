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

from examples.objectdetect import detect_data_prepare
from myutil import printutil
from examples.objectdetect.class_info import voc_classes, voc_classes_num


def faster_rcnn_model(x_input, reuse, is_training, FLAGS, anchor_set_size=9, fea_map_inds=None, box_reg=None, cls=None,
                      object_cls=None, cal_loss = True):
    """
    :param x_input: [batch, None, None, 3]
    :param reuse:
    :param is_training:
    :param FLAGS:
    :param anchor_set_size:每一组anchors的大小
    :param fea_map_inds:[batch_size_ind, W_ind, H_ind, k_ind]
    :param box_reg:[n, 4]
    :param cls:[n, 2]   background:[1,0], some object:[0,1]
    :param object_cls:[n, cls_num]
    :param cal_loss：如果cal_loss为true，计算并返回损失函数
    :return:
    """
    print('construct faster-rcnn model')
    # construct graph
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
    with tf.variable_scope('faster_rcnn_model', reuse=reuse):
        tl.layers.set_name_reuse(reuse)
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
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu(),
                                       padding='SAME', name='rpn_512')

            pred_cls = tl.layers.Conv2d(network, n_filter=2 * anchor_set_size, filter_size=(1, 1), strides=(1, 1),
                                        padding='SAME', name='cls_pred')  # [batch_size, W, H , 2k]

            pred_box = tl.layers.Conv2d(network, n_filter=4 * anchor_set_size, filter_size=(1, 1), strides=(1, 1),
                                        padding='SAME', name='box_pred')  # [batch_size, W, H , 4k]

        if not cal_loss: ##测试阶段，不需要计算损失函数
            return pred_cls, pred_box

        #TODO:
        # loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits=[output], targets=[tf.reshape(target, [-1])], \
        #                                                           weights=[tf.reshape(target_weight, [-1])], \
        #                                                           name='sequence_loss_by_example')
        # cost = tf.reduce_sum(loss) / FLAGS.batch_size
        # cost = tf.reduce_sum(loss) / tf.reduce_sum(tf.cast(y_length, dtype=tf.float32))

        return pred_cls, pred_box, loss, cost


def train_faster_rcnn(FLAGS):
    print("start train faster rcnn model")
    # 2.load data
    train_data_info = detect_data_prepare.generate_train_pathes(pickle_save_path=FLAGS.input_dir)
    total_data_num = len(train_data_info)

    valid_set_num = 100
    train_data_set = train_data_info[:(total_data_num - valid_set_num)]  # 训练集
    valid_data_set = train_data_info[(total_data_num - valid_set_num):]  # 验证集
    print('size of train_data_set:{}, size of valid_data_set:{}'.format(len(train_data_set), len(valid_data_set)))

    # TODO: train_data_info to batch
    train_input_queue = tf.train.slice_input_producer([train_data_set])
    train_batch = tf.train.shuffle_batch(train_input_queue, FLAGS.batch_size, capacity=32, min_after_dequeue=10, num_threads=12)
    

    # 3.build graph：including loss function，learning rate decay，optimization operation
    x_train = tf.placeholder(tf.float32, [FLAGS.batch_size, None, None, 3])
    fea_map_inds = tf.placeholder(tf.int16, [None, 4])  #[Image_ind, W_ind, H_ind, k_ind]
    box_reg = tf.placeholder(tf.float32, [None, 4])
    box_class = tf.placeholder(tf.int16, [None, 2])
    object_class = tf.placeholder(tf.int16, [None, voc_classes_num])

    #def faster_rcnn_model(x_input, reuse, is_training, FLAGS, anchor_set_size=9,
    #                       fea_map_inds=None, box_reg=None, cls=None, object_cls=None, cal_loss = True):
    pred_cls_train, pred_box_train, loss_train, cost_train = faster_rcnn_model(
            x_train, reuse=False, is_training=True, FLAGS=FLAGS, fea_map_inds=fea_map_inds,
            box_reg=box_reg, cls=box_class, object_cls=object_class, cal_loss=True)  # train

    pred_cls_pred, pred_box_pred, loss_pred, cost_pred = faster_rcnn_model(
            x_train, reuse=True, is_training=False, FLAGS=FLAGS, fea_map_inds=fea_map_inds,
            box_reg=box_reg, cls=box_class, object_cls=object_class, cal_loss=True)  # train)  # validate/test

    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                               staircase=True)
    # learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step,
    #                                           end_learning_rate=0.00001)
    incr_global_step = tf.assign(global_step, global_step + 1)

    #TODO:修改需要训练的模型参数
    # train_params = net.all_params
    train_params = tf.trainable_variables()
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(
            cost_train, var_list=train_params)

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
    sv = tf.train.Supervisor(logdir=FLAGS.summary_dir, save_summaries_secs=10, saver=None)
    with sv.managed_session(config=config) as sess:
        print('start optimization...')
        with sess.as_default():
            # tl.layers.initialize_global_variables(sess)
            init_ = sess.run(init_op)
            pred_cls_train.print_params()
            pred_cls_train.print_layers()
            pred_box_train.print_params()
            pred_box_train.print_layers()
            tl.layers.print_all_variables()

        # load check point if FLAGS.checkpoint is not None
        if FLAGS.checkpoint is not None:
            saver.restore(sess, FLAGS.checkpoint)

        for round in range(FLAGS.max_max_epoch):

            for file_index in range(len(train_files)):  # 遍历训练样本
                data_list = translation_data_prepare.load_train_data(train_files[file_index])

                for step in range(len(data_list)):
                    current_data = data_list[step]
                    if max(current_data['y_len_list']) > 400:
                        print(" max(current_data['y_len_list']) > 400")
                        print(current_data['y_len_list'])
                        continue

                    start_time = time.time()
                    fetches = {'train_op': train_op, 'global_step': global_step,
                               'inc_global_step': incr_global_step, 'encode_final_state': encode_net_train.final_state}
                    # x_train, y_train, target, x_length, y_length, target_weight
                    feed_dict = {
                        x_train: current_data['x_list'], y_train: current_data['y_list'],
                        target: current_data['target_list'], x_length: current_data['x_len_list'],
                        y_length: current_data['y_len_list'], target_weight: current_data['target_weight']
                    }

                    if (step + 1) % FLAGS.print_info_freq == 0:
                        fetches['cost'] = cost_train
                        fetches['learning_rate'] = learning_rate

                    if (step + 1) % FLAGS.summary_freq == 0:
                        fetches['summary_op'] = sv.summary_op  # sv.summary_op = summary.merge_all()

                    result = sess.run(fetches, feed_dict=feed_dict)

                    if (step + 1) % FLAGS.summary_freq == 0:
                        sv.summary_computed(sess, result['summary_op'], global_step=result['global_step'])

                    if (step + 1) % FLAGS.print_info_freq == 0:
                        rate = FLAGS.batch_size / (time.time() - start_time)
                        print("epoch:{}\t, rate:{:.2f} sentences/sec".format(round, rate))
                        print("global step:{}".format(result['global_step']))
                        print("cost:{:.4f}".format(result['cost']))
                        print("learning rate:{:.6f}".format(result['learning_rate']))
                        encode_final_state = result['encode_final_state']

                        if (step + 1) % 1000 == 0:
                            print(encode_final_state[0].c)
                            print(encode_final_state[0].h)
                            print(encode_final_state[1].c)
                            print(encode_final_state[1].h)
                        print()

                    if (result['global_step'] + 1) % FLAGS.save_model_freq == 0:
                        print("save model")
                        if not os.path.exists(FLAGS.save_model_dir):
                            os.mkdir(FLAGS.save_model_dir)
                        saver.save(sess, os.path.join(FLAGS.save_model_dir, 'model'), global_step=global_step)

                    if (result['global_step'] + 1) % FLAGS.valid_freq == 0 or step == 0:
                        print("validate model...")  #TODO
                        # valid_cost = 0.0
                        # fetches = {'cost_valid': cost_valid, 'predict_valid': predict}
                        # # net_valid, cost_valid, encode_net_valid, encode_net_valid, _, predict
                        # valid_num = 100
                        # for valid_index in range(valid_num):
                        #     valid_data = valid_data_list[valid_index]
                        #     feed_dict = {
                        #         x_train: valid_data['x_list'], y_train: valid_data['y_list'],
                        #         target: valid_data['target_list'], x_length: valid_data['x_len_list'],
                        #         y_length: valid_data['y_len_list'], target_weight: valid_data['target_weight']
                        #     }
                        #     result_valid = sess.run(fetches, feed_dict=feed_dict)
                        #     valid_cost += result_valid['cost_valid']
                        # valid_cost /= valid_num
                        # # print("average valid cost:{:.5f} on {} sentences".format(valid_cost, valid_num * FLAGS.batch_size))
                        # printutil.mod_print("average valid cost:{:.5f} on {} sentences".format(valid_cost,
                        #                                                                        valid_num * FLAGS.batch_size),
                        #                     fg=printutil.ANSI_WHITE, bg=printutil.ANSI_GREEN_BACKGROUND,
                        #                     mod=printutil.MOD_UNDERLINE)
        print('optimization finished!')


# 6.testing
def test(FLAGS):
    print("start test faster-rcnn model")
    # TODO: load pretrained model and run test
