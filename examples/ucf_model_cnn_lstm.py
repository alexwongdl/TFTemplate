"""
Created by Alex Wang
on 2018-06-11
reference:Long-term Recurrent Convolutional Networks for Visual Recognition and Description

export CUDA_VISIBLE_DEVICES='2'
python MainEntryTrainUCF.py \
    --task=train_ucf_lstm \
    --is_training=True \
    --max_iter=200000 \
    --batch_size=32 \
    --learning_rate=1 \
    --decay_step=10000 \
    --decay_rate=0.9 \
    --dropout=0.5 \
    --input_dir=`/alexwang/data \
    --save_model_dir=~/alexwang/workspace/ucf_model_1 \
    --save_model_freq=1000 \
    --print_info_freq=10 \
    --summary_dir=~/alexwang/workspace/ucf_summary_1
"""
import os
import time
import math

import tensorflow as tf
import tensorlayer as tl

from ucf_label import label_to_idx_map

slim = tf.contrib.slim

classes_num = len(label_to_idx_map)
image_size = 17
channel_num = 1024
time_step = 10
hidden_num = 200


def block_reduction_b(inputs, scope=None, reuse=None):
    """Builds Reduction-B block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockReductionB', [inputs], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                branch_0 = slim.conv2d(branch_0, 192, [3, 3], stride=2,
                                       padding='VALID', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 256, [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, 320, [7, 1], scope='Conv2d_0c_7x1')
                branch_1 = slim.conv2d(branch_1, 320, [3, 3], stride=2,
                                       padding='VALID', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                           scope='MaxPool_1a_3x3')
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])


def block_inception_c(inputs, scope=None, reuse=None):
    """Builds Inception-C block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockInceptionC', [inputs], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = tf.concat(axis=3, values=[
                    slim.conv2d(branch_1, 256, [1, 3], scope='Conv2d_0b_1x3'),
                    slim.conv2d(branch_1, 256, [3, 1], scope='Conv2d_0c_3x1')])
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 448, [3, 1], scope='Conv2d_0b_3x1')
                branch_2 = slim.conv2d(branch_2, 512, [1, 3], scope='Conv2d_0c_1x3')
                branch_2 = tf.concat(axis=3, values=[
                    slim.conv2d(branch_2, 256, [1, 3], scope='Conv2d_0d_1x3'),
                    slim.conv2d(branch_2, 256, [3, 1], scope='Conv2d_0e_3x1')])
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 256, [1, 1], scope='Conv2d_0b_1x1')
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])


def inception_lstm_model(x_input, y_input, x_length, reuse, is_training, dropout, batch_size):
    """
    :param x_input:(batch_size, time_step, 17, 17, 1024)
    :param y_input:(batch_size, class_num)
    :param reuse:
    :param is_training:
    :return:
    """
    with tf.variable_scope('Inception_LSTM', 'InceptionV4_LSTM', [x_input, y_input], reuse=reuse) as scope:
        initializer = tf.random_uniform_initializer(-0.04, 0.04)
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # reshape x_input to (batch_size * time_step, 17, 17, 1024)
            x_input_squeeze = tf.reshape(x_input,
                                         (batch_size * time_step,
                                          image_size, image_size, channel_num),
                                         name='x_input_squeeze')

            net = block_reduction_b(x_input_squeeze, 'Mixed_7a')
            for idx in range(3):
                block_scope = 'Mixed_7' + chr(ord('b') + idx)
                net = block_inception_c(net, block_scope)

            kernel_size = net.get_shape()[1:3]
            net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='AvgPool_1a')
            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
            # 1 x 1 x 1536
            # net = slim.dropout(net, dropout, scope='Dropout_1b')
            net = slim.flatten(net, scope='PreLogitsFlatten')

            # reshape x_input to (batch_size , time_step, 1536)
            net = tf.reshape(net, (batch_size, time_step, -1), name='x_input_flat')
            # net = tf.reshape(net, (batch_size, -1), name='x_input_flat')
            net = tl.layers.InputLayer(net)
            encode_net = tl.layers.DropoutLayer(net, keep=dropout, is_fix=True, is_train=is_training,
                                                name='source_dropout_embed')
            encode_net = tl.layers.DynamicRNNLayer(encode_net, cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
                                                   cell_init_args={'forget_bias': 0.0, 'state_is_tuple': True},
                                                   n_hidden=hidden_num, initializer=initializer, sequence_length=x_length,
                                                   return_seq_2d=True, n_layer=2, return_last=True, name='encode_rnn')

            net = tf.reshape(encode_net.outputs, (batch_size, hidden_num))
            # 1536
            logits = slim.fully_connected(net, classes_num, activation_fn=None, scope='Logits')

            cost_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_input, logits=logits, name='cross_entropy'))

            L2 = 0
            for w in tl.layers.get_variables_with_name('Inception_LSTM', True, True):
                L2 += tf.contrib.layers.l2_regularizer(0.0001)(w)
            cost = cost_entropy + L2
            correct_prediction = tf.equal(tf.argmax(logits, 1), y_input)
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            y_pred_softmax = tf.argmax(logits, 1)

            return net, cost, acc, y_pred_softmax, cost_entropy, L2


def _parse_ucf_features(record):
    features = {"img": tf.FixedLenFeature((), tf.string, default_value=''),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0),
                "width": tf.FixedLenFeature((), tf.int64, default_value=0),
                "height": tf.FixedLenFeature((), tf.int64, default_value=0),
                "channel": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(record, features)
    # for key in parsed_features:
    #     print(key, type(parsed_features[key]))

    # print(type(parsed_features['img']))
    img = tf.decode_raw(parsed_features['img'], tf.float32)
    img_reshape = tf.reshape(img, (
        tf.stack([time_step, parsed_features['width'], parsed_features['height'], channel_num])))
    return img_reshape, parsed_features['width'], parsed_features['height'], parsed_features['channel'], \
           parsed_features['label'], tf.constant(10)


def train_model(FLAGS):
    """
    :param FLAGS:
    :return:
    """
    tfrecords_list = [os.path.join(FLAGS.input_dir, 'ucf_train_data.tfrecord')]
    dataset = tf.data.TFRecordDataset(tfrecords_list)
    dataset = dataset.map(_parse_ucf_features)
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.repeat(FLAGS.max_iter).batch(FLAGS.batch_size)

    iterator = dataset.make_initializable_iterator()
    next_elem = iterator.get_next()
    img_reshape, img_width, img_height, img_channel, img_label, x_length = next_elem

    # valid
    dataset_valid = tf.data.TFRecordDataset([os.path.join(FLAGS.input_dir, 'ucf_valid_data.tfrecord')])
    dataset_valid = dataset_valid.map(_parse_ucf_features).shuffle(buffer_size=200)
    dataset_valid = dataset_valid.repeat(-1).batch(32)
    iterator_valid = dataset_valid.make_initializable_iterator()
    img_reshape_valid, img_width_valid, img_height_valid, img_channel_valid, img_label_valid, x_length_valid = \
        iterator_valid.get_next()

    ## build model op
    batch_size = FLAGS.batch_size
    x_input = tf.placeholder(dtype=tf.float32, shape=(None, time_step, image_size, image_size, channel_num),
                             name='x_input')
    y_input = tf.placeholder(dtype=tf.int64, shape=(None), name='y_input')
    x_length_input = tf.placeholder(dtype=tf.int32, shape=(None), name='x_length_input')
    dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=(), name='dropout')
    net, cost, acc, y_pred_softmax, cost_entropy, L2 = inception_lstm_model(
        x_input, y_input, x_length_input, reuse=False, is_training=True, dropout=FLAGS.dropout,
        batch_size=batch_size)

    x_input_valid = tf.placeholder(dtype=tf.float32, shape=(None, time_step, image_size, image_size, channel_num),
                                   name='x_input')
    y_input_valid = tf.placeholder(dtype=tf.int64, shape=(None), name='y_input')
    x_len_input_valid = tf.placeholder(dtype=tf.int32, shape=(None), name='x_len_input_valid')
    dropout_valid = tf.placeholder(dtype=tf.float32, shape=(), name='dropout')

    net_valid, cost_valid, acc_valid, y_pred_softmax_valid, cost_entropy_valid, L2_valid = \
        inception_lstm_model(x_input_valid, y_input_valid, x_len_input_valid, reuse=True, is_training=False,
                             dropout=1.0, batch_size=batch_size)

    ## train op
    global_step = tf.train.get_or_create_global_step()
    inc_global_step = tf.assign(global_step, global_step + 1)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                               FLAGS.decay_step, FLAGS.decay_rate,
                                               staircase=True)


    # train_vars = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, train_vars), clip_norm=5, name='clip_grads')
    # train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).apply_gradients(zip(grads, train_vars))
    ## use AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(cost)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

    ## summary op
    cost_summary = tf.summary.scalar('cost', cost)
    learning_rate_summary = tf.summary.scalar('learning_rate', learning_rate)
    cost_entropy_summary = tf.summary.scalar('cost_entropy', cost_entropy)
    acc_summary = tf.summary.scalar('acc', acc)
    acc_valid_summary = tf.summary.scalar('acc_valid', acc_valid)
    L2_summary = tf.summary.scalar('L2', L2)

    ## tf.summary.merge_all is deprecated
    # summary_op = tf.summary.merge_all()
    summary_op = tf.summary.merge([cost_summary, learning_rate_summary,
                                   cost_entropy_summary, acc_summary,
                                   L2_summary])

    ## training/ valid
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        sess.run(iterator_valid.initializer)
        ## tf.train.SummaryWriter is deprecated
        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, graph=sess.graph)

        if FLAGS.checkpoint is not None:
            saver.restore(sess, FLAGS.checkpoint)

        for step in range(FLAGS.max_iter):
            start_time = time.time()
            fetches = {'train_op': train_op,
                       'global_step': global_step,
                       'inc_global_step': inc_global_step}

            if (step + 1) % FLAGS.print_info_freq == 0 or step == 0:
                fetches['cost'] = cost
                fetches['cost_entropy'] = cost_entropy
                fetches['learning_rate'] = learning_rate
                fetches['accuracy'] = acc

            if (step + 1) % FLAGS.summary_freq == 0:
                fetches['summary_op'] = summary_op

            img_reshape_val, img_width_val, img_height_val, img_channel_val, img_label_val, input_len_val = \
                sess.run([img_reshape, img_width, img_height, img_channel, img_label, x_length])
            # print('img_reshape_val', img_reshape_val)
            # print('img_width_val', img_width_val)
            # print('img_height_val', img_height_val)
            # print('img_channel_val', img_channel_val)
            # print('img_label_val', img_label_val)

            result = sess.run(fetches, feed_dict={
                x_input: img_reshape_val,
                y_input: img_label_val,
                dropout_placeholder: FLAGS.dropout,
                x_length_input: input_len_val})

            if (step + 1) % FLAGS.save_model_freq == 0:
                print("save model")
                if not os.path.exists(FLAGS.save_model_dir):
                    os.mkdir(FLAGS.save_model_dir)
                saver.save(sess, os.path.join(FLAGS.save_model_dir, 'model'), global_step=global_step)

            if (step + 1) % FLAGS.summary_freq == 0:
                summary_writer.add_summary(result['summary_op'], result['global_step'])

            if (step + 1) % FLAGS.print_info_freq == 0 or step == 0:
                epoch = math.ceil(result['global_step'] * 1.0 / FLAGS.print_info_freq)
                rate = FLAGS.batch_size / (time.time() - start_time)
                print("epoch:{}\t, rate:{:.2f} image/sec".format(epoch, rate))
                print("global step:{}".format(result['global_step']))
                print("cost:{:.4f}".format(result['cost']))
                print("cost entropy:{:.4f}".format(result['cost_entropy']))
                print("accuracy:{:.4f}".format(result['accuracy']))
                print("learning rate:{:.6f}".format(result['learning_rate']))
                print("")

            if (step + 1) % FLAGS.valid_freq == 0:
                batch_num = 43
                accuracy_average = 0
                for i_valid in range(batch_num):
                    img_reshape_a, img_width_a, img_height_a, img_channel_a, img_label_a, input_len_a = \
                        sess.run([img_reshape_valid, img_width_valid,
                                  img_height_valid, img_channel_valid, img_label_valid, x_length_valid])
                    accuracy, summary_str, global_step_val = sess.run(
                        [acc_valid, acc_valid_summary, global_step],
                        feed_dict={
                            x_input_valid: img_reshape_a,
                            y_input_valid: img_label_a,
                            dropout_valid: 1.0,
                            x_len_input_valid: input_len_a
                        })
                    summary_writer.add_summary(summary_str, global_step_val + i_valid)

                    print('valid accuracy:{:.4f}'.format(accuracy))
                    accuracy_average += accuracy
                accuracy_average /= batch_num
                print('valid average accuracy:{:.4f}'.format(accuracy_average))

        summary_writer.close()


if __name__ == '__main__':
    train_model()
