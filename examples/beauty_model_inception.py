"""
Created by Alex Wang
convert 0~5 value to 0~100 score , use classification model
"""

import os
import time
import math
import shutil

from scipy.stats import pearsonr
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorlayer as tl

import inception_v4
import inception_preprocessing

image_size = inception_v4.inception_v4.default_image_size


def build_inception_model(x_input, y_input, reg_input, reuse, is_training, FLAGS):
    dropout = FLAGS.dropout
    batch_size = FLAGS.batch_size
    arg_scope = inception_v4.inception_v4_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_v4.inception_v4(x_input, is_training=is_training,
                                                       num_classes=1001,
                                                       dropout_keep_prob=dropout,
                                                       reuse=reuse,
                                                       create_aux_logits=False)

    with tf.variable_scope('Beauty', 'BeautyV1', reuse=reuse) as scope:
        # added by Alex Wang for face beauty predict
        # preidct_dropout = slim.dropout(end_points['PreLogitsFlatten'], dropout, scope='Dropout_1b')
        mid_full_conn = slim.fully_connected(end_points['PreLogitsFlatten'],
                                             1000, activation_fn=tf.nn.relu,
                                             scope='mid_full_conn',
                                             trainable=is_training,
                                             reuse=reuse)

        predict_conn = slim.fully_connected(mid_full_conn,
                                            100, activation_fn=None,
                                            scope='100_class_conn',
                                            trainable=is_training,
                                            reuse=reuse)

        beauty_weight = tf.convert_to_tensor([[i] for i in range(0, 100)], dtype=tf.float32)
        regress_conn = tf.matmul(tf.nn.softmax(predict_conn), beauty_weight)  # 32 * 1

        y_pred_softmax = tf.argmax(predict_conn, 1)
        correct_prediction = tf.equal(y_pred_softmax, y_input)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        reg_input_reshape = tf.reshape(reg_input, shape=(batch_size, 1))

        diff = tf.subtract(tf.cast(reg_input_reshape, tf.float32),
                           tf.cast(regress_conn, tf.float32))
        cost_rmse = tf.reduce_mean(tf.square(diff), name='cost_rmse')
        # cost_rmse = tf.reduce_mean(tf.exp(tf.abs(diff)), name='cost_rmse')

        ## define cost
        cost_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_input, logits=predict_conn, name='cross_entropy'))
        L2 = 0
        for w in tl.layers.get_variables_with_name('InceptionV4', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.0001)(w)
        for w in tl.layers.get_variables_with_name('Beauty', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.0001)(w)

        # cost = 10000 * cost_entropy + L2 + 0.001 * cost_rmse
        # cost = 10 * cost_entropy + L2 + 100 * cost_rmse
        # cost = L2 + 0.01 * cost_rmse + 0.01 * cost_entropy

        cost = L2 + 1000 * cost_rmse

        end_points['cost_rmse'] = cost_rmse
        end_points['predict'] = y_pred_softmax
        end_points['regress_conn'] = regress_conn
        end_points['predict_conn'] = predict_conn
        end_points['regress_label'] = diff
        end_points['predict_softmax'] = tf.nn.softmax(predict_conn)
        end_points['beauty_weight'] = beauty_weight
        end_points['acc'] = acc
        end_points['cost_entropy'] = cost_entropy
        end_points['L2'] = L2
        end_points['cost'] = cost

    return end_points


def _parse_ucf_features_train(record):
    features = {"img": tf.FixedLenFeature((), tf.string, default_value=''),
                "label": tf.FixedLenFeature((), tf.float32, default_value=0),
                "score": tf.FixedLenFeature((), tf.int64, default_value=0),
                "width": tf.FixedLenFeature((), tf.int64, default_value=0),
                "height": tf.FixedLenFeature((), tf.int64, default_value=0),
                "channel": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(record, features)

    img = tf.decode_raw(parsed_features['img'], tf.float32)
    img_reshape = tf.reshape(img, (
        tf.stack([parsed_features['width'], parsed_features['height'], parsed_features['channel']])))

    img_reshape = tf.image.random_flip_left_right(img_reshape)
    img_reshape = tf.image.random_brightness(img_reshape, max_delta=10. / 255.)
    img_reshape = tf.image.random_saturation(img_reshape, lower=0.9, upper=1.1)
    angles = tf.random_uniform(shape=(1,), minval=-0.2, maxval=0.2,
                               dtype=tf.float32,
                               seed=tf.set_random_seed(1234), name=None)

    noise = tf.random_normal(shape=tf.shape(img_reshape), mean=0.0, stddev=0.01, dtype=tf.float32)
    img_reshape = img_reshape + noise

    img_reshape = tf.contrib.image.rotate(tf.expand_dims(img_reshape, 0), angles)
    img_reshape = inception_preprocessing.preprocess_for_eval(
        tf.squeeze(img_reshape), image_size, image_size,
        central_fraction=0.9, scope='preprocess_train')

    return img_reshape, parsed_features['width'], parsed_features['height'], parsed_features['channel'], \
           parsed_features['score'] * 5, parsed_features['label'] * 20


def _parse_ucf_features_test(record):
    features = {"img": tf.FixedLenFeature((), tf.string, default_value=''),
                "label": tf.FixedLenFeature((), tf.float32, default_value=0),
                "score": tf.FixedLenFeature((), tf.int64, default_value=0),
                "width": tf.FixedLenFeature((), tf.int64, default_value=0),
                "height": tf.FixedLenFeature((), tf.int64, default_value=0),
                "channel": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(record, features)
    # for key in parsed_features:
    #     print(key, type(parsed_features[key]))

    # print(type(parsed_features['img']))
    img = tf.decode_raw(parsed_features['img'], tf.float32)
    img_reshape = tf.reshape(img, (
        tf.stack([parsed_features['width'], parsed_features['height'], parsed_features['channel']])))
    img_reshape = inception_preprocessing.preprocess_for_eval(
        img_reshape, image_size, image_size,
        central_fraction=1, scope='preprocess_test')

    return img_reshape, parsed_features['width'], parsed_features['height'], parsed_features['channel'], \
           parsed_features['score'] * 5, parsed_features['label'] * 20


def train_model(FLAGS):
    batch_size = FLAGS.batch_size

    tfrecords_list = [os.path.join(FLAGS.input_dir, 'train_tfrecords_5')]
    dataset = tf.data.TFRecordDataset(tfrecords_list)

    dataset = dataset.map(_parse_ucf_features_train)
    dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.repeat(-1).batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_elem = iterator.get_next()
    img_reshape, img_width, img_height, img_channel, img_label, img_reg = next_elem

    # valid
    # dataset_valid = tf.data.TFRecordDataset([os.path.join(FLAGS.input_dir, 'test_tfrecords_5')])

    dataset_valid = tf.data.TFRecordDataset([os.path.join(FLAGS.input_dir, 'test_tfrecords_5')])
    dataset_valid = dataset_valid.map(_parse_ucf_features_test).shuffle(buffer_size=500)
    dataset_valid = dataset_valid.repeat(-1).batch(batch_size)
    iterator_valid = dataset_valid.make_initializable_iterator()
    next_elem_valid = iterator_valid.get_next()
    img_reshape_valid, img_width_valid, img_height_valid, \
    img_channel_valid, img_label_valid, img_reg_valid = next_elem_valid

    # build model
    x_input = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3))
    y_input = tf.placeholder(tf.int64, shape=(None))
    reg_input = tf.placeholder(tf.float32, shape=(None))
    end_points_train = build_inception_model(x_input, y_input, reg_input, reuse=False,
                                             is_training=True, FLAGS=FLAGS)

    x_input_valid = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3))
    y_input_valid = tf.placeholder(tf.int64, shape=(None))
    reg_input_valid = tf.placeholder(tf.float32, shape=(None))
    end_point_test = build_inception_model(x_input_valid, y_input_valid,
                                           reg_input_valid, reuse=True,
                                           is_training=False, FLAGS=FLAGS)

    ## TODO: should defined before train_op
    # https://github.com/tensorflow/tensorflow/issues/7244
    variables = slim.get_variables_to_restore()
    variables_to_restore = [v for v in variables if v.name.split('/')[0] == 'InceptionV4']

    ## train op
    global_step = tf.train.get_or_create_global_step()
    inc_global_step = tf.assign(global_step, global_step + 1)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                               FLAGS.decay_step, FLAGS.decay_rate,
                                               staircase=True)

    trainable_variables = []
    trainable_variables.extend(tf.trainable_variables(scope='InceptionV4/Mixed_6a'))
    trainable_variables.extend(tf.trainable_variables(scope='InceptionV4/Mixed_6b'))
    trainable_variables.extend(tf.trainable_variables(scope='InceptionV4/Mixed_6c'))
    trainable_variables.extend(tf.trainable_variables(scope='InceptionV4/Mixed_6d'))
    trainable_variables.extend(tf.trainable_variables(scope='InceptionV4/Mixed_6e'))
    trainable_variables.extend(tf.trainable_variables(scope='InceptionV4/Mixed_6f'))
    trainable_variables.extend(tf.trainable_variables(scope='InceptionV4/Mixed_6g'))
    trainable_variables.extend(tf.trainable_variables(scope='InceptionV4/Mixed_6h'))
    trainable_variables.extend(tf.trainable_variables(scope='InceptionV4/Mixed_7a'))
    trainable_variables.extend(tf.trainable_variables(scope='InceptionV4/Mixed_7b'))
    trainable_variables.extend(tf.trainable_variables(scope='InceptionV4/Mixed_7c'))
    trainable_variables.extend(tf.trainable_variables(scope='InceptionV4/Mixed_7d'))
    trainable_variables.extend(tf.trainable_variables(scope='InceptionV4/Logits'))
    trainable_variables.extend(tf.trainable_variables(scope='Beauty'))

    print('trainable_variables:')
    print(trainable_variables)
    # TODO: * use 'slim.learning.create_train_op' instead of 'optimizer.minimize'
    # and add update_ops
    # https://blog.csdn.net/qq_25737169/article/details/79616671
    # train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(
    #     end_points_train['cost'])
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False)
    train_op = slim.learning.create_train_op(end_points_train['cost'], optimizer,
                                             global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    ## summary op
    cost_summary = tf.summary.scalar('cost', end_points_train['cost'])
    acc_summary = tf.summary.scalar('acc', end_points_train['acc'])
    learning_rate_summary = tf.summary.scalar('learning_rate', learning_rate)
    cost_rmse_summary = tf.summary.scalar('cost_rmse', end_points_train['cost_rmse'])
    cost_entropy_summary = tf.summary.scalar('cost_entropy', end_points_train['cost_entropy'])
    L2_summary = tf.summary.scalar('L2', end_points_train['L2'])
    rmse_valid_summary = tf.summary.scalar('rmse_valid', end_point_test['cost_rmse'])
    image_summary = tf.summary.image('input_img', img_reshape)

    regress_conn_summary = tf.summary.tensor_summary('regress_conn', end_points_train['regress_conn'])
    y_input_summary = tf.summary.tensor_summary('y_input', y_input)
    regress_label_summary = tf.summary.tensor_summary('regress_label', end_points_train['regress_label'])

    ## tf.summary.merge_all is deprecated
    # summary_op = tf.summary.merge_all()
    summary_op = tf.summary.merge([cost_summary, learning_rate_summary,
                                   cost_entropy_summary, acc_summary,
                                   cost_rmse_summary, image_summary,
                                   L2_summary, regress_conn_summary,
                                   y_input_summary, regress_label_summary])

    saver = tf.train.Saver(variables_to_restore)
    saver_all = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if not os.path.exists(FLAGS.save_model_dir):
        os.mkdir(FLAGS.save_model_dir)
    if not os.path.exists(FLAGS.summary_dir):
        os.mkdir(FLAGS.summary_dir)
    log_path = os.path.join(FLAGS.summary_dir, 'result.log')
    with tf.Session(config=config) as sess, open(log_path, 'w') as writer:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
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
                       'inc_global_step': inc_global_step,
                       'update_ops': update_ops}

            if (step + 1) % FLAGS.print_info_freq == 0 or step == 0:
                fetches['cost'] = end_points_train['cost']
                fetches['acc'] = end_points_train['acc']
                fetches['regress_conn'] = end_points_train['regress_conn']
                fetches['predict_conn'] = end_points_train['predict_conn']
                fetches['regress_label'] = end_points_train['regress_label']
                fetches['predict_softmax'] = end_points_train['predict_softmax']
                fetches['beauty_weight'] = end_points_train['beauty_weight']
                fetches['cost_rmse'] = end_points_train['cost_rmse']
                fetches['cost_entropy'] = end_points_train['cost_entropy']
                fetches['learning_rate'] = learning_rate
                fetches['L2'] = end_points_train['L2']
                fetches['predict'] = end_points_train['predict']

            if (step + 1) % FLAGS.summary_freq == 0:
                fetches['summary_op'] = summary_op

            img_reshape_val, img_width_val, img_height_val, \
            img_channel_val, img_label_val, img_reg_val = \
                sess.run([img_reshape, img_width, img_height,
                          img_channel, img_label, img_reg])
            # print("shape of img_reshape_val:{}, shape of img_label_val:{}".format(
            #     img_reshape_val.shape, img_label_val.shape
            # ))

            result = sess.run(fetches, feed_dict={
                x_input: img_reshape_val,
                y_input: img_label_val,
                reg_input: img_reg_val
            })

            if (step + 1) % FLAGS.save_model_freq == 0:
                print("save model")
                saver_all.save(sess, os.path.join(FLAGS.save_model_dir, 'model'), global_step=global_step)

            if (step + 1) % FLAGS.summary_freq == 0:
                summary_writer.add_summary(result['summary_op'], result['global_step'])

            if (step + 1) % FLAGS.print_info_freq == 0 or step == 0:
                epoch = math.ceil(result['global_step'] * 1.0 / FLAGS.print_info_freq)
                rate = FLAGS.batch_size / (time.time() - start_time)
                print("epoch:{}\t, rate:{:.2f} image/sec".format(epoch, rate))
                print("global step:{}".format(result['global_step']))
                print("cost:{:.6f}".format(result['cost']))
                print("acc:{:.4f}".format(result['acc']))
                print("cost entropy:{:.6f}".format(result['cost_entropy']))
                print("cost rmse:{:.6f}".format(result['cost_rmse']))
                print("np cost rmse:{:.6f}".format(np.mean(np.square(result['regress_label'].flat))))
                test = np.subtract(result['regress_conn'], np.reshape(img_label_val, newshape=(batch_size, 1)))
                # print("regression_conn - img_label_val:{}".format(test.flat[:]))
                # print("rmse sum:{:.6f}, length of img_label_val:{}".
                #       format(np.sum(np.abs(test)), len(img_label_val)))
                print("shape of predict_conn:{}".format(result["predict_conn"].shape))
                print("shape of regression_conn:{}".format(result["regress_conn"].shape))
                print("shape of label:{}".format(img_label_val.shape))
                print("L2:{:.6f}".format(result['L2']))
                print("learning rate:{:.10f}".format(result['learning_rate']))
                print("regress_conn:{}".format(result["regress_conn"].flat[0:32]))
                print("regression_conn - img_label_val tf:{}".format(result['regress_label'].flat[0:32]))

                print("label:{}".format(img_label_val[0:32]))
                print("reg_label:{}".format(img_reg_val[0:32]))
                print("predict:{}".format(result['predict'][0:32]))
                print("predict_softmax:{}".format(result['predict_softmax'].flat[0:20]))
                # print("shape of predict_softmax:{}, shape of beauty weight:{}".format(
                #     result['predict_softmax'].shape, result['beauty_weight'].shape
                # ))
                print("summary:{}".format(FLAGS.summary_dir))
                print("")

            if (step + 1) % FLAGS.valid_freq == 0:
                batch_num = int(1100 / batch_size)
                accuracy_average = 0
                rmse_average = 0
                pearson_average = 0
                for i_valid in range(batch_num):
                    img_reshape_a, img_width_a, img_height_a, \
                    img_channel_a, img_label_a, img_reg_a = \
                        sess.run([img_reshape_valid, img_width_valid,
                                  img_height_valid, img_channel_valid,
                                  img_label_valid, img_reg_valid])

                    accuracy, rmse, predict_valid, regress_conn_valid, \
                    summary_str, global_step_val, predict_conn_valid = sess.run(
                        [end_point_test['acc'], end_point_test['cost_rmse'],
                         end_point_test['predict'], end_point_test['regress_conn'],
                         rmse_valid_summary, global_step, end_point_test['predict_conn']],
                        feed_dict={
                            x_input_valid: img_reshape_a,
                            y_input_valid: img_label_a,
                            reg_input_valid: img_reg_a
                        })
                    summary_writer.add_summary(summary_str, global_step_val + i_valid)

                    pearson_val = pearsonr(img_reg_a.flat[:], regress_conn_valid.flat[:])[0]
                    if i_valid == 0:
                        print('predict:{}'.format(predict_valid))
                        print('label:{}'.format(img_label_a))
                        print('predict_conn_valid:{}'.format(predict_conn_valid.flat[0:20]))
                        # print('reg_label:{}'.format(img_reg_a))
                        print('regress_conn:{}'.format(regress_conn_valid.flat[:]))
                    print('valid acc:{:.4f}, valid rmse:{:.4f}, pearson cor:{:.3f}'.
                          format(accuracy, rmse, pearson_val))
                    accuracy_average += accuracy
                    rmse_average += rmse
                    pearson_average += pearson_val
                accuracy_average /= batch_num
                rmse_average /= batch_num
                pearson_average /= batch_num
                print('valid av_acc:{:.4f}, av_rmse:{:.4f}, av_peason:{:.4f}'.
                      format(accuracy_average, rmse_average, pearson_average))
                writer.write('iter{}, valid av_acc:{:.4f}, av_rmse:{:.4f}, av_peason:{:.4f}\n'.
                             format(step, accuracy_average, rmse_average, pearson_average))
                writer.flush()

                if pearson_average >= 0.89:
                    shutil.copytree(FLAGS.save_model_dir,
                                    "{}_{:.5f}_{}".format(FLAGS.save_model_dir, pearson_average, global_step_val))
        summary_writer.close()
