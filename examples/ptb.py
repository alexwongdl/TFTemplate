"""
Created by Alex Wang
On 2017-10-29
Recurrent Neural Networks on Penn Tree Bank dataset:https://www.tensorflow.org/tutorials/recurrent
"""
import math
import os
import time

import tensorflow as tf
import tensorlayer as tl

import ptb_reader

def rnn_model(x_input, y_input, reuse, is_training):
    print('construct rnn model')
    #TODO: construct graph

def train_rnn(FLAGS):
    print("start train rnn model")
    # 2.load data
    train_data, test_data, valid_data, word_to_id, id_to_word = ptb_reader.ptb_raw_data(path=FLAGS.input_dir)

    ##TODO: input queue

    # 3.build graph：including loss function，learning rate decay，optimization operation
    net, cost, _, _ = rnn_model(x_input, y_input, False, is_training=True)  # train
    _, _, acc, y_pred = rnn_model(x_batch_val, y_batch_val, True, is_training=False)  # validate/test
    validate_batch_num = int(X_val.shape[0] / FLAGS.batch_size)

    steps_per_epoch = int(math.ceil(X_train.shape[0] / FLAGS.batch_size))
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                               staircase=True)
    # learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step,
    #                                           end_learning_rate=0.00001)
    incr_global_step = tf.assign(global_step, global_step + 1)

    train_params = net.all_params
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(
            cost, var_list=train_params)

    # 4.summary
    input_images = tf.image.convert_image_dtype(x_input, dtype=tf.uint8, saturate=True)
    with tf.name_scope("input_summary"):
        tf.summary.image("input_summary", input_images)
    tf.summary.scalar('cost', cost)
    tf.summary.scalar('learning_rate', learning_rate)

    # 5.training、valid、save check point in loops
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sv = tf.train.Supervisor(logdir=FLAGS.summary_dir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:
        print('start optimization...')
        # load check point if FLAGS.checkpoint is not None
        if FLAGS.checkpoint is not None:
            saver.restore(sess, FLAGS.checkpoint)

        for step in range(FLAGS.max_iter):
            start_time = time.time()
            fetches = {'train_op': train_op, 'global_step': global_step, 'inc_global_step': incr_global_step}
            if (step + 1) % FLAGS.print_info_freq == 0:
                fetches['cost'] = cost
                fetches['learning_rate'] = learning_rate

            if (step + 1) % FLAGS.summary_freq == 0:
                fetches['summary_op'] = sv.summary_op

            result = sess.run(fetches)

            if (step + 1) % FLAGS.print_info_freq == 0:
                epoch = math.ceil(result['global_step'] * 1.0 / steps_per_epoch)
                rate = FLAGS.batch_size / (time.time() - start_time)
                print("epoch:{}\t, rate:{:.2f} image/sec".format(epoch, rate))
                print("global step:{}".format(result['global_step']))
                print("cost:{:.4f}".format(result['cost']))
                print("learning rate:{:.6f}".format(result['learning_rate']))
                print()

            if (step + 1) % FLAGS.save_model_freq == 0:
                print("save model")
                if not os.path.exists(FLAGS.save_model_dir):
                    os.mkdir(FLAGS.save_model_dir)
                saver.save(sess, os.path.join(FLAGS.save_model_dir, 'model'), global_step=global_step)

            if (step + 1) % FLAGS.valid_freq == 0 or step == 0:
                print("validate model...")
                fetches['accuracy'] = acc
                fetches['y_pred'] = y_pred
                total_acc = 0
                total_num = 0
                for i in range(validate_batch_num - 1):
                    batch_img_test = X_val[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
                    batch_label_test = y_val[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
                    test_result = sess.run(fetches, feed_dict={x_place: batch_img_test, y_place: batch_label_test})
                    total_acc += test_result['accuracy'] * FLAGS.batch_size
                    total_num += FLAGS.batch_size
                    if i < 10:
                        print("label:{}".format(batch_label_test))
                        print("predict:{}".format(test_result['y_pred']))
                valid_acc = total_acc / total_num
                print("valid accuracy:{:.5f} for {} images".format(valid_acc, total_num))

        print('optimization finished!')

        # 6.testing