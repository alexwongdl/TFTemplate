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


def rnn_model(x_input, y_input, reuse, is_training, FLAGS):
    print('construct rnn model')
    # rnn_mode - the low level implementation of lstm cell: one of CUDNN, BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and lstm_block_cell classes.
    # TODO: construct graph
    initializer = tf.random_normal_initializer(-FLAGS.init_scale, FLAGS.init_scale)
    with tf.variable_scope('ptb_model', reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = tl.layers.EmbeddingInputlayer(x_input, vocabulary_size=FLAGS.vocab_size, embedding_size=FLAGS.vocab_dim, E_init= initializer, name='embedding')
        net = tl.layers.DropoutLayer(net, keep=FLAGS.keep_prob, is_fix=True, is_training= is_training, name='dropout_embed')
        

def train_rnn(FLAGS):
    print("start train rnn model")
    # 2.load data
    train_data, test_data, valid_data, word_to_id, id_to_word = ptb_reader.ptb_raw_data(path=FLAGS.input_dir)
    ## input queue
    x_train, y_train, train_epoch_size = ptb_reader.ptb_data_queue(train_data, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps)

    # 3.build graph：including loss function，learning rate decay，optimization operation
    net, cost, _, _ = rnn_model(x_train, y_train, False, is_training=True)  # train

    x_test = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_steps], name='x_test')
    y_test = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_steps], name='y_test')
    x_valid_data, y_valid_data, valid_epoch_size = ptb_reader.ptb_data_batch(valid_data, FLAGS.batch_size,
                                                                             FLAGS.num_steps)
    _, _, acc, y_pred = rnn_model(x_test, y_test, True, is_training=False)  # validate/test
    validate_batch_num = valid_epoch_size

    steps_per_epoch = train_epoch_size
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                               staircase=True)
    # learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step,
    #                                           end_learning_rate=0.00001)
    incr_global_step = tf.assign(global_step, global_step + 1)

    train_params = net.all_params
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(
            cost, var_list=train_params)

    # 4.TODO:summary
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
                for i in range(validate_batch_num):
                    batch_x_test = x_valid_data[i]
                    batch_y_test = y_valid_data[i]
                    test_result = sess.run(fetches, feed_dict={x_test: batch_x_test, y_test: batch_y_test})
                    total_acc += test_result['accuracy'] * FLAGS.batch_size
                    total_num += FLAGS.batch_size
                    if i < 10:
                        print('check data')
                        #TODO:
                        # print("label:{}".format(batch_label_test))
                        # print("predict:{}".format(test_result['y_pred']))
                valid_acc = total_acc / total_num
                print("valid accuracy:{:.5f} for {} images".format(valid_acc, total_num))

        print('optimization finished!')

        # 6.testing
