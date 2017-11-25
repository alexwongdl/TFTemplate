"""
Created by Alex Wang
On 2017-10-27
Neural Machine Translation: https://github.com/tensorflow/nmt#introduction

decoder / attention wrapper API：https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/seq2seq/python/ops
attention_wrapper.py
basic_decoder.py
beam_search_decoder.py
beam_search_ops.py
decoder.py
helper.py
loss.py

Corpus：
        * French-English:http://113.215.21.41/1Q2W3E4R5T6Y7U8I9O0P1Z2X3C4V5B/www.statmt.org/wmt10/training-giga-fren.tar
        https://nlp.stanford.edu/projects/nmt/
        http://www.statmt.org/moses/?n=Moses.LinksToCorpora
        https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus

dataset：
        training-giga-fren.tar(French-English)：http://www.statmt.org/wmt10/training-giga-fren.tar
"""

import tensorflow as tf
import tensorlayer as tl
import translation_data_prepare

# tl.layers.Seq2Seq()

def rnn_model(x_input, y_input, x_length, y_length, target_weight, reuse, is_training, FLAGS):
    print('construct rnn model')
    # rnn_mode - the low level implementation of lstm cell: one of CUDNN, BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and lstm_block_cell classes.
    #construct graph

    ## warn: 使用tf.random_normal_initializer可能导致模型无法收敛
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
    with tf.variable_scope('nmt_model', reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        source_input = tl.layers.EmbeddingInputlayer(x_input, vocabulary_size=FLAGS.vocab_size, embedding_size=FLAGS.vocab_dim,
                                            E_init=initializer, name='source_embedding')
        target_input = tl.layers.EmbeddingInputlayer(y_input, vocabulary_size=FLAGS.vocab_size, embedding_size=FLAGS.vocab_dim,
                                            E_init=initializer, name='target_embedding')

        encode_net = tl.layers.DropoutLayer(source_input, keep=FLAGS.keep_prob, is_fix=True, is_train=is_training, name='source_dropout_embed')
        encode_net = tl.layers.DynamicRNNLayer(encode_net, cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
                                               cell_init_args={'forget_bias': 0.0, 'state_is_tuple': True}, n_hidden=FLAGS.vocab_dim,
                                               initializer=initializer, sequence_length=x_length, return_seq_2d=True,
                                               return_last=False, name='encode_rnn') #取最终结果

        decode_net = tl.layers.DropoutLayer(target_input, keep=FLAGS.keep_prob, is_fix=True, is_train=is_training, name='target_dropout_embed')
        decode_net = tl.layers.DynamicRNNLayer(decode_net, cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
                                               cell_init_args={'forget_bias': 0.0, 'state_is_tuple': True}, n_hidden=FLAGS.vocab_dim,
                                               initial_state=encode_net.final_state,
                                               initializer=initializer, sequence_length=y_length, return_seq_2d=True,
                                               return_last=False, name='decode_rnn')

        net = tl.layers.DropoutLayer(decode_net, keep=FLAGS.keep_prob, is_fix=True, is_train=is_training, name='dropout_2')
        # lstm 的输出结果用DenseLayer投影到字典上
        net = tl.layers.DenseLayer(net, n_units=FLAGS.vocab_size, W_init=initializer, b_init=initializer,
                                   act=tf.identity, name='denselayer')
        output = net.outputs
        predict = tf.reshape(tf.arg_max(output, 1), shape=[FLAGS.batch_size, -1], name='predict')

        ## loss function  tf.contrib.legacy_seq2seq.sequence_loss_by_example
        """
         Weighted cross-entropy loss for a sequence of logits (per example).

         Args:
           logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
           targets: List of 1D batch-sized int32 Tensors of the same length as logits.
           weights: List of 1D batch-sized float-Tensors of the same length as logits.
           average_across_timesteps: If set, divide the returned cost by the total
             label weight.
           softmax_loss_function: Function (labels, logits) -> loss-batch
             to be used instead of the standard softmax (the default if this is None).
             **Note that to avoid confusion, it is required for the function to accept
             named arguments.**
           name: Optional name for this operation, default: "sequence_loss_by_example".

         Returns:
           1D batch-sized float Tensor: The log-perplexity for each sequence.
        """
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits=[output], targets=[tf.reshape(y_input, [-1])], \
                                                                  weights=[tf.ones_like(tf.reshape(y_input, [-1]),
                                                                                        dtype=tf.float32)], \
                                                                  name='sequence_loss_by_example')
        cost = tf.reduce_sum(loss * target_weight) / FLAGS.batch_size

        return net, cost, encode_net, decode_net, loss, predict


def train_rnn(FLAGS):
    print("start train rnn model")
    # 2.load data/dict
    fr_word_to_id, fr_id_to_word = translation_data_prepare.load_dict(FLAGS.source_dict)
    en_word_to_id, en_id_to_word = translation_data_prepare.load_dict(FLAGS.target_dict)

    train_data, test_data, valid_data, word_to_id, id_to_word = ptb_reader.ptb_raw_data(data_path=FLAGS.input_dir)
    # x_train, y_train, train_epoch_size = ptb_reader.ptb_data_queue(train_data, batch_size=FLAGS.batch_size,
    #                                                                num_steps=FLAGS.num_steps)
    x_train_data, y_train_data, train_epoch_size = ptb_reader.ptb_data_batch(train_data, batch_size=FLAGS.batch_size,
                                                                             num_steps=FLAGS.num_steps)

    # 3.build graph：including loss function，learning rate decay，optimization operation
    x_train = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_steps], name='x_train')
    y_train = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_steps], name='y_train')
    net, cost, lstm_train_1, lstm_train_2, _, _ = rnn_model(x_train, y_train, False, is_training=True,
                                                            FLAGS=FLAGS)  # train

    # x_test = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_steps], name='x_test')
    # y_test = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_steps], name='y_test')
    x_valid_data, y_valid_data, valid_epoch_size = ptb_reader.ptb_data_batch(valid_data, FLAGS.batch_size,
                                                                             FLAGS.num_steps)
    net_valid, cost_valid, lstm_val_1, lstm_val_2, _, predict = rnn_model(x_train, y_train, True, is_training=False,
                                                                          FLAGS=FLAGS)  # validate/test
    validate_batch_num = valid_epoch_size

    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                               staircase=True)
    # learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step,
    #                                           end_learning_rate=0.00001)
    incr_global_step = tf.assign(global_step, global_step + 1)

    # train_params = net.all_params
    # train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(
    #         cost, var_list=train_params)
    train_vars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, train_vars), clip_norm=FLAGS.max_grad_norm, name='clip_grads')
    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).apply_gradients(zip(grads, train_vars))
    init_op = tf.global_variables_initializer()

    # 4.summary
    tf.summary.scalar('cost', cost)
    tf.summary.scalar('learning_rate', learning_rate)
    

    # 5.training、valid、save check point in loops
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sv = tf.train.Supervisor(logdir=FLAGS.summary_dir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:
        print('start optimization...')
        with sess.as_default():
            # tl.layers.initialize_global_variables(sess)
            init_ = sess.run(init_op)
            net.print_params()
            net.print_layers()
            tl.layers.print_all_variables()

        # load check point if FLAGS.checkpoint is not None
        if FLAGS.checkpoint is not None:
            saver.restore(sess, FLAGS.checkpoint)

        for round in range(FLAGS.max_max_epoch):
            # lstm初始化
            # with sess.as_default():
            #     state1 = tl.layers.initialize_rnn_state(lstm_train_1.initial_state)
            #     state2 = tl.layers.initialize_rnn_state(lstm_train_2.initial_state)
            state1_init_c, state1_init_h, state_init2_c, state_init2_h = sess.run(
                    [lstm_train_1.initial_state.c, lstm_train_1.initial_state.h,
                     lstm_train_2.initial_state.c, lstm_train_2.initial_state.h],
                    feed_dict={x_train: x_train_data[0], y_train: y_train_data[0]})
            state1 = (state1_init_c, state1_init_h)
            state2 = (state_init2_c, state_init2_h)

            for step in range(train_epoch_size):
                start_time = time.time()
                fetches = {'train_op': train_op, 'global_step': global_step, 'inc_global_step': incr_global_step,
                           'lstm1_final_state_c': lstm_train_1.final_state.c,
                           'lstm1_final_state_h': lstm_train_1.final_state.h,
                           'lstm2_final_state_c': lstm_train_2.final_state.c,
                           'lstm2_final_state_h': lstm_train_2.final_state.h}
                feed_dict = {
                    lstm_train_1.initial_state.c: state1[0],
                    lstm_train_1.initial_state.h: state1[1],
                    lstm_train_2.initial_state.c: state2[0],
                    lstm_train_2.initial_state.h: state2[1],
                    x_train: x_train_data[step], y_train: y_train_data[step]
                }

                if (step + 1) % FLAGS.print_info_freq == 0:
                    fetches['cost'] = cost
                    fetches['learning_rate'] = learning_rate

                if (step + 1) % FLAGS.summary_freq == 0:
                    fetches['summary_op'] = sv.summary_op

                result = sess.run(fetches, feed_dict=feed_dict)
                state1 = (result['lstm1_final_state_c'], result['lstm1_final_state_h'])
                state2 = (result['lstm2_final_state_c'], result['lstm2_final_state_h'])

                if (step + 1) % FLAGS.print_info_freq == 0:
                    rate = FLAGS.batch_size / (time.time() - start_time)
                    print("epoch:{}\t, rate:{:.2f} sentences/sec".format(round, rate))
                    print("global step:{}".format(result['global_step']))
                    print("cost:{:.4f}".format(result['cost']))
                    print("learning rate:{:.6f}".format(result['learning_rate']))
                    print(state1[0])
                    print()

                if (result['global_step'] + 1) % FLAGS.save_model_freq == 0:
                    print("save model")
                    if not os.path.exists(FLAGS.save_model_dir):
                        os.mkdir(FLAGS.save_model_dir)
                    saver.save(sess, os.path.join(FLAGS.save_model_dir, 'model'), global_step=global_step)

                if (result['global_step'] + 1) % FLAGS.valid_freq == 0 or step == 0:
                    print("validate model...")
                    # 初始化lstm

                    # print("valid cost:{:.5f} for {} sentences".format(valid_cost, total_num))

        print('optimization finished!')

# 6.testing
def test(FLAGS):
    print("start test rnn model")
    #TODO: load pretrained model and run test


