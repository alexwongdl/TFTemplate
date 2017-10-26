"""
Created by Alex Wang
On 2017-10-25
test cnn on mnist dataset
"""
import tensorflow as tf
import tensorlayer as tl
import math


def cnn_model(x_hole, y_hole, reuse, is_training):
    with tf.variable_scope('cnn_model', reuse):
        tl.layers.set_name_reuse(reuse)
        net = tl.layers.InputLayer(x_hole, name='input')

        net = tl.layers.Conv2dLayer(net, act=None, shape=[5, 5, 1, 64], name='conv_1')
        net = tl.layers.BatchNormLayer(net, is_training=is_training, act=tf.nn.relu, name='batch_norm_1')
        net = tl.layers.PoolLayer(net, name='pool_1')

        net = tl.layers.Conv2dLayer(net, act=None, shape=[3, 3, 64, 64], name='conv_2')
        net = tl.layers.BatchNormLayer(net, is_training=is_training, act=tf.nn.relu, name='batch_norm_2')
        net = tl.layers.PoolLayer(net, name='pool_2')

        net = tl.layers.FlattenLayer(net, name='flatten')
        net = tl.layers.DenseLayer(net, n_units=384, act=tf.nn.relu, name='dense_1')
        net = tl.layers.DenseLayer(net, n_units=256, act=tf.nn.relu, name='dense_2')
        net = tl.layers.DenseLayer(net, n_units=10, act=tf.identity, name='dense_3')

        y_pred = net.outputs
        cost_entropy = tl.cost.cross_entropy(y_hole, y_pred, name='cost_entropy')
        L2 = 0
        for w in tl.layers.get_variables_with_name('cnn_model', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(w)
        cost = cost_entropy + L2

        correct_prediction = tf.equal(tf.argmax(y_pred, 1), y_hole)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, acc

def train_mnist(FLAGS):
    print("start_train_mnist")
    # 2.load data
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28),
                                                                                 path=FLAGS.input_dir)
    x_hole = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='x_hole')
    y_hole = tf.placeholder(dtype=tf.int8, shape=[None, 28, 28], name='y_hole')
    x_batch, y_batch = tf.train.shuffle_batch([X_train, y_train],FLAGS.batch_size, 100, 10, 12)

    # 3.build graph：including loss function，learning rate decay，optimization operation
    net, cost, _ = cnn_model(x_hole, y_hole, False, is_training=True)  # train
    _, _, acc = cnn_model(x_hole, y_hole, True, is_training=False)  # validate/test

    steps_per_epoch = int(math.ceil(X_train.shape[0]/FLAGS.batch_size))
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate, staircase=FLAGS.stair)

    train_params = net.all_params
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    # 4.summary
    input_images = tf.image.convert_image_dtype(x_hole, dtype=tf.uint8, saturate=True)
    with tf.name_scope("input_summary"):
        tf.summary.image("input_summary", input_images)
    tf.summary.scalar('cost', cost)
    tf.summary.scalar('learning_rate', learning_rate)

    # 5.training、valid、save check point in loops
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sv = tf.train.Supervisor(logdir=FLAGS.summary_dir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config) as sess:
        print('start optimization...')


        print('optimization finished!')


def test_mnist(FLAGS):
    print("start_test_mnist")
