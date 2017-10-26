"""
Created by Alex Wang
On 2017-10-25
test cnn on mnist dataset
"""
import tensorflow as tf
import tensorlayer as tl


def cnn_model(x_hole, y_hole, reuse, is_training):
    with tf.variable_scope('cnn_model', reuse):
        tl.layers.set_name_reuse(reuse)
        net = tl.layers.InputLayer(x_hole, name='input')

        net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=[5, 5, 1, 64], name='conv_1')
        net = tl.layers.BatchNormLayer(net, is_training=is_training, act=tf.nn.relu, name='batch_norm_1')
        net = tl.layers.PoolLayer(net, name='pool_1')

        net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=[3, 3, 64, 64], name='conv_2')
        net = tl.layers.BatchNormLayer(net, is_training=is_training, act=tf.nn.relu, name='batch_norm_2')
        net = tl.layers.PoolLayer(net, name='pool_2')

        net = tl.layers.FlattenLayer(net, name='flatten')
        net = tl.layers.DenseLayer(net, n_units=384, act=tf.nn.relu, name='dense_1')
        net = tl.layers.DenseLayer(net, n_units=256, act=tf.nn.relu, name='dense_2')
        net = tl.layers.DenseLayer(net, n_units=10, act=tf.identity, name='dense_3')

        y_pred = net.outputs
        cost_entropy = tl.cost.cross_entropy(y_hole, y_pred, name='cost_entropy')
        L2 = 0
        for w in tl.layers.get_variables_with_name('model_W', True, True):
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

    # 3.build graph：including loss function，learning rate decay，optimization operation
    net, cost, _ = cnn_model(x_hole, y_hole, False, is_training=True)  # train
    _, _, acc = cnn_model(x_hole, y_hole, True, is_training=False)  # validate/test

    # 4.summary


def test_mnist(FLAGS):
    print("start_test_mnist")
