"""
Created by Alex Wang on 2018-03-24
"""

import tensorflow as tf
from tensorflow.python import debug as tf_debug

a = tf.random_normal(shape=(2,3))
b = tf.random_normal(shape=(3,4))
c = tf.matmul(a, b)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    c_value = sess.run([c])
    print(c_value)

