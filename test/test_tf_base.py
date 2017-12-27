"""
Created by Alex Wang on 2017-06-23
"""
import tensorflow as tf

def add(var_one):
    var_one = tf.add(var_one, var_one)
    return var_one

def test_reuse_variables():
    sess = tf.Session()

    summary_writer = tf.summary.FileWriter('E://temp/tensorflow/log')
    var_one = tf.constant(1)
    var_summary = tf.summary.scalar(var_one.op.name ,tensor=var_one)
    summary_op = tf.summary.merge_all()

    for i in range(10):
        var_one = add(var_one)
        summary_str = sess .run( summary_op)
        summary_writer.add_summary (summary_str, i)
        var_value = sess.run([var_one])
        print(var_value)
    sess.close()
    summary_writer.close()

def test_learning_rate():
    decay_step =1000
    for i in range(1000000):
        decayed_learning_rate = 1.0 * 0.995 ** (i / decay_step)
        if i % 1000 == 0:
            print(decayed_learning_rate)

if __name__ == "__main__":
    # test_reuse_variables()
    test_learning_rate()