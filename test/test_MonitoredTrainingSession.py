"""
Created by Alex Wang on 2018-04-18
training deep learning model with asynchronous distributed mode

https://www.tensorflow.org/deploy/distributed

python test_MonitoredTrainingSession.py
-files="examples.cvs"
-cluster='{\"worker\":{\"count\":3,\"gpu\":33},\"ps\":{\"count\":1}}';
"""

import tensorflow as tf


def _data_preprocess(fea):
    """
    process feature string
    :param fea:
    :return:
    """
    # TODO: decode tfrecords
    co1, co2, co3, label = tf.decode_csv(fea, record_defaults=[[1.0]] * 4, field_delim=',')
    feature = tf.stack([co1, co2, co3])
    feature = tf.reshape(feature, shape=[3])
    return feature, tf.cast(label, tf.int32)


tf.app.flags.DEFINE_string('tables', '', 'table_list')
tf.app.flags.DEFINE_string('task_index', None, 'worker task index')
tf.app.flags.DEFINE_string('ps_hosts', "", "ps hosts")
tf.app.flags.DEFINE_string('worker_hosts', "", "worker hosts")
tf.app.flags.DEFINE_string('job_name', "", "job name:worker or ps")
FLAGS = tf.app.flags.FLAGS

print('files:', FLAGS.files)
print('task_index:', FLAGS.task_index)
print('ps_hosts', FLAGS.ps_hosts)
print('worker_hosts', FLAGS.worker_hosts)
print('job_name', FLAGS.job_name)
batch_size = 10

ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
worker_count = len(worker_hosts)
task_index = int(FLAGS.task_index)

is_chief = task_index == 0  # regard worker with index 0 as chief
print('is chief:', is_chief)

server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=task_index)
if FLAGS.job_name == "ps":
    server.join()

worker_device = "/job:worker/task:%d/cpu:%d" % (task_index, 0)
print("worker device:", worker_device)

# 1. load data
with tf.device(worker_device):
    dataset = tf.data.Dataset.TFRecordDataset([FLAGS.files])

    dataset = dataset.map(_data_preprocess)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_initializable_iterator()
    next_elems = iterator.get_next()
    (feature_batch, label_batch) = next_elems
    label_batch = tf.reshape(label_batch, [batch_size])

# 2. construct network
available_worker_device = "/job:worker/task:%d" % (task_index)
with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device,cluster=cluster)):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    W = tf.get_variable(shape=(3, 2), dtype=tf.float32, initializer=tf.random_normal_initializer(), name='W_0')
    b = tf.get_variable(shape=(1, 2), dtype=tf.float32, initializer=tf.random_normal_initializer(), name='b_0')
    predict = tf.matmul(feature_batch, W) + b
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict, labels=label_batch))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    print("start training")


# 3. run session
hooks = [tf.train.StopAtStepHook(last_step=50000)]
step = 0
init_op = [iterator.initializer, tf.global_variables_initializer(),
           tf.local_variables_initializer()]
with tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief, hooks=hooks) as sess:
    sess.run(init_op)
    while not sess.should_stop():
        step += 1
        _, loss_val, global_step_val = sess.run([optimizer, loss, global_step])
        if step % 2000 == 0:
            print("step:{}, loss:{}".format(step, loss_val))
