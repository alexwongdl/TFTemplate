"""
Created by Alex Wang on 2017-06-22
数据预取

example 1:
    train_input_queue = tf.train.slice_input_producer( [X_train, y_train], shuffle=True, capacity=10 * batch_size)
    x_batch, y_batch = tf.train.shuffle_batch([train_input_queue[0], train_input_queue[1]], batch_size,  200, 20, num_threads=20)
    coord, threads = data_batch_fetch.start_queue_runner(sess)
    for i in range(10):
        x_value, y_value = sess.run([x_batch, y_batch])
        print(x_value.shape)
        print(y_value.shape)
    data_batch_fetch.stop_queue_runner(coord, threads)

example 2:
    train_input_queue = tf.train.slice_input_producer( [img_one_train, img_two_train, label_train], shuffle=True, capacity=10 * batch_size)
    img_one_queue = get_image(train_input_queue[0])
    img_two_queue = get_image(train_input_queue[1])
    label_queue = train_input_queue[2]

    batch_img_one, batch_img_two, batch_label = tf.train.shuffle_batch([img_one_queue, img_two_queue, label_queue],\
                                                                       batch_size=batch_size,capacity =  10 + 10* batch_size,\
                                                                       min_after_dequeue = 10,num_threads=16,\
                                                                      shapes=[(image_width, image_height, image_channel),\
                                                                              (image_width, image_height, image_channel),()])
"""
import tensorflow as tf

def start_queue_runner(sess):
    """
    启动queue runner，同步多个线程的启动和结束，使用多线程的时候需要用到Coordinator这个类
    :param sess:
    :return: coord, threads
    """
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    return coord, threads

def stop_queue_runner(coord, threads):
    """
    停止queue runner
    :param coord:
    :param threads:
    :return:
    """
    coord.request_stop()
    coord.join(threads)