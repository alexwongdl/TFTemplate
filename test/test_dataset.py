"""
Created by Alex Wang on 2018-03-19
tf.data.Dataset
"""
import numpy as np
import tensorflow as tf
import cv2

def dataset_one_shot():
    """
    Dataset一次遍历，不需要显式初始化，不支持参数化
    :return:
    """
    array = np.array(range(10))
    dataset = tf.data.Dataset.from_tensor_slices(array)
    iterator = dataset.make_one_shot_iterator()
    next_one = iterator.get_next()

    sess = tf.Session()
    for i in range(10):
        next_value = sess.run([next_one])
        print(next_value)

def dataset_initialized():
    """
    初始化的Dataset，支持placeholder参数
    :return:
    """
    array = np.array(range(10))
    array_two = np.array(range(10, 20))
    param = tf.placeholder(tf.int32, shape=[len(array)], name='param')
    dataset = tf.data.Dataset.from_tensor_slices(param)
    iterator = dataset.make_initializable_iterator()
    next_one = iterator.get_next()

    sess = tf.Session()
    sess.run(iterator.initializer, feed_dict={param: array})
    for i in range(10):
        next_value = sess.run([next_one])
        print(next_value)

    sess.run(iterator.initializer, feed_dict={param: array_two})
    for i in range(10):
        next_value = sess.run([next_one])
        print(next_value)

def _parse_tfrecords_func(record):
    """
    解析tfrecords数据
    :param record:
    :return:
    """
    features = {"img": tf.FixedLenFeature((),tf.string, default_value = ''),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0),
                "width": tf.FixedLenFeature((), tf.int64, default_value=0),
                "height": tf.FixedLenFeature((), tf.int64, default_value=0),
                "channel": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(record, features)
    for key in parsed_features:
        print(key, type(parsed_features[key]))

    print(type(parsed_features['img']))
    img = tf.decode_raw(parsed_features['img'], tf.uint8)
    img_reshape = tf.reshape(img, (tf.stack([parsed_features['width'], parsed_features['height'], parsed_features['channel']])))
    return img, parsed_features['width'], parsed_features['height'], parsed_features['channel'], img_reshape

def dataset_tfrecords():
    """
    Dataset读tfrecords
    :return:
    """
    tfrecords_files = ['tfrecords_example']
    dataset = tf.data.TFRecordDataset(tfrecords_files)
    dataset = dataset.map(_parse_tfrecords_func)
    dataset.repeat()

    iterator = dataset.make_initializable_iterator()
    next_elem = iterator.get_next()
    sess = tf.Session()
    sess.run(iterator.initializer)

    for i in range(1):
        next_elem_value = sess.run(next_elem)
        print(type(next_elem_value))
        img, img_width, img_height, img_channel, img_reshape = next_elem_value
        cv2.imshow('img_reshape', img_reshape)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # dataset_one_shot()
    # dataset_initialized()
    dataset_tfrecords()