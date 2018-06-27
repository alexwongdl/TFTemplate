"""
Created by Alex Wang on 2018-03-19
"""
import cv2
import numpy as np
import tensorflow as tf


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def image_to_bytes():
    """
    图像转化成bytes并还原
    :return:
    """
    img = cv2.imread('imgs/running_man.jpg')
    img_shape = img.shape
    img_string = img.tostring()
    img_arr = np.fromstring(img_string, np.uint8)
    img_reconstruct = np.reshape(img_arr, img_shape)

    cv2.imshow('img', img_reconstruct)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def tfrecords_save():
    """
    测试写tfrecords文件
    :return:
    """
    tfrecords_path = 'tfrecords_example'
    tfwriter = tf.python_io.TFRecordWriter(tfrecords_path)
    img = cv2.imread('imgs/running_man.jpg')
    label = 1
    img_shape = img.shape

    img_string = img.tostring()

    example = tf.train.Example(features = tf.train.Features(feature={
        'img':bytes_feature(img_string),
        'label':int64_feature(label),
        'width':int64_feature(img_shape[0]),
        'height':int64_feature(img_shape[1]),
        'channel':int64_feature(img_shape[2])
    }))
    tfwriter.write(example.SerializeToString())
    tfwriter.close()

def tfrecords_load():
    """
    测试读tfrecords文件
    :return:
    """
    tfrecords_path = 'tfrecords_example'
    tfreader = tf.python_io.tf_record_iterator(tfrecords_path)
    i = 0
    for string_record in tfreader:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        img = np.fromstring(example.features.feature['img'].bytes_list.value[0], np.uint8)
        width = int(example.features.feature['width'].int64_list.value[0])
        height = int(example.features.feature['height'].int64_list.value[0])
        channel = int(example.features.feature['channel'].int64_list.value[0])
        img_reshape = np.reshape(img, (width, height, channel))
        label = int(example.features.feature['label'].int64_list.value[0])
        print(len(img))
        print('label:', label)
        cv2.imshow('img' + str(i), img_reshape)
        i += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # image_to_bytes()

    tfrecords_save()
    tfrecords_load()


