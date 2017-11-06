"""
Created by Alex Wang on 20170704
"""
import tensorflow as tf
import sys
import collections
import os
from  tfutil import data_batch_fetch
import numpy as np


def _read_words(filename):
    """
    :param filename:
    :return: 所有句子连成一个字符串，切分单词构成一个列表
    """
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    """
    :param filename:
    :return:
        word_to_id--单词到id的映射
        id_to_word--id到单词的映射
    """
    str_list = _read_words(filename)
    counter_one = collections.Counter(str_list)
    count_pairs = sorted(counter_one.items(), key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))
    # print(len(word_to_id))  # 10000
    return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
    """
    :param filename:
    :param word_to_id:
    :return:
        word_ids：单词对应的id列表
    """
    str_list = _read_words(filename)
    word_ids = [word_to_id[word] for word in str_list if word in word_to_id]
    return word_ids


def ptb_raw_data(data_path=None):
    """
    :param data_path:
    :return:
    """
    train_path = os.path.join(data_path, "ptb.train.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")

    word_to_id, id_to_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    return train_data, test_data, valid_data, word_to_id, id_to_word


def ptb_data_batch(raw_data, batch_size, num_steps):
    """
    构造[batch, num_steps]格式数据
    :param raw_data:
    :param batch_size:
    :param num_steps:一个句子中的单词个数
    :return:
    """
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.reshape(raw_data[0:batch_size * batch_len], [batch_size, batch_len])
    epoch_size = (batch_len - 1) // num_steps
    x_ = []
    y_ = []
    for i in range(epoch_size):
        x_.append(data[:, i * num_steps: (i + 1) * num_steps])
        y_.append(data[:, i * num_steps + 1: (i + 1) * num_steps + 1])
    return x_, y_, epoch_size


def ptb_data_queue(raw_data, batch_size, num_steps):
    """
    range_input_producer构造数据输入queu
    :param raw_data:
    :param batch_size:
    :param num_steps:一个句子中的单词个数
    :return:
    """
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size  ## 每一个batch包含的单词个数
    data = tf.reshape(raw_data[0: batch_size * batch_len], [batch_size, batch_len])
    epoch_size = (batch_len - 1) // num_steps

    index = tf.train.range_input_producer(epoch_size, shuffle=False,
                                          num_epochs=None).dequeue()  ##如果没有dequeue，TypeError: unsupported operand type(s) for *: 'FIFOQueue' and 'int'
    x = tf.strided_slice(data, [0, index * num_steps], [batch_len, (index + 1) * num_steps])
    y = tf.strided_slice(data, [0, index * num_steps + 1], [batch_len, (index + 1) * num_steps + 1])

    return x, y, epoch_size


def test_ptb_data_queue():
    data_path = "../data/ptb"
    data_path = os.path.abspath(data_path)
    # data_path = "/home/recsys/hzwangjian1/learntf/ptb_data"
    train_data, test_data, valid_data, word_to_id, id_to_word = ptb_raw_data(data_path)
    print(len(train_data))
    sess = tf.Session()
    x, y, _ = ptb_data_queue(train_data, batch_size=20, num_steps=10)
    coord, threads = data_batch_fetch.start_queue_runner(sess)
    for i in range(10):
        print("round :" + str(i))
        x_value, y_value = sess.run([x, y])
        print(len(x_value))
        # if i % 1000 == 0:
        for i in range(len(x_value)):
            x_words = [id_to_word[id] for id in x_value[i] if id in id_to_word]
            print("x_words:" + " ".join(x_words))
            y_words = [id_to_word[id] for id in y_value[i] if id in id_to_word]
            print("y_words:" + " ".join(y_words))
    data_batch_fetch.stop_queue_runner(coord, threads)
    sess.close()


def test_ptb_test_data():
    data_path = "../data/ptb"
    data_path = os.path.abspath(data_path)
    # data_path = "/home/recsys/hzwangjian1/learntf/ptb_data"
    train_data, test_data, valid_data, word_to_id, id_to_word = ptb_raw_data(data_path)
    print(len(train_data))

    x, y, epoch_size = ptb_data_batch(test_data, 20, 10)

    for i in range(epoch_size):
        x_batch = x[i]
        y_batch = y[i]
        for j in range(len(x_batch)):
            x_words = [id_to_word[id] for id in x_batch[j] if id in id_to_word]
            print("x_words:" + " ".join(x_words))
            y_words = [id_to_word[id] for id in y_batch[j] if id in id_to_word]
            print("y_words:" + " ".join(y_words))
        print()

if __name__ == "__main__":
    # str_list =_read_words("E://data/ptb/data/ptb.test.txt")
    # for str in str_list:
    #     print(str)

    # test_ptb_data_queue()
    test_ptb_test_data()
