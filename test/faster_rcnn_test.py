"""
Created by Alex Wang on 2017-12-18
测试tensorflow 函数
"""
import numpy as np
import random
import tensorflow as tf
import tensorlayer as tl


def test_tensor_index():
    """
    tensor索引
    :return:
    """
    feature_map = np.array([[[[110], [111]], [[120], [121]], [[130], [131]]],
                            [[[210], [211]], [[220], [221]], [[230], [231]]],
                            [[[310], [311]], [[320], [321]], [[330], [331]]]])
    print(feature_map.shape)

    inds = [[0, 2], [1, 1]]
    result = tf.gather_nd(feature_map, inds)
    sess = tf.Session()
    result_val = np.array(sess.run([result]))
    print(result_val[0])
    print(result_val[0].shape)


def test_tensor_reshape():
    """
    测试高维tensor reshape
    :return:
    """
    feature_map = np.array([[[[1100, 1101, 1102, 1103], [1110, 1111, 1112, 1113]],
                             [[1200, 1201, 1202, 1203], [1210, 1211, 1212, 1213]],
                             [[1300, 1301, 1302, 1303], [1310, 1311, 1312, 1313]]],

                            [[[2100, 2101, 2102, 2103], [2110, 2111, 2112, 2113]],
                             [[2200, 2201, 2202, 2203], [2210, 2211, 2212, 2213]],
                             [[2300, 2301, 2302, 2303], [2310, 2311, 2312, 2313]]],

                            [[[3100, 3101, 3102, 3103], [3110, 3111, 3112, 3113]],
                             [[3200, 3201, 3202, 3203], [3210, 3211, 3212, 3213]],
                             [[3300, 3301, 3302, 3303], [3310, 3311, 3312, 3313]]]])
    # feature_map = np.array([[[[1100, 1101, 1102, 1103], [1110, 1111, 1112, 1113]],
    #                          [[1210, 1211, 1212, 1213]],
    #                          [[1300, 1301, 1302, 1303], [1310, 1311, 1312, 1313]]],
    #
    #                         [[[2100, 2101, 2102, 2103], [2110, 2111, 2112, 2113]],
    #                          [[2200, 2201, 2202, 2203], [2210, 2211, 2212, 2213]],
    #                          [[2300, 2301, 2302, 2303]]],
    #
    #                         [[[3100, 3101, 3102, 3103], [3110, 3111, 3112, 3113]],
    #                          [[3200, 3201, 3202, 3203], [3210, 3211, 3212, 3213]],
    #                          [[3300, 3301, 3302, 3303], [3310, 3311, 3312, 3313]]]])
    # print(feature_map.shape)
    # print(feature_map[0])
    # print(feature_map[0].shape)
    # print(feature_map[1])
    # print(feature_map[1].shape)
    # print(feature_map[2])
    # print(feature_map[2].shape)
    shape_one = tf.shape(feature_map)
    shape_org = shape_one
    # 调整tensor最后两维,tensorflow不支持replace
    shape_one = tf.expand_dims(shape_one, 0)
    shape_one = tf.slice(shape_one, [0, 0], [1, 3])
    shape_one = tf.concat([shape_one, tf.convert_to_tensor([[2, 2]])], 1)
    shape_one = tf.squeeze(shape_one)


    result = tf.reshape(feature_map, [3, 3, 2, 2, 2])
    sess = tf.Session()
    result_val, shape_value, shape_org_value = sess.run([result, shape_one, shape_org])
    result_val = np.array(result_val)
    print(result_val[0])
    print(result_val[0].shape)
    print(shape_value)
    print(shape_org_value)



def test_hstack():
    """
    在每一行之前加一维
    :return:
    """
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[0]] * len(a))
    # b = np.array([[0],[0],[0]])
    result = np.hstack((b, a))
    print(result)


def test_pool():
    image = np.random.normal(0, 1.0, (1, 254, 256, 3))
    print(image.shape)
    input_net = tl.layers.InputLayer(inputs=tf.cast(tf.convert_to_tensor(image), dtype=tf.float32), name='input')
    # network = tl.layers.Conv2d(input_net, n_filter=64, filter_size=(3, 3),strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
    network = tl.layers.MaxPool2d(net=input_net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
    pool1 = network
    network = tl.layers.MaxPool2d(net=network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
    pool2 = network
    network = tl.layers.MaxPool2d(net=network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
    pool3 = network
    network = tl.layers.MaxPool2d(net=network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')
    pool4 = network

    sess = tf.Session()
    outputs = sess.run([pool1.outputs, pool2.outputs, pool3.outputs, pool4.outputs])
    for item in outputs:
        item_np = np.array(item)
        print(item_np.shape)


def test_pool_np():
    """
    In tf.nn.max_pool
    If padding = "SAME": output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
    If padding = "VALID": output_spatial_shape[i] =
                                ceil((input_spatial_shape[i] - (window_shape[i] - 1) * dilation_rate[i]) / strides[i]).
    :return:
    """
    width = 252
    for i in range(4):
        width = np.ceil(width / 2)
    print(width)


def arr_inverse():
    a = [1, 2, 3, 4]
    b = a[::-1]
    print(b)


def test_permutation():
    ind = np.random.permutation(10)
    a = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70], [8, 80], [9, 90], [10, 100]]
    print([a[i] for i in ind])

def test_list_concat():
    index = 0
    print([0] + [1,2,3])

def test_argmax():
    a = tf.convert_to_tensor([[0,1],[1,0],[0,1],[1,0]])
    b = tf.argmax(a, axis=1)
    sess = tf.Session()
    b_value = sess.run([b])
    print(b_value)


if __name__ == '__main__':
    print("test tensorflow functions...")
    # test_tensor_index()
    test_tensor_reshape()
    # test_hstack()
    # test_pool()
    # test_pool_np()
    # arr_inverse()
    # test_permutation()
    test_list_concat()
    test_argmax()