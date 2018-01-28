"""
Created by Alex Wang on 2017-12-18
测试tensorflow 函数
"""
import numpy as np
import random
import tensorflow as tf
import tensorlayer as tl
import math
import matplotlib.pyplot as plt

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


def test_numpy_random_batch():
    a = np.asarray(range(10))
    print(a)
    permutation = np.random.permutation(len(a))
    print(a[permutation[:3]])

def plot_smooth_L1():
    x_value = np.arange(0, 2, 0.1)
    x = tf.convert_to_tensor(np.arange(0, 2, 0.1))
    y1 = 0.5 * x ** 2
    y2 = tf.abs(x) - 0.5
    # y_n = tf.cond(x < 1, lambda:0.5 * x ** 2, lambda:tf.abs(x) - 0.5)
    y_n = tf.where(tf.less(x, tf.ones_like(x)), y1, y2)

    sess = tf.Session()
    y1_value, y2_value, yn_value = sess.run([y1, y2, y_n])

    plt.plot(x_value, y1_value, 'r--', x_value, y2_value, 'bs', x_value, yn_value, 'g^')
    plt.show()

def test_arr_part():
    arr = [1,2,3,4]
    arr_part = arr[0:3]
    arr_new = arr_part + [5,6]
    print(arr_new)

def test_gather():
    a = tf.convert_to_tensor([1,2,3,4,5])
    a_shape = tf.shape(a)
    # b = tf.gather(a, tf.convert_to_tensor(np.arange(0,4)))
    b = a[0:3]
    b_shape = tf.shape(b)
    c = tf.concat([b, tf.convert_to_tensor([7,8])], axis=0)
    sess = tf.Session()
    b_value, b_shape_value, a_shape_value, c_value = sess.run([b, b_shape, a_shape, c])

    print(b_value)
    print(b_shape_value)
    print(a_shape_value)
    print(c_value)

def decode_box():
    a = tf.convert_to_tensor([[1,2,3,4],[5,6,7,8]])
    b = a[0,1] * a[0,2]
    sess=tf.Session()
    b_value = sess.run([b])
    print(b_value) # 6
    c=tf.tile(a, [b,1])
    c = tf.reshape(c, shape=tf.convert_to_tensor([6, 2, -1]))
    c_value = sess.run([c])
    print(c_value)

def fetch_list():
    a = tf.convert_to_tensor([[1,2,3,4],[5,6,7,8]])
    b_list = []
    for i in range(4):
        b_list.append(tf.concat([a, tf.convert_to_tensor([[1,2,3,4]])], axis=0))
    sess = tf.Session()
    result = sess.run([b_list])
    print(result)

def get_nonzero_values():
    input = np.array([[1,0,3,5,0,8,6]])
    X = tf.placeholder(tf.int32,[None,7])
    zeros = tf.cast(tf.zeros_like(X),dtype=tf.bool)
    ones = tf.cast(tf.ones_like(X),dtype=tf.bool)
    loc = tf.where(input!=0,ones,zeros)
    result=tf.boolean_mask(input,loc)
    with tf.Session() as sess:
        out = sess.run([result],feed_dict={X:input})
        print (np.array(out))

def test_arr():
    length = 3
    fea_map_inds_batch = [[],[],[]]
    print(fea_map_inds_batch[1])
    for i in range(length):
        if i == 2:
            fea_map_inds_batch[i].extend([0,1,2,3])
    print(fea_map_inds_batch)

    negative_label = [0] * 21
    negative_label[0] = 1
    print(negative_label)


def test():
    str = '0.3042498819144103,0.20476117337838895,0.21719539894943102,0.38694691624627925,0.439820473380658,0.30495317861408927,0.2317912361449736,0.23876461141176422,0.02935335914618486,0.017870509490888097,0.020418725961011994,0.04720642058950474,0.07258404554925826,0.08330419175994523,0.03394149880271002,0.023326928480506433,0.006858298898792115,0.006678120021276475,0.009209223026271054,0.014491760233770598,0.012499995367874207,0.0070682862112403065,0.0057898072352677915,0.004394987710605141,0.1648415168062271,0.10057785094726568,0.11287694944114601,0.16464250835201985,0.19532339009329888,0.11597133704898224,0.08924400707847203,0.10333382667598622,0.10319670706319475,0.07963469291894555,0.07469050052099618,0.16060622707095665,0.15941304237026435,0.09860936359393167,0.10281592302853798,0.10770886854466462'
    sub_strs = str.split(",")
    sum_a = 0
    for s in sub_strs:
        sum_a += float(s) * float(s)
    print(len(sub_strs))
    print(sum_a)

def test_gather_nd():
    a = np.random.random((16,8,8,3,21))
    print(a.shape)
    index = [[7, 3, 3, 1],[10, 3, 3, 2]]
    # index = [[0, 0], [1, 1]]
    a_p = tf.placeholder(dtype=tf.float32, shape=[16,8,8,3,21], name='a_p')
    index_p = tf.placeholder(dtype=tf.int32, shape=[None, 4], name='index_p')

    gathered_data = tf.gather_nd(params=a_p, indices=index_p)
    sess = tf.Session()
    result = sess.run([gathered_data], feed_dict={a_p:a, index_p:index})
    print(result)

if __name__ == '__main__':
    print("test tensorflow functions...")
    # test_tensor_index()
    # test_tensor_reshape()
    # test_hstack()
    # test_pool()
    # test_pool_np()
    # arr_inverse()
    # test_permutation()
    # test_list_concat()
    # test_argmax()
    # test_numpy_random_batch()
    # plot_smooth_L1()
    # test_arr_part()
    # test_gather()
    # decode_box()
    # fetch_list()
    # get_nonzero_values()
    # test_arr()
    test()
    test_gather_nd()