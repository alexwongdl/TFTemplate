"""
Created by Alex Wang on 2017-12-22
尺度变换：
    图像与feature map的对应关系  image_to_feature_map
"""

import numpy as np


def image_to_feature_map(length, pool_layer_num):
    """
    计算经过k层pool layer之后图片长度的变化
    :param length: width or height of image
    :param pool_layer_num: pool layer 层数，e.g.4
    :return:
    """
    for i in range(pool_layer_num):
        length = np.ceil(length / 2)
    return length


def IoU(a, b, area_a, area_b):
    """
    计算 IoU, intersect of union
    left top cornor is (0,0)
    :param a:[left, top, right, bottom]
    :param b:[left, top, right, bottom]
    :return:
    """
    intersect_left = max(a[0], b[0])
    intersect_top = max(a[1], b[1])
    intersect_right = min(a[2], b[2])
    intersect_bottom = min(a[3], b[3])

    intersect_area = (intersect_right - intersect_left) * (intersect_bottom - intersect_top)
    return intersect_area / (area_a + area_b - intersect_area)


if __name__ == '__main__':
    print(IoU([1, 1, 10, 9], [5, 6, 15, 14], 72, 80))  # 0.10948905109489052
