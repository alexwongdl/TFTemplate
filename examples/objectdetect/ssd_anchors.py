"""
Created by Alex Wang on 2018-1-1
"""
import numpy as np
import collections

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def cal_anchors_areas(anchors):
    """
    计算anchors的面积
    :param anchors:
    :return:
    """
    areas = []
    for anchor in anchors:
        area = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
        areas.append(area)
    return areas


def _generate_anchors(base_size=16, scale=2):
    width, height = base_size * scale, base_size * scale
    anchors = []
    anchors.append(np.array([0, 0, width, height]))
    anchors.append(np.array([0, height / 4, width, height * 3 / 4]))
    anchors.append(np.array([width / 4, 0, width * 3 / 4, height]))
    return anchors


def generate_anchors_ssd(base_size=16, scale=2):
    """
    SSD anchors 平移调整坐标
    :param base_size:
    :param ratios:
    :param scale:
    :return:
    """
    anchors = _generate_anchors(base_size, scale)
    new_anchors = []
    # transform = base_size * scale / 2 - 8
    for anchor in anchors:
        # anchor += transform
        new_anchors.append(anchor)
    # print(new_anchors)
    areas = cal_anchors_areas(new_anchors)
    return new_anchors, areas


def cal_ssd_anchors():
    """
    ssd anchors数据准备
    :return:
    """
    anchor_nametuple = collections.namedtuple('anchor_nametuple',
                                              {'ssd_anchors', 'ssd_anchors_area', 'ssd_anchors_step',
                                               'ssd_anchors_step_num'})
    ssd_anchors = []
    ssd_anchors_area = []
    ssd_anchors_step = []
    ssd_anchors_step_num = []

    layer1, layer1_area = (generate_anchors_ssd(scale=2))  # 32 * 32      16*16
    layer2, layer2_area = (generate_anchors_ssd(scale=4))  # 64 * 64      8*8
    layer3, layer3_area = (generate_anchors_ssd(scale=8))  # 128 * 128    4*4
    layer4, layer4_area = (generate_anchors_ssd(scale=16))  # 256 * 256   2*2
    layer5, layer5_area = (generate_anchors_ssd(scale=32))  # 512 * 512   1*1

    ssd_anchors_area.append(layer1_area)
    ssd_anchors_area.append(layer2_area)
    ssd_anchors_area.append(layer3_area)
    ssd_anchors_area.append(layer4_area)
    ssd_anchors_area.append(layer5_area)

    anchor_base = 16
    ssd_anchors_step.append(anchor_base * 1)  # 32 feature_map上每移动一步对应的像素
    ssd_anchors_step.append(anchor_base * 2)  # 64
    ssd_anchors_step.append(anchor_base * 4)  # 128
    ssd_anchors_step.append(anchor_base * 8)  # 256
    ssd_anchors_step.append(anchor_base * 16)  # 512

    ssd_anchors_step_num.append(32 - 1)  # feature_map 上的移动步数
    ssd_anchors_step_num.append(16 - 1)
    ssd_anchors_step_num.append(8 - 1)
    ssd_anchors_step_num.append(4 - 1)
    ssd_anchors_step_num.append(1)

    ssd_anchors.append(layer1)
    ssd_anchors.append(layer2)
    ssd_anchors.append(layer3)
    ssd_anchors.append(layer4)
    ssd_anchors.append(layer5)

    anchor_info = anchor_nametuple(ssd_anchors=ssd_anchors, ssd_anchors_area=ssd_anchors_area,
                                   ssd_anchors_step=ssd_anchors_step, ssd_anchors_step_num=ssd_anchors_step_num)
    return ssd_anchors, ssd_anchors_area, ssd_anchors_step, ssd_anchors_step_num, anchor_info
    # return anchor_info


def test_cal_ssd_anchors():
    """
    测试ssd_anchors
    :return:
    """
    anchors, anchor_area, anchors_step, anchors_step_num, _ = cal_ssd_anchors()
    for anchors_temp in anchors:
        print(anchors_temp)
    print(anchors_step)
    print(anchor_area)

    print('-----------------------------------------')
    layer_num = 0
    layer_some = anchors[layer_num]
    print(layer_some)
    step_some = anchors_step[layer_num]
    print('step{}:{}'.format(layer_num, step_some))
    for i in range(anchors_step_num[layer_num]):
        new_layer = []
        for anchor in layer_some:
            anchor = np.array(anchor) + step_some * i
            new_layer.append(anchor)
        print(new_layer)

ssd_anchors, ssd_anchors_area, ssd_anchors_step, ssd_anchors_step_num, ssd_anchor_info = cal_ssd_anchors()

def ssd_anchors_plot():
    """
    在图像上展示SSDanchor
    :return:
    """
    layer_num = 1
    anchors, anchor_area, anchors_step, anchors_step_num, _ = cal_ssd_anchors()
    layer_some = anchors[layer_num]
    print(layer_some)
    step_some = anchors_step[layer_num]
    print('step{}:{}'.format(layer_num, step_some))
    new_layer = []
    for i in range(anchors_step_num[layer_num]):
        for anchor in layer_some:
            anchor = np.array(anchor) + step_some * i
            new_layer.append(anchor)

    print(new_layer)

    img = Image.open('E://data/test.JPG')
    img = img.resize((512,512))
    plt.imshow(img)
    currentAxis = plt.gca()
    for box in new_layer:
        rect = patches.Rectangle((box[0], box[1]), (box[2] - box[0]), (box[3] - box[1]), linewidth=3,
                                 edgecolor=colors[random.randint(0, len(colors)-1)],
                                 facecolor='none')
        currentAxis.add_patch(rect)
    plt.show()

if __name__ == '__main__':
    # test_cal_ssd_anchors()
    ssd_anchors_plot()
