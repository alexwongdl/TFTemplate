# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Non-Maximum Suppression就是需要根据score矩阵和region的坐标信息，从中找到置信度比较高的bounding box。
首先，NMS计算出每一个bounding box的面积，然后根据score进行排序，把score最大的bounding box作为队列中。
接下来，计算其余bounding box与当前最大score与box的IoU，去除IoU大于设定的阈值的bounding box。
然后重复上面的过程，直至候选bounding box为空。
最终，检测了bounding box的过程中有两个阈值，一个就是IoU，另一个是在过程之后，从候选的bounding box中剔除score小于阈值的bounding box。
"""

import numpy as np

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]  # (x1, y1) ⇒ bounding boxes 左上点坐标的集合
    x2 = dets[:, 2]
    y2 = dets[:, 3]  # (x2, y2) ⇒ bounding boxes 右下点坐标的集合
    scores = dets[:, 4]  # scores 向量

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 各个 bounding boxes 的面积
    order = scores.argsort()[::-1]         # 默认从小到大排序 [::-1]向量翻转

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)   # 选择score最大bounding box加入到候选队列
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # 计算IoU

        inds = np.where(ovr <= thresh)[0]  # 找出IoU小于overlap阈值的index
        order = order[inds + 1]  # 因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位

    return keep
