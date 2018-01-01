"""
Created by Alex Wang on 2017-12-09
"""
import sys
import os
import pickle
import xml.etree.ElementTree as ET
from multiprocessing import Pool
import time
import math
import json

import numpy as np
from PIL import Image

from scipy.sparse import csr_matrix
import tensorlayer as tl
import random

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  ##当前目录的上一级
sys.path.append(root_path)

from examples.objectdetect.class_info import voc_classes_num, voc_class_to_ind
from  examples.objectdetect import generate_anchors
from examples.objectdetect import scale_convert
from examples.objectdetect import img_aug

# 生成anchors及其面积
anchors = generate_anchors.generate_anchors()
anchors_areas = generate_anchors.cal_anchors_areas(anchors)


# print('anchors:')
# print(anchors)
# print('anchors_areas:')
# print(anchors_areas)


def load_image_annotations(annotations_root_path = None, image_root_path = None, image_list = None, pickle_save_path = None):
    """
    load roi info from pickle file or xml files
    :param annotations_root_path:
    :param image_root_path:
    :param image_list:
    :param pickle_save_path:
    :return:
    """
    annotations_list = []
    if os.path.exists(pickle_save_path):
        print('load roi info from pickle file')
        annotations_list = pickle.load(open(pickle_save_path, 'rb'))
        return annotations_list

    print('load roi info from xml files.')
    for image_name in image_list:
        xml_path = os.path.join(annotations_root_path, image_name + '.xml')
        xml_feature = load_pascal_annotation(xml_path, image_name + '.jpg')
        image_path = os.path.join(image_root_path, image_name + '.jpg')
        img_feature = get_image_feature(image_path)
        xml_feature.update(img_feature)
        annotations_list.append(xml_feature)

    print('save roi info in pickle file.')
    pickle.dump(annotations_list, open(pickle_save_path, 'wb'))
    return annotations_list


def get_image_list(file_path):
    """
    get image list
    :param file_path: e.g.: E:\data\VOCdevkit\VOC2007\ImageSets\Main\trainval.txt
    :return:
    """
    image_list = []
    with open(file_path, 'r') as reader:
        line = reader.readline()
        while line:
            image_list.append(line.strip())
            line = reader.readline()

    return image_list


def get_image_feature(image_path):
    """
    get image width, height
    :param image_path:
    :return:
    """
    img = Image.open(image_path)
    (width, height) = img.size
    return {'image_width': width, 'image_height': height, 'image_path': image_path}


def load_pascal_annotation(filename, image_name):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    :param filename: xml annotation file path
    :return:
    """
    # filename = os.path.join(data_path, 'Annotations', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')

    # Exclude the samples labeled as difficult
    non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
    if len(non_diff_objs) != len(objs):
        print ('Removed {} difficult objects from {}'.format(len(objs) - len(non_diff_objs), os.path.basename(filename)))
    objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    boxes_center_rep = np.zeros((num_objs, 4), dtype=np.uint16)  # box 的中心点表示
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, voc_classes_num), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    ishards = np.zeros((num_objs), dtype=np.int32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        diffc = obj.find('difficult')
        difficult = 0 if diffc == None else int(diffc.text)
        ishards[ix] = difficult

        cls = voc_class_to_ind[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        box_width = x2 - x1
        box_height = y2 - y1
        box_cx = x1 + box_width / 2
        box_cy = y1 + box_height / 2
        boxes_center_rep[ix, :] = [box_cx, box_cy, box_width, box_height]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        seg_areas[ix] = (x2 - x1) * (y2 - y1)

    overlaps = csr_matrix(overlaps)

    # overlaps_arr = overlaps.toarray()
    # print(overlaps_arr.max(axis=1))
    # print(overlaps_arr.argmax(axis=1))

    return {'boxes': boxes,
            'boxes_center_rep': boxes_center_rep,
            'gt_classes': gt_classes,
            'gt_ishard': ishards,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas,
            'xml_path': filename,
            'image_name': image_name}


def generate_train_pathes(roi_info_list = None, pickle_save_path = None, thread_num=4):
    """
    生成训练数据，每张图片随机选取256个anchors，正anchor和负anchors的占比接近于1:1，如果图像中少于128个正anchors，就用负样本来填充。
    考察训练集中的每张图像：
    a. 对每个标定的真值候选区域，与其重叠比例最大的anchor记为前景样本
    b. 对a)剩余的anchor，如果其与某个标定重叠比例大于0.7，记为前景样本；如果其与任意一个标定的重叠比例都小于0.3，记为背景样本
    c. 对a),b)剩余的anchor，弃去不用。
    d. 跨越图像边界的anchor弃去不用
    :param roi_info_list:
    :param pickle_save_path:结果保存路径，pkl文件
    :return:
    """
    if os.path.exists(pickle_save_path):
        print('load train_data from pickle file')
        train_info = pickle.load(open(pickle_save_path, 'rb'))
        return train_info

    pool = Pool(thread_num)
    train_info = pool.map(process_one_image_roi, roi_info_list)
    pool.close()
    pool.join()

    new_train_info = []  # 过滤掉正样本数为0的图片
    count_total = 0
    count_has_positive = 0
    for train_sample in train_info:
        count_total += 1
        if len(train_sample['positive_anchors']) > 0 :
            count_has_positive += 1
            new_train_info.append(train_sample)
    print('count_total:{},count_has_positive:{},size of new_train_info:{}'.format(count_total, count_has_positive, len(new_train_info)))
    # return train_info
    pickle.dump(new_train_info, open(pickle_save_path, 'wb'))
    return new_train_info

def process_one_image_roi(annotation_feature, pool_layer_num=4, base_size = 16, max_positive_num = 128, max_sample_num = 256):
    """
    在一张图片中随机选取ROI区域
    :param annotation_feature:
    :param pool_layer_num: VGG16中输入图像和feature map的比例为16:1， 4次pool
    :param base_size: feature map 上移动一格对应的像素个数
    :param thread_num：每张图片处理进程数
    :param max_positive_num：每张图片最多的anchor正样本数
    :param max_sample_num：每张图片最多的anchor样本数
    {'image_width':width, 'image_height':height,
            'image_path':image_path
            'boxes': boxes,             array([[262, 210, 323, 338],[164, 263, 252, 371],[4, 243,  66, 373],[240, 193, 294, 298],[276, 185, 311, 219]], dtype=uint16)
                                        [left, top, right, bottom]
            'gt_classes': gt_classes,   array([9, 9, 9, 9, 9])}
            'gt_ishard': ishards,       'gt_ishard': array([0, 0, 1, 0, 1])
            'gt_overlaps': overlaps,   <kx21 sparse matrix of type '<class 'numpy.float32'>'
            'flipped': False,
            'seg_areas': seg_areas,    array([ 7998.,  9701.,  8253.,  5830.,  1260.], dtype=float32)
            'xml_path':filename,
            'image_name': image_name}
    :return:
    """
    # 随机生成
    print('generate train data...')
    # TODO:
    # pool_layer_num = 4
    # base_size = 16

    W = scale_convert.image_to_feature_map(annotation_feature['image_width'], pool_layer_num)
    H = scale_convert.image_to_feature_map(annotation_feature['image_height'], pool_layer_num)
    annotation_feature['feature_map_W'] = W
    annotation_feature['feature_map_H'] = H

    # [W_ind, H_ind, k_ind], [n, 4] box回归, [n, 2] cls分类, [n, 4] anchor值 [cx,cy,w,h], [n, 4] anchor值 [x1,y1,x2,y2], [n, cls_num] object识别
    positive_anchors = []  # 最多128
    negative_anchors = []  # 最多positive_anchors的3倍，最多128

    anchors_info_list = []
    for w in range(W):
        for h in range(H):
            delt_x = base_size * w
            delt_y = base_size * h
            delta = np.array([[delt_x, delt_y, delt_x, delt_y]] * len(anchors))
            current_anchors = np.add(anchors, delta)
            # 一个anchor最多对应一个object，为每一个anchor找到最大IoU的object，丢弃超出图像范围的anchors
            temp_feature = {}
            temp_feature.update(annotation_feature)
            temp_feature['feature_map_w'] = w
            temp_feature['feature_map_h'] = h
            temp_feature['current_anchors'] = current_anchors
            anchors_info_list.append(temp_feature)
    print('image:{}, process {} anchors_set'.format(annotation_feature['image_path'], len(anchors_info_list)))
    start_time = time.time()
    # pool = Pool(thread_num)
    # match_result = pool.map(match_anchors_objects, anchors_info_list)
    # pool.close()
    # pool.join()
    match_result = []
    for anchor_info in anchors_info_list:
        match_result.append(match_anchors_objects(anchor_info))
    end_time = time.time()
    print('image {} process done, cost time:{}s'.format(annotation_feature['image_path'], end_time - start_time))
    for matches in match_result:
        if len(matches) <= 0:
            continue
        for sub_matches in matches:
            if sub_matches['label'] == 1:
                positive_anchors.append(sub_matches)
            else:
                negative_anchors.append(sub_matches)

    # print(positive_anchors)
    ## 正负样本数量控制
    # max_positive_num = 128
    # max_sample_num = 256
    positive_num = min(len(positive_anchors), max_positive_num)
    negative_num = min(len(negative_anchors), max_sample_num - max_positive_num, positive_num * 3)

    if positive_num < len(positive_anchors):
        ind = np.random.permutation(positive_num)
        positive_anchors = [positive_anchors[i] for i in ind]

    if negative_num < len(negative_anchors):
        ind = np.random.permutation(negative_num)
        negative_anchors = [negative_anchors[i] for i in ind]

    print('image:{}\tsize of positive_anchors:{}\t size of negative_anchors:{}'.
          format(annotation_feature['image_path'], len(positive_anchors), len(negative_anchors)))
    annotation_feature['positive_anchors'] = positive_anchors
    annotation_feature['negative_anchors'] = negative_anchors
    return annotation_feature

def match_anchors_objects(anchors_objects_info):
    """
    anchors和objects之间做匹配
    需要计算的内容：[W_ind, H_ind, k_ind], [n, 4] box回归, [n, 2] cls分类, [n, 4] anchors中心点表示法 [cx,cy,w,h],
                    [n, 4] anchor值 [x1,y1,x2,y2], [n, cls_num] object识别, [n, 4] object_box, label:0/1指示是否包含boject, IoU
    :param anchors_objects_info: 包含anchors信息和objects信息
    :return:
    """
    current_anchors = anchors_objects_info['current_anchors']
    image_width = anchors_objects_info['image_width']
    image_height = anchors_objects_info['image_height']
    boxes = anchors_objects_info['boxes']
    boxes_center_rep = anchors_objects_info['boxes_center_rep']
    gt_overlaps = anchors_objects_info['gt_overlaps'].toarray()
    seg_areas = anchors_objects_info['seg_areas']  # object面积
    w_ind = anchors_objects_info['feature_map_w']
    h_ind = anchors_objects_info['feature_map_h']
    negative_label = [0] * voc_classes_num
    negative_label[0] = 1

    match_result = []
    for i in range(len(current_anchors)):
        anchor = current_anchors[i]  # [left, top, right, bottom]
        if anchor[0] < 0 or anchor[1] < 0 or anchor[2] > image_width or anchor[3] > image_height:
            continue  # 忽略超过图片边界的anchors
        # anchor 中心点表示法
        anchor_width = anchor[2] - anchor[0]
        anchor_height = anchor[3] - anchor[1]
        cx = anchor[0] + anchor_width / 2
        cy = anchor[1] + anchor_height / 2

        anchor_area = anchors_areas[i]
        max_object_ind = -1
        max_IoU = -1
        for k in range(len(boxes)):  # 遍历object
            object_box = boxes[k]
            object_area = seg_areas[k]
            IoU = scale_convert.IoU(anchor, object_box, anchor_area, object_area)
            if IoU > max_IoU:
                max_object_ind = k
                max_IoU = IoU

        if max_IoU <= 0.3:  # 负样本
            match_result.append({'fea_map_ind': [w_ind, h_ind, i], 'box_reg': [0, 0, 0, 0], 'cls': [1, 0],
                                 'anchor_center_rep': [cx, cy, anchor_width, anchor_height], 'anchor': anchor,
                                 'object_cls': negative_label, 'object_box': [0, 0, 0, 0],
                                 'label': 0, 'IoU': 0})
        elif max_IoU >= 0.7:  # 正样本
            box_center_rep = boxes_center_rep[max_object_ind]
            tx = (box_center_rep[0] - cx) / anchor_width
            ty = (box_center_rep[1] - cy) / anchor_height
            tw = math.log(box_center_rep[2] / anchor_width)
            th = math.log(box_center_rep[3] / anchor_height)
            match_result.append({'fea_map_ind': [w_ind, h_ind, i], 'box_reg': [tx, ty, tw, th], 'cls': [0, 1],
                                 'anchor_center_rep': [cx, cy, anchor_width, anchor_height], 'anchor': anchor,
                                 'object_cls': gt_overlaps[max_object_ind], 'object_box': boxes[max_object_ind],
                                 'label': 1, 'IoU': max_IoU})
    # print(match_result)
    return match_result


def run_generate_windows():
    """
    本地机器上生成训练语料
    :return:
    """
    info = load_pascal_annotation('E://data/VOCdevkit/VOC2007/Annotations/004696.xml', '004696.jpg')
    print(info)

    image_list = get_image_list('E://data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt')
    print('len of image_list:{}'.format(len(image_list)))
    print(image_list[0:10])

    roi_info = load_image_annotations('E://data/VOCdevkit/VOC2007/Annotations/',
                                      'E://data/VOCdevkit/VOC2007/JPEGImages', image_list, 'E://data/voc_roi_info.pkl')
    print('len of roi_info:{}'.format(len(roi_info)))
    print(roi_info[0:5])
    print('---------------------')
    print(roi_info[0])
    print(roi_info[1])
    # print('----------test generate train data-----------')
    # for i in range(10):
    #     annotation_feature = process_one_image_roi(roi_info[i], pool_layer_num=4)
    #     print(annotation_feature)
    # print('----------test generate train data done-----------')
    aug_roi_info = img_aug.batch_image_augment('E://data/voc_roi_info.pkl', 'E://data/aug_voc_roi_info.pkl', 'E://data/VOC_data', repeat=5, thread_num=4)

    print('prepare train data...')
    train_info = generate_train_pathes(aug_roi_info[0:60], 'E://data/voc_train_data.pkl', thread_num=4)
    print('----------------------------------------------')
    print(train_info[0])

    # count_total = 0
    # count_has_positive = 0
    # for train_sample in train_info:
    #     count_total += 1
    #     if len(train_sample['positive_anchors']) > 0 :
    #         count_has_positive += 1
    # print('count_total:{},count_has_positive:{}'.format(count_total, count_has_positive))

def run_generate_linux():
    """
    服务器上生成训练语料
    {'image_name': '000005.jpg', 'xml_path': '/data/hzwangjian1/TFTemplate/VOCdevkit/VOC2007/Annotations/000005.xml',
    'image_path': '/data/hzwangjian1/TFTemplate/VOC_data/000005_0.jpg',
    'boxes_center_norm_rep': array([[ 0.3825 ,  0.7325 ,  0.1575 ,  0.36   ],
       [ 0.6025 ,  0.85125,  0.2275 ,  0.2975 ],
       [ 0.45   ,  0.6525 ,  0.1375 ,  0.295  ]]),
        'gt_classes': array([9, 9, 9], dtype=int32), 'gt_overlaps': <3x21 sparse matrix of type '<class 'numpy.float32'>'
	with 3 stored elements in Compressed Sparse Row format>,
	'seg_areas': array([ 8928, 10710,  6608]),
	'boxes': array([[122, 221, 184, 365],
       [196, 281, 286, 400],
       [152, 202, 208, 320]]),
    'boxes_center_rep': array([[ 153. ,  293. ,   62. ,  144. ],
       [ 241. ,  340.5,   90. ,  119. ],
       [ 180. ,  261. ,   56. ,  118. ]]),
    'image_width': 400, 'image_height': 400, 'feature_map_W': 25, 'feature_map_H': 25,
    'positive_anchors': [{'fea_map_ind': [8, 17, 6], 'box_reg': [0.20114942528735633, 0.077142857142857138, -0.33877373360949214, -0.19497267434751345], 'cls': [0, 1],
    'anchor_center_rep': [135.5, 279.5, 87.0, 175.0], 'anchor': array([  92.,  192.,  179.,  367.]),
     'object_cls': array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32),
     'object_box': array([122, 221, 184, 365]), 'label': 1, 'IoU': 0.51476952022577616}],
    'negative_anchors': [{'fea_map_ind': [4, 7, 3], 'box_reg': [0, 0, 0, 0], 'cls': [1, 0], 'anchor_center_rep': [71.5, 119.5, 127.0, 127.0], 'anchor': array([   8.,   56.,  135.,  183.]),
    'object_cls': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'object_box': [0, 0, 0, 0], 'label': 0, 'IoU': 0}]}
    :return:
    """
    data_root = '/data/hzwangjian1/TFTemplate'
    VOC_root = os.path.join(data_root, 'VOCdevkit')
    info = load_pascal_annotation(os.path.join(VOC_root, 'VOC2007/Annotations/004696.xml'), '004696.jpg')
    print(info)

    image_list = get_image_list(os.path.join(VOC_root, 'VOC2007/ImageSets/Main/trainval.txt'))
    print('len of image_list:{}'.format(len(image_list)))
    print(image_list[0:10])

    roi_info = load_image_annotations(os.path.join(VOC_root, 'VOC2007/Annotations/'),
                                      os.path.join(VOC_root, 'VOC2007/JPEGImages'), image_list,
                                      os.path.join(data_root, 'voc_roi_info.pkl'))
    print('len of roi_info:{}'.format(len(roi_info)))
    print(roi_info[0:5])
    print('---------------------')
    print(roi_info[0])
    print(roi_info[1])
    # print('----------test generate train data-----------')
    # for i in range(10):
    #     annotation_feature = process_one_image_roi(roi_info[i], pool_layer_num=4)
    #     print(annotation_feature)
    # print('----------test generate train data done-----------')
    aug_roi_info = img_aug.batch_image_augment(os.path.join(data_root, 'voc_roi_info.pkl'),
                                               os.path.join(data_root, 'aug_voc_roi_info.pkl'),
                                               os.path.join(data_root, 'VOC_data'), repeat=5, thread_num=10)

    print('prepare train data...')
    train_info = generate_train_pathes(aug_roi_info,  os.path.join(data_root, 'voc_train_data.pkl'), thread_num=10)
    print('----------------------------------------------')
    print(train_info[0])

if __name__ == '__main__':
    print('img_info_load starts running...')
    run_generate_windows()
    # run_generate_linux()

    # [[1, 3, 1, 3], [1, 3, 1, 3], [1, 3, 1, 3], [1, 3, 1, 3], [1, 3, 1, 3], [1, 3, 1, 3], [1, 3, 1, 3], [1, 3, 1, 3], [1, 3, 1, 3]]
    # delta = np.array([[1, 3, 1, 3]] * len(anchors))
    # print(delta)
    # print(delta.shape)  # should be 9
    # print(np.add(delta, delta))


    # print(scale_convert.image_to_feature_map(252, 4))  # should be 16
