"""
Created by Alex Wang on 2018-1-1
SSD数据准备
"""
import math
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  ##当前目录的上一级
sys.path.append(root_path)

from examples.objectdetect.ssd_anchors import ssd_anchors, ssd_anchors_area, ssd_anchors_step_size, ssd_anchors_step_num
from examples.objectdetect import detect_data_prepare
from examples.objectdetect import img_aug
from examples.objectdetect.class_info import voc_classes_num
from examples.objectdetect import scale_convert


def match_anchors_objects(annotation_feature, anchors_info_list):
    """
    anchors和objects之间做匹配
    需要计算的内容：[W_ind, H_ind, k_ind], [n, 4] box回归, [n, 2] cls分类, [n, 4] anchors中心点表示法 [cx,cy,w,h],
                    [n, 4] anchor值 [x1,y1,x2,y2], [n, cls_num] object识别, [n, 4] object_box, label:0/1指示是否包含boject, IoU
    :param annotation_feature:
    {'image_width':width, 'image_height':height,
            'image_path':image_path
            'boxes': boxes,             array([[262, 210, 323, 338],[164, 263, 252, 371],[4, 243,  66, 373],[240, 193, 294, 298],[276, 185, 311, 219]], dtype=uint16)
                                        [left, top, right, bottom]
            'boxes_center_rep'          array([[ 345.5,  294. ,  103. ,  166. ],[ 101. ,  191. ,   26. ,   74. ],[ 173.5,  280. ,  347. ,  194. ]])
            'boxes_center_norm_rep'     tensorlayer图像增强表示方法 array([[ 0.8625 ,  0.735  ,  0.2575 ,  0.415  ],[ 0.2525 ,  0.4775 ,  0.065  ,  0.1875 ],[ 0.43375,  0.7    ,  0.8675 ,  0.485  ]])
            'gt_classes': gt_classes,   array([9, 9, 9, 9, 9])}
            'gt_ishard': ishards,       'gt_ishard': array([0, 0, 1, 0, 1])
            'gt_overlaps': overlaps,   <kx21 sparse matrix of type '<class 'numpy.float32'>'
            'seg_areas': seg_areas,    array([ 7998.,  9701.,  8253.,  5830.,  1260.], dtype=float32)
            'xml_path':filename,
            'image_name': image_name}
    :param anchors_info_list:feature_map_w, feature_map_h, current_anchors, anchors_area
    :return:
    """
    boxes = annotation_feature['boxes']
    boxes_center_rep = annotation_feature['boxes_center_rep']
    gt_overlaps = annotation_feature['gt_overlaps'].toarray()
    seg_areas = annotation_feature['seg_areas']  # object面积

    current_anchors = anchors_info_list['current_anchors']
    anchors_areas = anchors_info_list['anchors_area']
    w_ind = anchors_info_list['feature_map_w']
    h_ind = anchors_info_list['feature_map_h']
    negative_label = [0] * voc_classes_num
    negative_label[0] = 1

    match_result = []
    for i in range(len(current_anchors)):
        anchor = current_anchors[i]  # [left, top, right, bottom]
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

        if max_IoU <= 0.1:  # 负样本
            match_result.append({'fea_map_ind': [w_ind, h_ind, i], 'box_reg': [0, 0, 0, 0], 'cls': [1, 0],
                                 'anchor_center_rep': [cx, cy, anchor_width, anchor_height], 'anchor': anchor,
                                 'object_cls': negative_label, 'object_box': [0, 0, 0, 0],
                                 'label': 0, 'IoU': 0})
        elif max_IoU >= 0.5:  # 正样本
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


negative_sample_max_num = [50, 20, 10,10]
def process_one_image_roi(annotation_feature):
    """
    在一张图片中随机选取ROI区域
    :param annotation_feature:
    {'image_width':width, 'image_height':height,
            'image_path':image_path
            'boxes': boxes,             array([[262, 210, 323, 338],[164, 263, 252, 371],[4, 243,  66, 373],[240, 193, 294, 298],[276, 185, 311, 219]], dtype=uint16)
                                        [left, top, right, bottom]
            'boxes_center_rep'          array([[ 345.5,  294. ,  103. ,  166. ],[ 101. ,  191. ,   26. ,   74. ],[ 173.5,  280. ,  347. ,  194. ]])
            'boxes_center_norm_rep'     tensorlayer图像增强表示方法 array([[ 0.8625 ,  0.735  ,  0.2575 ,  0.415  ],[ 0.2525 ,  0.4775 ,  0.065  ,  0.1875 ],[ 0.43375,  0.7    ,  0.8675 ,  0.485  ]])
            'gt_classes': gt_classes,   array([9, 9, 9, 9, 9])}
            'gt_ishard': ishards,       'gt_ishard': array([0, 0, 1, 0, 1])
            'gt_overlaps': overlaps,   <kx21 sparse matrix of type '<class 'numpy.float32'>'
            'seg_areas': seg_areas,    array([ 7998.,  9701.,  8253.,  5830.,  1260.], dtype=float32)
            'xml_path':filename,
            'image_name': image_name}
    :return:
    """
    # 随机生成
    print('generate train data...')
    start_time = time.time()
    # TODO:
    positive_anchors = []  # append每一层的正负样本
    negative_anchors = []
    for anchor_layer_index in range(len(ssd_anchors)):  # 获取每一层的anchors
        anchors_info_list = []
        # 为每一个anchor找最大IoU的object
        anchors_temp = ssd_anchors[anchor_layer_index]
        anchor_area_temp = ssd_anchors_area[anchor_layer_index]
        anchor_step_size_temp = ssd_anchors_step_size[anchor_layer_index]
        anchor_step_num_temp = ssd_anchors_step_num[anchor_layer_index]

        # 计算这一层上的所有anchors
        for w in range(anchor_step_num_temp):
            for h in range(anchor_step_num_temp):
                delt_x = anchor_step_size_temp * w
                delt_y = anchor_step_size_temp * h
                delta = np.array([[delt_x, delt_y, delt_x, delt_y]] * len(anchors_temp))

                temp_feature = {}
                temp_feature.update(annotation_feature)
                temp_feature['feature_map_w'] = w
                temp_feature['feature_map_h'] = h
                temp_feature['current_anchors'] = np.add(anchors_temp, delta)
                temp_feature['anchors_area'] = anchor_area_temp
                anchors_info_list.append(temp_feature)

        # 对每一个anchor计算和它最大IOU的object
        positive_anchors_layer = []
        negative_anchors_layer = []
        for anchor_info in anchors_info_list:  # 这一层上feature_map对应的每一个[x,h]上的anchors进行匹配
            match_result = match_anchors_objects(annotation_feature, anchor_info)  # list<dict>
            for match in match_result:
                if match['label'] == 1:
                    positive_anchors_layer.append(match)
                else:
                    negative_anchors_layer.append(match)
        positive_num = len(positive_anchors_layer)
        negative_num = len(negative_anchors_layer)
        try:
            if negative_num > positive_num * 3:
                negative_num_new = min(max(positive_num * 3, 10), negative_num)
                # negative_num_new = min(max(positive_num * 3, negative_sample_max_num[anchor_layer_index]), negative_num)
                ind = np.random.permutation(negative_num)
                negative_anchors_layer = [negative_anchors_layer[i] for i in ind[0:negative_num_new]]
        except Exception as e:
            print('debug :positive_num:{}\t negative_num:{}'.format(positive_num, negative_num))

        print('image:{}\tlayer:{}\tsize of positive_anchors:{}\t size of negative_anchors:{}'.
              format(annotation_feature['image_path'], anchor_layer_index, len(positive_anchors_layer),
                     len(negative_anchors_layer)))

        positive_anchors.append(positive_anchors_layer)
        negative_anchors.append(negative_anchors_layer)


    annotation_feature['positive_anchors'] = positive_anchors
    annotation_feature['negative_anchors'] = negative_anchors

    end_time = time.time()
    print('image {} process done, cost time:{}s'.format(annotation_feature['image_path'], end_time - start_time))
    return annotation_feature


def generate_train_pathes(roi_info_list=None, pickle_save_path=None, thread_num=4):
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
        has_positive_ = 0
        for positive_anchors_layer in train_sample['positive_anchors']:
            if len(positive_anchors_layer) > 0:
                has_positive_ = 1

        if has_positive_:
            count_has_positive += 1
            new_train_info.append(train_sample)
    print('count_total:{},count_has_positive:{},size of new_train_info:{}'.format(count_total, count_has_positive,
                                                                                  len(new_train_info)))
    # return train_info
    pickle.dump(new_train_info, open(pickle_save_path, 'wb'))
    return new_train_info


def ssd_data_prepare_windows():
    """
    在windows上准备训练数据
    :return:
    """
    image_list = detect_data_prepare.get_image_list('E://data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt')
    print('len of image_list:{}'.format(len(image_list)))
    print(image_list[0:3])

    roi_info = detect_data_prepare.load_image_annotations('E://data/VOCdevkit/VOC2007/Annotations/',
                                                          'E://data/VOCdevkit/VOC2007/JPEGImages', image_list,
                                                          'E://data/voc_roi_info.pkl')
    print('len of roi_info:{}'.format(len(roi_info)))
    aug_roi_info = img_aug.batch_image_augment('E://data/voc_roi_info.pkl', 'E://data/aug_voc_roi_info.pkl',
                                               'E://data/VOC_data', repeat=5, thread_num=4, img_size=[512, 512])
    train_info = generate_train_pathes(aug_roi_info[0:100], 'E://data/voc_train_data.pkl', thread_num=4)
    print(train_info[0])

def ssd_data_prepare_linux():
    """
    在linux上准备训练数据
    :return:
    """
    data_root = '/data/hzwangjian1/TFTemplate'
    VOC_root = os.path.join(data_root, 'VOCdevkit')

    image_list = detect_data_prepare.get_image_list(os.path.join(VOC_root, 'VOC2007/ImageSets/Main/trainval.txt'))
    print('len of image_list:{}'.format(len(image_list)))
    print(image_list[0:10])

    roi_info = detect_data_prepare.load_image_annotations(os.path.join(VOC_root, 'VOC2007/Annotations/'),
                                                          os.path.join(VOC_root, 'VOC2007/JPEGImages'), image_list,
                                                          os.path.join(data_root, 'voc_roi_info.pkl'))
    print('len of roi_info:{}'.format(len(roi_info)))
    print(roi_info[0:3])
    print('load roi info done---------------------------------------')
    aug_roi_info = img_aug.batch_image_augment(os.path.join(data_root, 'voc_roi_info.pkl'),
                                               os.path.join(data_root, 'aug_voc_roi_info.pkl'),
                                               os.path.join(data_root, 'VOC_data'), repeat=5, thread_num=10)
    print('aug roi info done---------------------------------------')
    train_info = generate_train_pathes(aug_roi_info, os.path.join(data_root, 'voc_train_data.pkl'), thread_num=10)


if __name__ == '__main__':
    ssd_data_prepare_windows()
    # ssd_data_prepare_linux()
