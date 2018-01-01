"""
Created by Alex Wang on 2018-1-1
SSD数据准备
"""
import numpy as np
import os
import sys
import pickle
from multiprocessing import Pool
import collections
import time

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  ##当前目录的上一级
sys.path.append(root_path)

from examples.objectdetect.ssd_anchors import ssd_anchors, ssd_anchors_area, ssd_anchors_step, ssd_anchors_step_num, ssd_anchor_info
from examples.objectdetect import detect_data_prepare
from examples.objectdetect import img_aug



def process_one_image_roi(annotation_feature, max_positive_num = 128, max_sample_num = 256):
    """
    在一张图片中随机选取ROI区域
    :param annotation_feature:
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
    


    return annotation_feature

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

def ssd_data_prepare_windows():
    """
    在windows上准备训练数据
    :return:
    """
    image_list = detect_data_prepare.get_image_list('E://data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt')
    print('len of image_list:{}'.format(len(image_list)))
    print(image_list[0:3])

    roi_info = detect_data_prepare.load_image_annotations('E://data/VOCdevkit/VOC2007/Annotations/',
                                      'E://data/VOCdevkit/VOC2007/JPEGImages', image_list, 'E://data/voc_roi_info.pkl')
    print('len of roi_info:{}'.format(len(roi_info)))
    aug_roi_info = img_aug.batch_image_augment('E://data/voc_roi_info.pkl', 'E://data/aug_voc_roi_info.pkl', 'E://data/VOC_data', repeat=5, thread_num=4, img_size= [512, 512])
    train_info = generate_train_pathes(aug_roi_info[0:60], 'E://data/voc_train_data.pkl', thread_num=4)

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
    train_info = generate_train_pathes(aug_roi_info,  os.path.join(data_root, 'voc_train_data.pkl'), thread_num=10)


if __name__ == '__main__':

    ssd_data_prepare_windows()
    # ssd_data_prepare_linux()