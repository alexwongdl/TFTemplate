"""
Created by Alex Wang on 2017-12-25
目标检测训练图像增强
"""
import os
import numpy as np
import random
import tensorflow as tf
import tensorlayer as tl
import scipy
from scipy.sparse import csr_matrix
import pickle

from examples.objectdetect import detect_data_prepare
from examples.objectdetect import class_info
from multiprocessing import Pool


def class_to_csr_matrix(clas):
    """
    :return:
    """
    overlaps = np.zeros((len(clas), class_info.voc_classes_num), dtype=np.float32)
    for i in range(len(clas)):
        overlaps[i, clas[i]] = 1.0
    return csr_matrix(overlaps)


def cal_box_area(boxes):
    """
    计算box面积
    :return:
    """
    area = []
    for box in boxes:
        area.append((box[2] - box[0]) * (box[3] - box[1]))
    return area


def left_top_right_bottom_to_cywh(box_list):
    """
    left, top, right, bottom
    [left, top, right, bottom] --> [cx, cy, w, h]
    :param left:
    :param top:
    :param right:
    :param bottom:
    :return:
    """
    result = []
    for box in box_list:
        left, top, right, bottom = box[0], box[1], box[2], box[3]
        width = right - left
        height = bottom - top
        result.append([left + width / 2, top + height / 2, width, height])
    return result


def left_top_right_bottom_to_cywh_norm(box_list, image_width, image_height):
    """
    left, top, right, bottom
    [left, top, right, bottom] --> [cx, cy, w, h] with value in range [0,1]
    :return:
    """
    result = []
    for box in box_list:
        left, top, right, bottom = box[0], box[1], box[2], box[3]
        left_norm = left / image_width
        right_norm = right / image_width
        top_norm = top / image_height
        bottom_norm = bottom / image_height

        width = (right_norm - left_norm)
        height = (bottom_norm - top_norm)
        result.append([left_norm + width / 2, top_norm + height / 2, width, height])
    return result


def cywh_norm_to_left_top_right_bottom(box_list, image_width, image_height):
    """
    cx,cy, width, height
    [cx, cy, w, h] with value in range [0,1]-->[left, top, right, bottom]
    :param cx:
    :param cy:
    :param width:
    :param height:
    :param image_width:
    :param image_height:
    :return:
    """
    result = []
    for box in box_list:
        cx, cy, width, height = box[0], box[1], box[2], box[3]
        left = round((cx - width / 2) * image_width)
        top = round((cy - height / 2) * image_height)
        right = round((cx + width / 2) * image_width)
        bottom = round((cy + height / 2) * image_height)
        result.append([left, top, right, bottom])
    return result


def _data_aug_fn(roi_info):
    image_path = roi_info['image_path']
    image = tl.vis.read_image(image_path)
    image_width = roi_info['image_width']
    image_height = roi_info['image_height']
    clas = roi_info['gt_classes']
    coords = left_top_right_bottom_to_cywh_norm(
            roi_info['boxes'].tolist(),
            image_width=image_width, image_height=image_height)  # [cx,cy,w,h] with value in range [0,1]

    jitter = roi_info['jitter']
    output_im_size = roi_info['output_im_size']
    """
    图像增强
    :param image: 原始图片 image = tl.vis.read_image(image_path])
    :param clas:  gt_classes  array([15, 13])
    :param coords:[cx, cy, w, h] with values in range [0,1]
            [[0.48125000000000007, 0.3557692307692307, 0.19583333333333336, 0.37637362637362637], [0.5114583333333333, 0.5659340659340659, 0.6520833333333333, 0.7087912087912087]]
    :param jitter:
    :param output_im_size: 目标图像大小
    :return:
    """
    # im, ann = data
    # clas, coords = ann
    ## resize到高宽合适的大小
    scale = np.max((output_im_size[1] / float(image.shape[1]),
                    output_im_size[0] / float(image.shape[0])))
    image, coords = tl.prepro.obj_box_imresize(image, coords,
                                               [int(image.shape[0] * scale) + 2, int(image.shape[1] * scale) + 2],
                                               is_rescale=True, interp='bicubic')
    ## 几何增强 geometric transformation
    image, coords = tl.prepro.obj_box_left_right_flip(image,
                                                      coords, is_rescale=True, is_center=True, is_random=True)
    image, clas, coords = tl.prepro.obj_box_shift(image, clas,
                                                  coords, wrg=0.1, hrg=0.1, is_rescale=True,
                                                  is_center=True, is_random=True)
    image, clas, coords = tl.prepro.obj_box_zoom(image, clas,
                                                 coords, zoom_range=(1 - jitter, 1 + jitter),
                                                 is_rescale=True, is_center=True, is_random=True)
    image, clas, coords = tl.prepro.obj_box_crop(image, clas, coords,
                                                 wrg=output_im_size[1], hrg=output_im_size[0],
                                                 is_rescale=True, is_center=True, is_random=True)
    ## 光度增强 photometric transformation
    image = tl.prepro.illumination(image, gamma=(0.5, 1.5),
                                   contrast=(0.5, 1.5), saturation=(0.5, 1.5), is_random=True)
    # image = tl.prepro.adjust_hue(image, hout=0.1, is_offset=True,is_clip=True, is_random=True)
    # image = tl.prepro.pixel_value_scale(image, 0.1, [0, 255], is_random=True)
    ## 把数值范围从 [0, 255] 转到 [-1, 1] (可选)
    # im = im / 127.5 - 1.
    # im = np.clip(im, -1., 1.)

    # 更新roi_info------------------------------------------------------------------------
    scipy.misc.imsave(roi_info['aug_image_path'], image)
    new_roi_info = {}
    new_roi_info['image_name'] = roi_info['image_name']
    new_roi_info['xml_path'] = roi_info['xml_path']
    new_roi_info['image_path'] = roi_info['aug_image_path']
    new_roi_info['boxes_center_norm_rep'] = np.array(coords)
    new_image_height, new_image_width = image.shape[0], image.shape[1]
    new_roi_info['gt_classes'] = np.array(clas)
    new_roi_info['gt_overlaps'] = class_to_csr_matrix(clas)
    new_boxes = cywh_norm_to_left_top_right_bottom(coords, new_image_width, new_image_height)
    new_roi_info['seg_areas'] = np.array(cal_box_area(new_boxes))
    new_roi_info['boxes'] = np.array(new_boxes)
    new_roi_info['boxes_center_rep'] = np.array(left_top_right_bottom_to_cywh(new_boxes))
    new_roi_info['image_width'] = new_image_width
    new_roi_info['image_height'] = new_image_height

    return new_roi_info


def batch_image_augment(roi_info_path=None, save_path=None, image_save_dir=None, repeat=5, thread_num=4,
                        img_size=[512, 512]):
    """
    :param roi_info_path: voc_roi_info.pkl
    :param save_path: 保存路径
    :param image_save_dir:新图片保存地址
    :param repeat: 每张图片生成的增强图片个数
    :param img_size:输出图像大小
    :return:
    """
    if os.path.exists(save_path):
        print('load aug roi info from pickle file')
        aug_roi_info_list = pickle.load(open(save_path, 'rb'))
        return aug_roi_info_list

    os.mkdir(image_save_dir, exist_ok=True)
    jitter = 0.2
    roi_info_list = detect_data_prepare.load_image_annotations(pickle_save_path=roi_info_path)
    new_roi_info_list = []
    for roi_info in roi_info_list:
        image_name = roi_info['image_name']
        sub_strs = image_name.split('.')

        for i in range(repeat):
            new_obj = {}
            new_obj.update(roi_info)
            new_image_name = '{}_{}.{}'.format(sub_strs[0], i, sub_strs[1])
            new_obj['aug_image_path'] = os.path.join(image_save_dir, new_image_name)
            new_obj['jitter'] = 0.2
            new_obj['output_im_size'] = img_size
            new_roi_info_list.append(new_obj)

    pool = Pool(thread_num)
    aug_roi_info_list = pool.map(_data_aug_fn, new_roi_info_list)
    pool.close()
    pool.join()

    pickle.dump(aug_roi_info_list, open(save_path, 'wb'))
    return aug_roi_info_list


def test_tl_image_aug():
    """
    tl 目标检测图像预处理测试
    :return:
    """
    roi_info = detect_data_prepare.load_image_annotations(pickle_save_path='E://data/voc_roi_info.pkl')
    object0 = roi_info[70]
    print(object0)

    image_width = object0['image_width']
    image_height = object0['image_height']
    xywh_list = left_top_right_bottom_to_cywh_norm(
            object0['boxes'].tolist(),
            image_width=image_width, image_height=image_height)

    # [cx, cy ,w ,h] --> [left, top, right, bottom]
    xywh_to_box_list = cywh_norm_to_left_top_right_bottom(xywh_list,
                                                          image_width, image_height)

    image = tl.vis.read_image(object0['image_path'])
    print('boxes.tolist---------')
    print(object0['boxes'].tolist())
    print('xywh_list------------------')
    print(xywh_list)
    print('xywh_to_box_list------------------')
    print(xywh_to_box_list)
    im_flip, coords_flip = tl.prepro.obj_box_left_right_flip(image, coords=xywh_list, is_rescale=True, is_center=True,
                                                             is_random=False)
    print('coords_flip----------------')
    print(coords_flip)
    im_shfit, new_cls, coords_shift = tl.prepro.obj_box_shift(image, coords=xywh_list, classes=object0['gt_classes'],
                                                              wrg=0.1, hrg=0.1,
                                                              is_rescale=True, is_center=True, is_random=False)
    print('random shift---------------')
    print(coords_shift)

    test_dir = "E://data/test/"
    tl.vis.draw_boxes_and_labels_to_image(image, classes=object0['gt_classes'], coords=xywh_list, scores=[],
                                          classes_list=class_info.voc_classes, bbox_center_to_rectangle=True,
                                          save_name=os.path.join(test_dir, '_im_original.jpg'))

    tl.vis.draw_boxes_and_labels_to_image(im_flip, classes=object0['gt_classes'], coords=coords_flip, scores=[],
                                          classes_list=class_info.voc_classes, bbox_center_to_rectangle=True,
                                          save_name=os.path.join(test_dir, '_im_flip.jpg'))

    tl.vis.draw_boxes_and_labels_to_image(im_shfit, classes=new_cls, coords=coords_shift, scores=[],
                                          classes_list=class_info.voc_classes, bbox_center_to_rectangle=True,
                                          save_name=os.path.join(test_dir, '_im_shift.jpg'))

    object0['aug_image_path'] = os.path.join(test_dir, 'test_image_aug.jpg')
    object0['jitter'] = 0.2
    object0['output_im_size'] = [400, 400]
    new_object0 = _data_aug_fn(object0)
    print(new_object0)
    print(new_object0['gt_overlaps'].toarray())
    image_2 = tl.vis.read_image(new_object0['image_path'])
    tl.vis.draw_boxes_and_labels_to_image(image_2, classes=new_object0['gt_classes'],
                                          coords=new_object0['boxes_center_norm_rep'], scores=[],
                                          classes_list=class_info.voc_classes, bbox_center_to_rectangle=True,
                                          save_name=os.path.join(test_dir, 'aug_image_path_label.jpg'))
    # draw_boxes_and_labels_to_image(image, classes=[], coords=[],scores=[], classes_list=[],bbox_center_to_rectangle=True, save_name=None):
    # image : RGB image in numpy.array, [height, width, channel].
    # classes : list of class ID (int).
    # coords : list of list for coordinates.
    #     - [x, y, x2, y2] (up-left and botton-right)
    #     - or [x_center, y_center, w, h] (set bbox_center_to_rectangle to True).
    #     scores : list of score (int). (Optional)
    #     classes_list : list of string, for converting ID to string.
    #     bbox_center_to_rectangle : boolean, defalt is False.
    #     If True, convert [x_center, y_center, w, h] to [x, y, x2, y2] (up-left and botton-right).
    # save_name : None or string
    # The name of image file (i.e. image.png), if None, not to save image.


if __name__ == '__main__':
    print("image_augment...")
    test_tl_image_aug()
    batch_image_augment('E://data/voc_roi_info.pkl', 'E://data/aug_voc_roi_info.pkl', 'E://data/VOC_data', repeat=5,
                        thread_num=4)
