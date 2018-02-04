"""
Created by Alex Wang
on 2017-12-21
"""
import pickle
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np

from examples.objectdetect import class_info
from examples.objectdetect.class_info import voc_classes

# 'k':black
colors = ['b', 'g', 'r', 'c', 'm', 'y']
def plot_voc_roi_info(image_root_path, voc_roi_info_path):
    """
    展示ROI信息
    :param voc_roi_info_path:  e.g. E://data/voc_roi_info.pkl
    :return:
    """
    annotations_list = pickle.load(open(voc_roi_info_path, 'rb'))
    # for item in annotations_list:
    while True:
        item_ind = random.randint(0, len(annotations_list))
        item = annotations_list[item_ind]
        print(item)
        # image_path = item['image_path']
        image_path = os.path.join(image_root_path, os.path.basename(item['image_path']))
        gt_classes = item['gt_classes']
        boxes = item['boxes']
        img = Image.open(image_path)
        cid = plt.gcf().canvas.mpl_connect('button_press_event', quit_figure)
        # cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)

        print(gt_classes)
        for class_ind in gt_classes:
            print(class_info.voc_classes[class_ind])

        plt.imshow(img)
        currentAxis = plt.gca()
        for box in boxes:
            rect = patches.Rectangle((box[0], box[1]), (box[2] - box[0]), (box[3] - box[1]), linewidth=3,
                                     edgecolor=colors[random.randint(0, len(colors)-1)],
                                     facecolor='none')
            currentAxis.add_patch(rect)
        plt.show()


def plot_train_data(image_root_path, train_data_info_path):
    """
    展示ROI信息
    :param train_data_info_path:  e.g. E://data/voc_train_data.pkl
    :return:
    """
    train_info = pickle.load(open(train_data_info_path, 'rb'))
    # for item in annotations_list:
    while True:
        item_ind = random.randint(0, len(train_info) -1)
        item = train_info[item_ind]

        # image_path = item['image_path']
        image_path = os.path.join(image_root_path, os.path.basename(item['image_path']))
        gt_classes = item['gt_classes']
        positive_anchors = item['positive_anchors']
        item['negative_anchors'] = []
        # positive_anchors = item['negative_anchors']
        print(item)
        img = Image.open(image_path)
        cid = plt.gcf().canvas.mpl_connect('button_press_event', quit_figure)
        # cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)

        print(gt_classes)
        for class_ind in gt_classes:
            print(class_info.voc_classes[class_ind])

        plt.imshow(img)
        currentAxis = plt.gca()

        for positive_boxes in positive_anchors:
            for positive_one in positive_boxes:
                box = positive_one['anchor']
                rect = patches.Rectangle((box[0], box[1]), (box[2] - box[0]), (box[3] - box[1]), linewidth=3,
                                         edgecolor=colors[random.randint(0, len(colors)-1)],
                                         facecolor='none')
                currentAxis.add_patch(rect)
        plt.show()

def plot_train_data_processed(image_root_path, train_data_info_path):
    """
    展示ROI信息
    :param train_data_info_path:  e.g. E://data/voc_train_data.pkl
    :return:
    """
    train_info = pickle.load(open(train_data_info_path, 'rb'))
    # for item in annotations_list:
    while True:
        item_ind = random.randint(0, len(train_info) -1)
        item = train_info[item_ind]
        print(item)

        item['index'] = 0
        item_processed = process_one_image(item, image_root_path)
        gt_classes = item['gt_classes']
        # positive_anchors = item['positive_anchors']
        positive_anchors = item['negative_anchors']
        img = item_processed['image']
        fea_map_inds_batch = item_processed['fea_map_inds']
        # print('fea_map_inds_batch:')
        # print(fea_map_inds_batch)
        for i in range(len(fea_map_inds_batch)):
            print('{}:{}'.format(i, fea_map_inds_batch[i]))

        # print('object_class:')
        # print(item_processed['object_class'])
        cid = plt.gcf().canvas.mpl_connect('button_press_event', quit_figure)
        # cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)

        print('gt_classes:')
        print(gt_classes)
        for class_ind in gt_classes:
            print(class_info.voc_classes[class_ind])

        plt.imshow(img)
        currentAxis = plt.gca()

        for positive_boxes in positive_anchors:
            for positive_one in positive_boxes:
                box = positive_one['anchor']
                rect = patches.Rectangle((box[0], box[1]), (box[2] - box[0]), (box[3] - box[1]), linewidth=3,
                                         edgecolor=colors[random.randint(0, len(colors)-1)],
                                         facecolor='none')
                currentAxis.add_patch(rect)
        plt.show()

from examples.objectdetect.ssd_anchors import ssd_anchors
ssd_anchors_layers_num = len(ssd_anchors)
def process_one_image(data, image_root_path):
    result = {}
    index = data['index']
    image_path = os.path.join(image_root_path, os.path.basename(data['image_path']))
    image = Image.open(image_path)
    result['image'] = np.array(image)

    fea_map_inds_batch = [[],[],[],[]]
    box_reg_batch =  [[],[],[],[]]
    box_class_batch =  [[],[],[],[]]
    object_class_batch =  [[],[],[],[]]

    for i in range(ssd_anchors_layers_num):
        positive_anchors_one_layer = data['positive_anchors'][i]
        for anchor in positive_anchors_one_layer:
            obj_cls_ind = np.argmax(anchor['object_cls'])
            obj_cls = voc_classes[obj_cls_ind]
            if obj_cls == 'person' and random.random() > 0.2:
                continue
            if obj_cls == 'car' and random.random() > 0.5:
                continue

            fea_map_inds_batch[i].append([index] + anchor['fea_map_ind'])
            box_reg_batch[i].append(anchor['box_reg'])
            box_class_batch[i].append(anchor['cls'])
            object_class_batch[i].append(anchor['object_cls'])

    for i in range(ssd_anchors_layers_num):
        negative_anchors_one_layer = data['negative_anchors'][i]
        for anchor in negative_anchors_one_layer:
            if random.random() > 0.1:
                continue
            fea_map_inds_batch[i].append([index] + anchor['fea_map_ind'])
            box_reg_batch[i].append(anchor['box_reg'])
            box_class_batch[i].append(anchor['cls'])
            object_class_batch[i].append(anchor['object_cls'])

    result['fea_map_inds'] = fea_map_inds_batch
    result['box_reg'] = box_reg_batch
    result['box_class'] = box_class_batch
    result['object_class'] = object_class_batch
    return result

def quit_figure(event):
    plt.close(event.canvas.figure)

if __name__ == '__main__':
    print('plot image info...')
    # plot_voc_roi_info('E://data/voc_roi_info.pkl')
    # plot_voc_roi_info('E://data/VOC_data','E://data/aug_voc_roi_info.pkl')
    # plot_train_data('E://data/VOC_data','E://data/voc_train_data.pkl')
    plot_train_data_processed('E://data/VOC_data','E://data/voc_train_data.pkl')
