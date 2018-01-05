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
from examples.objectdetect import ssd_data_prepare

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
        print(item)
        # image_path = item['image_path']
        image_path = os.path.join(image_root_path, os.path.basename(item['image_path']))
        gt_classes = item['gt_classes']
        # positive_anchors = item['positive_anchors']
        positive_anchors = item['negative_anchors']
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

def quit_figure(event):
    plt.close(event.canvas.figure)

if __name__ == '__main__':
    print('plot image info...')
    # plot_voc_roi_info('E://data/voc_roi_info.pkl')
    # plot_voc_roi_info('E://data/VOC_data','E://data/aug_voc_roi_info.pkl')
    plot_train_data('E://data/VOC_data','E://data/voc_train_data.pkl')
