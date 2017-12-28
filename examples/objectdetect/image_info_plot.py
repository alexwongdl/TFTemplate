"""
Created by Alex Wang
on 2017-12-21
"""
import pickle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import random

from examples.objectdetect import class_info

def plot_voc_roi_info(voc_roi_info_path):
    """
    展示ROI信息
    :param voc_roi_info_path:  e.g. E://data/voc_roi_info.pkl
    :return:
    """
    annotations_list = pickle.load(open(voc_roi_info_path, 'rb'))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    # for item in annotations_list:
    while True:
        item_ind = random.randint(0, len(annotations_list))
        item = annotations_list[item_ind]
        image_path = item['image_path']
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


def quit_figure(event):
    plt.close(event.canvas.figure)


if __name__ == '__main__':
    print('plot image info...')
    # plot_voc_roi_info('E://data/voc_roi_info.pkl')
    plot_voc_roi_info('E://data/aug_voc_roi_info.pkl')