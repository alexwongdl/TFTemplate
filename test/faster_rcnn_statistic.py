"""
Created by Alex Wang on 2017-12-20
faster-rcnn统计
"""
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import matplotlib.pyplot as plt

from myutil import pathutil
from examples.objectdetect import ssd_data_prepare
from examples.objectdetect.class_info import voc_classes, voc_classes_num

def image_shape(image_path):
    img = Image.open(image_path)
    (width, height) = img.size
    return (width, height)

def voc_image_width_height_statistic():
    """
    VOC 图像长宽统计
    VOC:
        width:(127, 500)
        height:(96, 500)
    :return:
    """
    image_path_list = []
    image_dir = "E://data/VOCdevkit/VOC2007/JPEGImages";
    dir_obs_list, dir_list = pathutil.list_files(image_dir)
    for file in dir_obs_list:
        if file.endswith(".jpg"):
            image_path_list.append(file)
    print(image_path_list[0:10])
    pool = ThreadPool(4)
    image_shape_list = pool.map(image_shape, image_path_list)
    pool.close()
    pool.join()

    print('size of image_shape_list:{}'.format(len(image_shape_list)))
    print(image_shape_list[0:10])

    width_list, height_list = zip(* image_shape_list)
    sorted_width_list = sorted(width_list)
    sorted_height_list = sorted(height_list)
    print("width:{}" .format( sorted_width_list[0:100]))
    print("width:{}".format( sorted_width_list[-100:-1]))
    print("height:{}".format( sorted_height_list[0:100]))
    print("height:{}".format( sorted_height_list[-100:-1]))


    fig = plt.figure()
    plt.hist(width_list)
    plt.show(0)

    fig1 = plt.figure()
    plt.hist(height_list)
    plt.show()

def obj_statistic():
    """
    统计训练样本中各个object的数量
    boat:667
    car:3912
    horse:1505
    bird:1610
    bicycle:1167
    person:9473
    bus:856
    tvmonitor:770
    __background__:421137
    bottle:521
    cat:2202
    chair:1737
    dog:2373
    diningtable:779
    cow:806
    sofa:1153
    train:1532
    sheep:618
    pottedplant:1070
    motorbike:1372
    aeroplane:1228
    :return:
    """
    train_data_file = 'E://data/voc_train_data.pkl'
    train_info = ssd_data_prepare.generate_train_pathes(pickle_save_path=train_data_file)
    print(train_info[0])

    cls_obj_count = dict()
    for train_data in train_info:
        for layer in train_data['positive_anchors']:
            for anchor in layer:
                obj_cls_ind = np.argmax(anchor['object_cls'])
                obj_cls = voc_classes[obj_cls_ind]
                cls_obj_count[obj_cls] = cls_obj_count[obj_cls] + 1 if obj_cls in cls_obj_count else 1

        for layer in train_data['negative_anchors']:
            for anchor in layer:
                background = voc_classes[0]
                cls_obj_count[background] = cls_obj_count[background] + 1 if background in cls_obj_count else 1

    for key, value in cls_obj_count.items():
        print('{}:{}'.format(key, value))

if __name__ == '__main__':
    print("start statistic...")
    # voc_image_width_height_statistic()
    obj_statistic()
