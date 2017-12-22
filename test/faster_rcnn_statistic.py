"""
Created by Alex Wang on 2017-12-20
faster-rcnn统计
"""
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import matplotlib.pyplot as plt

from myutil import pathutil

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


if __name__ == '__main__':
    print("start statistic...")
    voc_image_width_height_statistic()
