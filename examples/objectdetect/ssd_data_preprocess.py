"""
Created by Alex Wang
On 2018-02-08
"""
import random
from multiprocessing import Pool

import numpy as np
from scipy.misc import imread

from examples.objectdetect.class_info import voc_classes, voc_classes_num
from examples.objectdetect.ssd_anchors import ssd_anchors

# anchors 层数
ssd_anchors_layers_num = len(ssd_anchors)

negative_label = [0] * voc_classes_num
negative_label[0] = 1


def process_one_image(data):
    result = {}
    index = data['index']
    # image = Image.open(data['image_path'])
    image = imread(data['image_path'], mode='RGB')
    result['image'] = np.array(image)

    fea_map_inds_batch = []
    box_reg_batch = []
    box_class_batch = []
    object_class_batch = []

    for i in range(ssd_anchors_layers_num):
        fea_map_inds_batch.append([])
        box_reg_batch.append([])
        box_class_batch.append([])
        object_class_batch.append([])

    for i in range(ssd_anchors_layers_num):
        positive_anchors_one_layer = data['positive_anchors'][i]
        for anchor in positive_anchors_one_layer:
            obj_cls_ind = np.argmax(anchor['object_cls'])
            obj_cls = voc_classes[obj_cls_ind]
            # if obj_cls == 'person' and random.random() > 0.2:
            # if obj_cls == 'person':
            #     continue
            # if obj_cls == 'car' and random.random() > 0.5:
            # if obj_cls == 'car':
            #     continue

            # test gather_nd
            # anchor['fea_map_ind'] = [ 1000000000000, 1000000000000, anchor['fea_map_ind'][2]]  ##get 0

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

    for i in range(ssd_anchors_layers_num):
        negative_anchors_one_layer = data['negative_anchors'][i]
        if len(fea_map_inds_batch[i]) == 0 and len(negative_anchors_one_layer) > 0:
            permutation_ind = np.random.permutation(len(negative_anchors_one_layer))
            anchor = negative_anchors_one_layer[permutation_ind[0]]
            fea_map_inds_batch[i].append([index] + anchor['fea_map_ind'])
            box_reg_batch[i].append(anchor['box_reg'])
            box_class_batch[i].append(anchor['cls'])
            object_class_batch[i].append(anchor['object_cls'])

    for i in range(ssd_anchors_layers_num):
        positive_anchors_one_layer = data['positive_anchors'][i]
        if len(fea_map_inds_batch[i]) == 0 and len(positive_anchors_one_layer) > 0:
            permutation_ind = np.random.permutation(len(positive_anchors_one_layer))
            anchor = positive_anchors_one_layer[permutation_ind[0]]
            fea_map_inds_batch[i].append([index] + anchor['fea_map_ind'])
            box_reg_batch[i].append(anchor['box_reg'])
            box_class_batch[i].append(anchor['cls'])
            object_class_batch[i].append(anchor['object_cls'])

    for i in range(ssd_anchors_layers_num):
        if len(fea_map_inds_batch[i]) == 0:
            fea_map_inds_batch[i].append([index] + [1, 1, 1])
            box_reg_batch[i].append([0, 0, 0, 0])
            box_class_batch[i].append([1, 0])
            object_class_batch[i].append(negative_label)

    result['fea_map_inds'] = fea_map_inds_batch
    result['box_reg'] = box_reg_batch
    result['box_class'] = box_class_batch
    result['object_class'] = object_class_batch
    return result


def batch_preprocess(train_data_batch):
    """

    :param train_data_batch:
    :return:
    """
    x_train_batch = []  # [FLAGS.batch_size, None, None, 3]
    fea_map_inds_batch = []  # [ssd_anchors_layers_num, None, 4]
    box_reg_batch = []  # [ssd_anchors_layers_num, None, 4]
    box_class_batch = []  # [ssd_anchors_layers_num, None, 2]
    object_class_batch = []  # [ssd_anchors_layers_num, None, voc_classes_num]

    for i in range(ssd_anchors_layers_num):
        fea_map_inds_batch.append([])
        box_reg_batch.append([])
        box_class_batch.append([])
        object_class_batch.append([])

    # print(train_data_batch[0]['image_path'])
    index = 0
    for data in train_data_batch:
        data['index'] = index
        index += 1
    pool = Pool(8)
    result = pool.map(process_one_image, train_data_batch)
    pool.close()
    pool.join()

    for data in result:
        x_train_batch.append(data['image'])
        for i in range(ssd_anchors_layers_num):
            fea_map_inds_batch[i].extend(data['fea_map_inds'][i])
            box_reg_batch[i].extend(data['box_reg'][i])
            box_class_batch[i].extend(data['box_class'][i])
            object_class_batch[i].extend(data['object_class'][i])

    return x_train_batch, fea_map_inds_batch, box_reg_batch, box_class_batch, object_class_batch
