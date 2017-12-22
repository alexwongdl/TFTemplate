"""
Created by Alex Wang on 2017-12-09
"""


voc_classes = ('__background__', # always index 0
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

voc_classes_num = len(voc_classes)
voc_class_to_ind = dict(zip(voc_classes, range(voc_classes_num)))

if __name__ == '__main__':
    print(voc_classes[7])