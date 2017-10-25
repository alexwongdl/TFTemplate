"""
Created by Alex Wang
On 2017-10-24
"""
import argparse

from myutil.myprint import *
from examples.mnist import *
import tensorlayer as tl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default='mehod_one', help='method type', type= str)
    ## common params

    ## params for train mnist

    ## params for test mnist

    FLAGS = parser.parse_args()
    arg_parse_print(FLAGS)

    if FLAGS.type == 'train_mnist':
        tl.files.load_mnist_dataset()
        train_mnist(FLAGS)
    elif FLAGS.type == 'test_mnist':
        test_mnist(FLAGS)
