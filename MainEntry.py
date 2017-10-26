"""
Created by Alex Wang
On 2017-10-24

commom steps:
1.argparser parameters
2.load data
3.build graph：including loss function，learning rate decay，optimization operation
4.summary
5.training、valid、save check point in loops
6.testing
"""
import argparse

from examples.mnist import *
from myutil.myprint import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default='mehod_one', help='method type', type=str)
    ## common params
    parser.add_argument('--max_iter', default=20000, help='max iterate times', type=int)
    parser.add_argument('--batch_size', default=16, help='batch size', type=int)
    parser.add_argument('--input_path', default=None, help='input data path', type=str)
    ## params for train mnist

    ## params for test mnist

    FLAGS = parser.parse_args()
    arg_parse_print(FLAGS)

    if FLAGS.type == 'train_mnist':
        # tl.files.load_mnist_dataset()
        train_mnist(FLAGS)
    elif FLAGS.type == 'test_mnist':
        test_mnist(FLAGS)
