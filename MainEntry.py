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
    parser.add_argument('-t', '--task', default='mehod_one', help='method type', type=str)
    """
    common params
    """
    parser.add_argument('--is_training', default=False, help='is training')

    parser.add_argument('--max_iter', default=20000, help='max training iterate times', type=int)
    parser.add_argument('--batch_size', default=16, help='batch size', type=int)
    # learning raate  exponential_decay  decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    parser.add_argument('--learning_rate', default=0.01, help='learning rate')
    parser.add_argument('--decay_step', default=10000, help='decay step')
    parser.add_argument('--decay_rate', default=0.9, help='decay rate')

    # model log/save path
    parser.add_argument('--input_dir', default=None, help='input data path', type=str)
    parser.add_argument('--save_model_dir', default=None, help='model dir', type=str)
    parser.add_argument('--save_model_freq', default=10000, help='save check point frequence', type=int)
    parser.add_argument('--summary_dir', default=None, help='summary dir')
    parser.add_argument('--summary_freq', default=100, help='summary frequency')
    parser.add_argument('--print_info_freq', default=100, help='print training info frequency')
    # load checkpoint for initialization or inferencing if checkpoint is not None
    parser.add_argument('--checkpoint', default=None, help='pretrained model')

    """
    params for train mnist
    """
    """
    params for test mnist
    """
    FLAGS = parser.parse_args()
    arg_parse_print(FLAGS)

    if FLAGS.task == 'train_mnist':
        # tl.files.load_mnist_dataset()
        train_mnist(FLAGS)
    elif FLAGS.task == 'test_mnist':
        test_mnist(FLAGS)
