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
import os
import argparse

from myutil.myprint import *
from examples.translation_data_prepare import prepare_data, corpora_to_id
from examples import translation_vanilla

os.environ["CUDA_VISIBLE_DEVICES"]='0'

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
    parser.add_argument('--learning_rate', default=1.0, help='learning rate', type=float)
    parser.add_argument('--decay_step', default=10000, help='decay step', type=int)
    parser.add_argument('--decay_rate', default=0.9, help='decay rate', type=float)

    # model log/save path
    parser.add_argument('--input_dir', default=None, help='input data path', type=str)
    parser.add_argument('--save_model_dir', default=None, help='model dir', type=str)
    parser.add_argument('--save_model_freq', default=10000, help='save check point frequence', type=int)
    parser.add_argument('--summary_dir', default=None, help='summary dir', type=str)
    parser.add_argument('--summary_freq', default=100, help='summary frequency', type=int)
    parser.add_argument('--print_info_freq', default=100, help='print training info frequency', type=int)
    # load checkpoint for initialization or inferencing if checkpoint is not None
    parser.add_argument('--checkpoint', default=None, help='pretrained model', type=str)
    parser.add_argument('--valid_freq', default=10000, help='validate frequence', type=int)

    """
    params for train nmt vanilla
    """
    parser.add_argument('--init_scale', default=0.04, help='the initial scale of the weights', type = float)
    parser.add_argument('--max_grad_norm', default=10, help='the maximum permissible norm of the gradient', type=float)
    # parser.add_argument('--max_epoch', default=14, help='the number of epochs trained with the initial learning rate', type=int) -- decay step
    parser.add_argument('--max_max_epoch', default=55, help='the total number of epochs for training', type=int)
    parser.add_argument('--keep_prob', default=0.35, help='the probability of keeping weights in the dropout layer', type=float)
    # vocabulary
    parser.add_argument('--vocab_dim', default=200, help='vocabulary dimension', type=int)

    """
    params for prepare data
    """
    parser.add_argument('--corpora_one_path', default=None, help='path of corpora one')
    parser.add_argument('--corpora_two_path', default=None, help='path of corpora two')
    parser.add_argument('--corpora_combine_path', default=None, help='path of combine one and two corpora')
    parser.add_argument('--dict_one_path', default=None, help='path of dict one')
    parser.add_argument('--dic_two_path', default=None, help='path of dict two')
    parser.add_argument('--corpora_combine_ID_path', default=None, help='path of combine one and two corpora with number representation')

    FLAGS = parser.parse_args()
    arg_parse_print(FLAGS)

    if FLAGS.task == 'train_nmt':
        print('train vanilla translation model.')
        translation_vanilla.train_rnn(FLAGS)
    elif FLAGS.task == 'test_nmt':
        print('test vanilla translation model.')
        # test_rnn(FLAGS)
    elif FLAGS.task =='prepare_data':
        print('start prepare data.')
        # def prepare_data(corpora_one, corpora_two, corpora_combine, dic_one_path, dic_two_path, corpora_combine_ID):
        prepare_data(FLAGS.corpora_one_path, FLAGS.corpora_two_path, FLAGS.corpora_combine_path,
                     FLAGS.dict_one_path, FLAGS.dic_two_path, FLAGS.corpora_combine_ID_path)
    elif FLAGS.task == 'corpus_to_id':
        corpora_to_id(FLAGS.corpora_one_path, FLAGS.corpora_two_path, FLAGS.corpora_combine_path,
                     FLAGS.dict_one_path, FLAGS.dic_two_path, FLAGS.corpora_combine_ID_path)