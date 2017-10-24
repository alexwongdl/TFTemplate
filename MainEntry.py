"""
Created by Alex Wang
On 2017-10-24
"""
import argparse

from myutil.myprint import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default='mehod_one', help='method type', type= str)
    FLAGS = parser.parse_args()
    arg_parse_print(FLAGS)