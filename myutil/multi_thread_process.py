"""
Created by Alex Wang
On 2017-08-31

演示用map进行多线程和多进程操作
"""
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from urllib.request import urlopen
import os
import datetime

cwd = os.getcwd()
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  ##当前目录的上一级
file_path = os.path.join(parent_path, 'info.log')

def word_count(line):
    sub_str = line.split(' ')
    return len(sub_str)

def multi_thread(lines):
    pool = ThreadPool(4) # Sets the pool size to 4
    results = pool.map(word_count, lines)
    print(len(results))
    #close the pool and wait for the work to finish
    pool.close()
    pool.join()


def multi_process(lines):
    pool = Pool(4) # Sets the pool size to 4
    results = pool.map(word_count, lines)
    print(len(results))
    #close the pool and wait for the work to finish
    pool.close()
    pool.join()

def ord_process(lines):
    results = [word_count(line) for line in lines]
    print(len(results))

if __name__ == '__main__':
    print(parent_path)
    print(file_path)
    lines = []
    with open(file_path, 'r') as handler:
        line = handler.readline()
        while line:
            lines.append(line)
            line = handler.readline()

    lines = lines * 100000
    start_time = datetime.datetime.now()
    multi_process(lines)
    print(datetime.datetime.now() - start_time)
    start_time = datetime.datetime.now()
    multi_thread(lines)
    print(datetime.datetime.now() - start_time)
    start_time = datetime.datetime.now()
    ord_process(lines)
    print(datetime.datetime.now() - start_time)