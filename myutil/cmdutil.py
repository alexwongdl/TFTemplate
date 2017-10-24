'''
Created on 2017-05-16
@author:Alex Wang
执行shell命令
'''

import os
import subprocess
import threading
import time


def run(cmd, timeout=None, timeit=False):
    if timeit:
        start_time = time.time()
        # print('[+] start({}) {}'.format(timeout, ' '.join(cmd)))
        d = subprocess.run(cmd, stdout=subprocess.PIPE, timeout=timeout)
        elapsed_time = time.time() - start_time
        tid = threading.current_thread().ident
        pid = os.getpid()
        print('[+][{}][{}] done({})({}) {}'.format(pid, tid, elapsed_time, timeout, ' '.join(cmd)))
        return d.returncode, d.stdout
    d = subprocess.run(cmd, stdout=subprocess.PIPE, timeout=timeout)
    return d.returncode, d.stdout

def run_shell(cmd_str, timeout = 20):
    d = subprocess.run(cmd_str, stdout=subprocess.PIPE, timeout=timeout, shell = True)
    pid = os.getpid()
    return d.returncode, d.stdout

def test():
    filename =""
    timeout = 20
    retcode, stdout = run(['ffprobe', '-v', 'quiet', '-show_format', '-show_streams', '-print_format', 'json', filename],timeout=timeout)
    cmd_str = 'ls -ll | wc -l'
    run_shell(cmd_str)