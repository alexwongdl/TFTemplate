"""
Created by Alex Wang
On 20170922
"""
import os

def get_basename(path):
    """
    获取路径的最后一部分
    :return:
    """
    return os.path.basename(path)

def dir_clear(dir_path):
    """
    清空一个目录
    :param dir_path:
    :return:
    """
    for file in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path,file)):
            os.remove(os.path.join(dir_path,file))

def list_dirs(path):
    """
    列出当前目录下的所有目录
    :param path:
    :return:
    """
    dir_obs_list = [] # 绝对路径
    dir_list = []  # 相对路径
    for file in os.listdir(path):
        obs_file_path = os.path.join(path, file)
        if os.path.isdir(obs_file_path):
            dir_obs_list.append(obs_file_path)
            dir_list.append(file)
    return dir_obs_list, dir_list

def list_files(path):
    """
    列出当前目录下的所有文件（非目录）
    :param path:
    :return:
    """
    file_obs_list = [] # 绝对路径
    file_list = []  # 相对路径
    for file in os.listdir(path):
        obs_file_path = os.path.join(path, file)
        if os.path.isfile(obs_file_path):
            file_obs_list.append(obs_file_path)
            file_list.append(file)
    return file_obs_list, file_list

def dir_exist(dir_path):
    """
    判断目录是否存在
    :param dir_path:
    :return:
    """
    if os.path.isdir(dir_path):
        return True
    else:
        return False

def file_exist(file_path):
    """
    判断文件（非目录）是否存在
    :param file_path:
    :return:
    """
    if os.path.isfile(file_path):
        return True
    else:
        return False

def test_current_dir():
    print(os.getcwd()) ##当前目录
    print(os.path.abspath(__file__)) ##当前文件
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  ##当前目录的上一级
    print(cwd)

if __name__ == "__main__":
    print("__main__")
    print(get_basename('http://flv3.bn.netease.com/videolib3/1707/24/HhsvJ4943/HD/HhsvJ4943-mobile.mp4'))

    dir_obs_list, dir_list = list_dirs('E://workspace/gitlab')
    print(", ".join(dir_obs_list))
    print(", ".join(dir_list))

    file_obs_list, file_list = list_files('E://workspace/gitlab')
    print(", ".join(file_obs_list))
    print(", ".join(file_list))

    test_current_dir()