'''
Created on 2017-5-17
@author: Alex Wang

Config文件解析模块，config文件格式如下，中括号[]内是section，section内是key-value的配置neirong
example:
[db]
db_host = 127.0.0.1
db_port = 22
db_user = root
db_pass = rootroot

[concurrent]
thread = 10
processor = 20
'''

import configparser

def get_value(path, section, key):
    """
    读取配置文件
    :param path:
    :param section:
    :param key:
    :return:
    """
    config = configparser.ConfigParser()
    config.read(path)
    return config.get(section, key)

def set_value(path, section, key, value):
    """
    更新配置文件
    :param path:
    :param section:
    :param key:
    :param value:
    :return:
    """
    config = configparser.ConfigParser()
    config.read(path)
    sections = config.sections()
    if not sections.__contains__(section):
        config.add_section(section)
    config.set(section, key, value)
    config.write(open(path, 'w'))

