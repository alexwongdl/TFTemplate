"""
Created by Alex Wang
on 2017-07-25
"""

import datetime


def current_date_format():
    """
    获取当前时间字符串
    :return:
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def current_day_format():
    """
    获取当前日期字符串
    :return:
    """
    return datetime.datetime.now().strftime("%Y-%m-%d")


def kdays_ago_date_format(kdays):
    """
    获取n天之前的时间字符串
    :param kdays:
    :return:
    """
    kdays_ago_date = datetime.datetime.now() - datetime.timedelta(days=kdays)
    return kdays_ago_date.strftime("%Y-%m-%d %H:%M:%S")


def kdays_ago_day_format(kdays):
    """
    获取n天之前日期字符串
    :param kdays:
    :return:
    """
    kdays_ago_date = datetime.datetime.now() - datetime.timedelta(days=kdays)
    return kdays_ago_date.strftime("%Y-%m-%d")

def time_delta_date_format(kdays = 0, khours = 0, kmins = 0, ksec = 0):
    new_time = datetime.datetime.now() + datetime.timedelta(days = kdays, hours = khours, minutes=kmins, seconds=ksec)
    return new_time.strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    print(current_date_format())
    print(current_day_format())

    print(kdays_ago_date_format(5))
    print(kdays_ago_day_format(5))

    print(time_delta_date_format(khours=-2, ksec=-4))