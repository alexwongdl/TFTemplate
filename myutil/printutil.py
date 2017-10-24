"""
Created by Alex Wang
On 2017-09-21
控制台输出带颜色的文字方法

格式：\033[显示方式;前景色;背景色m   分号分隔

说明：
前景色            背景色           颜色
---------------------------------------
30                40              黑色
31                41              红色
32                42              绿色
33                43              黃色
34                44              蓝色
35                45              紫红色
36                46              青蓝色
37                47              白色
显示方式           意义
-------------------------
0                终端默认设置
1                高亮显示
4                使用下划线
5                闪烁
7                反白显示
8                不可见

例子：
\033[1;31;40m    <!--1-高亮显示 31-前景色红色  40-背景色黑色-->
\033[0m          <!--采用终端默认设置，即取消颜色设置-->
"""

ANSI_BLACK = 30
ANSI_RED = 31
ANSI_GREEN = 32
ANSI_YELLOW = 33
ANSI_BLUE = 34
ANSI_PURPLE = 35
ANSI_CYAN = 36
ANSI_WHITE = 37


ANSI_BLACK_BACKGROUND = 40
ANSI_RED_BACKGROUND = 41
ANSI_GREEN_BACKGROUND = 42
ANSI_YELLOW_BACKGROUND = 43
ANSI_BLUE_BACKGROUND = 44
ANSI_PURPLE_BACKGROUND = 45
ANSI_CYAN_BACKGROUND = 46
ANSI_WHITE_BACKGROUND = 47

MOD_DEFAULT = 0
MOD_HIGHLIGHT = 1
MOD_UNDERLINE = 4
MOD_FLICKER = 5
MOD_INVERSE = 7
MOD_HIDE = 8

def mod_print(message, fg=ANSI_WHITE, bg=ANSI_BLACK_BACKGROUND, mod=MOD_DEFAULT):
    """
    格式化输出
    :param message:
    :param fg:
    :param bg:
    :param mod:
    :return:
    """
    print('\033[{};{};{}m'.format(fg, bg, mod) + message + '\033[0m')

def test():
    print('\033[1;32;40m')
    print('*' * 50)
    print('*HOST:\t', 2002)
    print('*URI:\t', 'http://127.0.0.1')
    print('*ARGS:\t', 111)
    print('*TIME:\t', '22:28')
    print('*' * 50)
    print('\033[0m')

if __name__ == "__main__":
    print("__main__")
    # test()
    mod_print("test_print", ANSI_GREEN, ANSI_BLACK_BACKGROUND, MOD_UNDERLINE)