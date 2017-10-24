'''
Created on 2017-05-16
@author:Alex Wang
字符串处理
'''

def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        return bytes_or_str.decode('utf-8')
    else:
        return bytes_or_str

def to_bytes(bytes_or_str):
    if isinstance(bytes_or_str, str):
        return bytes_or_str.encode('utf-8')
    else:
        return bytes_or_str

def test():
    print("test")
    print(to_bytes("kdjfkd"))
    print(to_str(b'dsfjksdl'))