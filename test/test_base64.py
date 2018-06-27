"""
Created by Alex Wang
on 20180521
"""
import os
import base64
import cStringIO

import numpy as np
from PIL import Image
import cv2
import tensorflow as tf


def base64_encode_img(img):
    """
    :param img:
    :return:
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('shape of img_rgb:', img_rgb.shape)
    pil_img = Image.fromarray(img_rgb)

    buf = cStringIO.StringIO()
    pil_img.save(buf, format="JPEG", quality=100)
    b64code = base64.urlsafe_b64encode(buf.getvalue())
    # b64code = base64.b64encode('abcdefgdisoaufd,0.342,0.456,0.987')
    return b64code

    # img_tensor = tf.convert_to_tensor(buf.getvalue())
    # img_base64 = tf.encode_base64(img_tensor)
    # with tf.Session() as sess:
    #     img_base64_val = sess.run([img_base64])
    # return img_base64_val[0]

def base64_decode_img(b64code):
    """
    :param b64code:
    :return:
    """
    # img_base64_data = base64.b64decode(b64code)
    # img_nparr = np.fromstring(img_base64_data, np.uint8)
    # img = cv2.imdecode(img_nparr, cv2.COLOR_BGR2RGB)
    # print('shape of img:{}'.format(img.shape))

    base64_tensor = tf.convert_to_tensor(b64code, dtype=tf.string)
    img_str = tf.decode_base64(base64_tensor)
    img = tf.image.decode_image(img_str, channels=3)

    with tf.Session() as sess:
        # img_str_result = sess.run([img_str])[0]
        # print('img_str_result:{}'.format(img_str_result))
        img_value = sess.run([img])[0]
        print(img_value.shape)


def test_base64_img():
    """
    :return:
    """
    img_path = 'data/dl.jpg'
    img = cv2.imread(img_path)
    b64code = base64_encode_img(img)
    print('b64code:{}'.format(b64code))
    base64_decode_img(b64code)


if __name__ == '__main__':
    print('test')
    test_base64_img()
