# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:label_smooth.py
# software: PyCharm
import tensorflow as tf
import numpy as np


def label_smooth(y_true, label_smoothing):
    """
     标签平滑：
        通过标签平滑使得极值点有界限，而不是inf
        标签平滑可以缩小类内间距，增大类间间距

    :param y_true: one hot 标签 (n,k)
    :param label_smoothing: 平滑值 为一个小常数 (n,k)

    :return:smoothed_label
    """

    label_smoothing = tf.constant(label_smoothing, dtype=tf.float32)

    k = tf.shape(tf.constant(y_true))[-1]
    k = tf.cast(k, tf.float32)
    # print(k)
    smoothed_label = y_true * (1 - label_smoothing) + label_smoothing / k
    return smoothed_label


if __name__ == '__main__':
    y_true_ = np.random.randint(0, 2, (4, 20))
    smoothed_label_ = label_smooth(y_true_, 0.001)
    print(smoothed_label_)
