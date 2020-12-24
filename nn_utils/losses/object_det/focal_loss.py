# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:focal_iou_loss.py
# software: PyCharm

import tensorflow as tf
import numpy as np


def sigmoid_focal_loss(y_true, y_pred, gamma=2, alpha=0.25):
    """
    计算 sigmoid focal loss
    Reference Paper:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    :param:
        y_true:所属类别的真实值
            (n,k,classes)
        y_pred:所属类别的预测值
            (n,k,classes)
        gamma:exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha:optional alpha weighting factor to balance positives vs negatives.
        根据论文的实验，最佳取值为gamma=2, alpha=0.25
    :return:

        sigmoid focal loss:
            (n,k,classes)
    """

    sigmoid_ce = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    # pt
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    factor = tf.pow(1 - pt, gamma)
    alpha = y_true * alpha + (1 - y_true) * (1 - alpha)

    return alpha * factor * sigmoid_ce


def softmax_focal_loss(y_true, y_pred, gamma, alpha):
    """
    计算 softmax focal loss
    Reference Paper:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    :param:
        y_true:所属类别的真实值
            (n,k,classes)
        y_pred:所属类别的预测值
            (n,k,classes)
        gamma:exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha:optional alpha weighting factor to balance positives vs negatives.
        根据论文的实验，最佳取值为gamma=2, alpha=0.25

    :return:
        softmax focal loss:
            (n,k,)
    """

    # 为了避免nan和inf 需要将输入进行裁剪
    # (epsilon, 1-epsilon)
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
    softmax_ce = - y_true * tf.math.log(y_pred)
    factor = tf.math.pow(1 - y_pred, gamma)
    alpha = alpha * y_true

    return alpha * factor * softmax_ce


if __name__ == '__main__':

    # ###################
    #   测试focal_loss
    # ###################
    y_true_ = np.random.randint(0, 2, (10, 3, 20)).astype('float32')
    y_pred_ = np.random.randint(0, 2, (10, 3, 20)).astype('float32')
    # sigmoid_focal_losses = sigmoid_focal_loss(y_true_, y_pred_)
    # print(sigmoid_focal_losses.shape)

    # ###########
    #   测试iou
    # ###########
    # (1, 4) (1, 20 ,4)做broadcast
    # random1 = tf.random.normal(shape=(1,))
    # random2 = tf.random.normal(shape=(1, 20,))
    # shape = tf.broadcast_static_shape(random1.shape, random2.shape)
    # print(shape)
