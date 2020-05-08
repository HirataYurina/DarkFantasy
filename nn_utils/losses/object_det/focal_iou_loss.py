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


# --------------------------------------------- #
#   这里的iou用作计算每个pred box与其他box之间的IOU
#   用作nms中
# --------------------------------------------- #
def box_iou(b1, b2):
    """
        计算传统iou
        传统使用ln范数作为metric和loss有以下问题：
            iou与box回归的目标函数之间，没有强关联性，即：ln范数相同时，计算出的iou是不一样的
        假设，使用iou作为loss，有以下问题：
            当两个目标之间没有交集时，iou=0，导致无法衡量没有交集的目标在box坐标值上的差距
            当iou=0时，神经元关闭，无法进行反向传播，无法进行优化

        :param:
            b1: tensor, shape=(n1, 4) xywh xy:中心点
            b2: tensor, shape=(n2, 4) xywh

        :return:
            iou: (n1, n2)
    """

    # 改变shape 方便后续进行broadcast
    # (n1, 1, 4)
    b1 = tf.expand_dims(b1, axis=-2)
    # (1, n2, 4)
    b2 = tf.expand_dims(b2, axis=0)

    b1_xy = b1[..., 0:2]
    b1_wh = b1[..., 2:4]
    b2_xy = b2[..., 0:2]
    b2_wh = b2[..., 2:4]

    b1_left = b1_xy - b1_wh / 2
    b1_right = b1_xy + b1_wh / 2
    b2_left = b2_xy - b2_wh / 2
    b2_right = b2_xy + b2_wh / 2

    inter_left = tf.maximum(b1_left, b2_left)
    inter_right = tf.minimum(b1_right, b2_right)
    inter_wh = tf.maximum(0, inter_right - inter_left)

    # (n1, n2)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union = b1_area + b2_area - inter_area
    iou = inter_area / union

    return iou


def d_iou(b1, b2):
    """
        Calculate DIoU loss on anchor boxes
        giou存在的问题：
            在horizontal和vertical例子中，误差较大
            在回归过程中，当两个目标之间存在交集的时候，giou会退化为iou
            giou的收敛速度较慢
        diou:
            diou_loss = 1 - iou - distance(b,bgt)^2 / c^2
            c:  对角线的长度
        Reference Paper:
            "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
            https://arxiv.org/abs/1911.08287

        :param b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        :param b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        :return: diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    b1_xy = b1[..., 0:2]
    b1_wh = b1[..., 2:4]
    b1_min = b1_xy - b1_wh / 2
    b1_max = b1_xy + b1_wh / 2
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]

    b2_xy = b2[..., 0:2]
    b2_wh = b2[..., 2:4]
    b2_min = b2_xy - b2_wh / 2
    b2_max = b2_xy + b2_wh / 2
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]

    intersect_min = tf.maximum(b1_min, b2_min)
    intersect_max = tf.minimum(b1_max, b2_max)
    intersect_wh = tf.maximum(0, intersect_max - intersect_min)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    union = b1_area + b2_area - intersect_area
    # iou
    # (batch, w, h, num_anchor)
    iou = intersect_area / union
    # 中心点距离
    center_distance = tf.reduce_sum(tf.square(b1_xy - b2_xy), axis=-1)
    # diagonal distance
    diagonal_min = tf.minimum(b1_min, b2_min)
    diagonal_max = tf.maximum(b1_max, b2_max)
    diagonal_wh = tf.maximum(0, diagonal_max - diagonal_min)
    diagonal_distance = tf.reduce_sum(tf.square(diagonal_wh), axis=-1)

    diou = iou - center_distance / (diagonal_distance + tf.keras.backend.epsilon())
    diou = tf.expand_dims(diou, axis=-1)
    return diou


def g_iou(box1, box2):
    """
        Calculate GIoU loss on anchor boxes
        Reference Paper:
            "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
            https://arxiv.org/abs/1902.09630

        :param box2: (batch, w, h, anchor_num, 4)
        :param box1: (batch, w, h, anchor_num, 4)
        giou = iou - (AC-u)/AC
        AC:最小的能覆盖两个box的凸区域
        Returns
        -------
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 得到x1,y1,x2,y2
    # (batch, w, h, anchor_num, 2)
    b1_xy = box1[..., :2]
    b1_wh = box1[..., 2:4]
    b1_left = b1_xy - b1_wh / 2
    b1_right = b1_xy + b1_wh / 2

    b2_xy = box2[..., :2]
    b2_wh = box2[..., 2:4]
    b2_left = b2_xy - b2_wh / 2
    b2_right = b2_xy + b2_wh / 2

    inter_left = tf.maximum(b1_left, b2_left)
    inter_right = tf.minimum(b1_right, b2_right)
    inter_wh = tf.maximum(0, inter_right - inter_left)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    # (batch, w, h, anchor_num)
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    # 添加epsilon防止分母为0
    iou = inter_area / (b1_area + b2_area - inter_area + tf.keras.backend.epsilon())
    # 计算generate区域
    generate_left = tf.minimum(b1_left, b2_left)
    generate_right = tf.maximum(b1_right, b2_right)
    generate_wh = generate_right - generate_left
    generate_area = generate_wh[..., 1] * generate_wh[..., 0]
    # giou = iou - (ac - u) / ac
    giou = iou - (generate_area - (b1_area + b2_area - inter_area)) / (generate_area + tf.keras.backend.epsilon())
    # giou (batch, w, h, anchor_num) -> (batch, w, h, anchor_num, 1)
    giou = tf.expand_dims(giou, axis=-1)

    return giou


if __name__ == '__main__':

    # ----------------------- #
    #   测试focal_loss
    # ----------------------- #
    y_true_ = np.random.randint(0, 2, (10, 3, 20)).astype('float32')
    y_pred_ = np.random.randint(0, 2, (10, 3, 20)).astype('float32')
    # sigmoid_focal_losses = sigmoid_focal_loss(y_true_, y_pred_)
    # print(sigmoid_focal_losses.shape)

    # ----------------------- #
    #   测试iou
    # ----------------------- #
    # (1, 4) (1, 20 ,4)做broadcast
    # random1 = tf.random.normal(shape=(1,))
    # random2 = tf.random.normal(shape=(1, 20,))
    # shape = tf.broadcast_static_shape(random1.shape, random2.shape)
    # print(shape)

    # ----------------------- #
    #   测试diou
    # ----------------------- #

    # ----------------------- #
    #   测试giou
    # ----------------------- #
