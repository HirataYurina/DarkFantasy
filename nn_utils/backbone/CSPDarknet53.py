# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:CSPDarknet53.py
# software: PyCharm

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from layers.Mish import Mish


# ------------------------ #
# 单次卷积 无激活函数层和BN层
# ------------------------ #
def darknet_conv2d(inputs, filters, kernel_size, strides, use_bias=True):
    padding = 'valid' if strides == 2 else 'same'

    y = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                      kernel_regularizer=keras.regularizers.l2(5e-4))(inputs)

    return y


# -------------------- #
# 卷积 BN + Mish
# -------------------- #
def darknet_con2d_bn_mish(inputs, filters, kernel_size, strides):
    y = darknet_conv2d(inputs, filters, kernel_size, strides, use_bias=False)
    y = layers.BatchNormalization()(y)
    y = Mish()(y)

    return y


# ----------------------------------- #
#   残差块：
#   应用cross stage partial进行梯度分流
# ----------------------------------- #
def res_block(x, num_filters, num_blocks, all_narrow=True):
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    preconv = darknet_con2d_bn_mish(x, num_filters, 3, 2)
    # 开始梯度分流
    # 直接连接到过度层
    shortconv = darknet_con2d_bn_mish(preconv, num_filters // 2 if all_narrow else num_filters, 1, 1)
    # 进入卷积block
    mainconv = darknet_con2d_bn_mish(preconv, num_filters // 2 if all_narrow else num_filters, 1, 1)

    for i in range(num_blocks):
        y = darknet_con2d_bn_mish(mainconv, num_filters // 2, 1, 1)
        '''
            1.如果all_narrow=True，就是remove bottleneck
                在CSPNet论文中，作者指出太多的bottleneck会使得太多的计算单元处于闲置状态，
                移除bottleneck可以使得神经元的利用率更加高
            2.过去CNN网络的计算消耗大的原因，主要是因为梯度冗余，大量的重复梯度参与反向传播
                Reference Paper:
                "CSPNet:a new backbone that can enhance learning capability of cnn"
                https://arxiv.org/abs/1911.11929v1
        '''
        y = darknet_con2d_bn_mish(y, num_filters // 2 if all_narrow else num_filters, 3, 1)
        # 残差连接
        mainconv = layers.Add()([mainconv, y])
    # 后卷积
    postconv = darknet_con2d_bn_mish(mainconv, num_filters // 2 if all_narrow else num_filters, 1, 1)
    # 进行连接
    route = layers.Concatenate()([postconv, shortconv])
    # partial transition layer
    outputs = darknet_con2d_bn_mish(route, num_filters, 1, 1)

    return outputs


def cspdarknet_body(x):
    """
        CSPDarknet
        :param x: backbone inputs (608, 608, 3)
        :return: backbone outputs (19, 19, 1024) (38, 38, 512) (76, 76, 1024)
    """
    y = darknet_con2d_bn_mish(x, 32, 3, 1)
    y = res_block(y, 64, 1, all_narrow=False)
    y = res_block(y, 128, 2)
    y = res_block(y, 256, 8)
    y = res_block(y, 512, 8)
    y = res_block(y, 1024, 4)
    return y


# if __name__ == '__main__':
    # x = keras.Input(shape=(608, 608, 3))
    # outputs = cspdarknet_body(x)
    # cspdarknet = keras.Model(x, outputs)
    # cspdarknet.summary()
