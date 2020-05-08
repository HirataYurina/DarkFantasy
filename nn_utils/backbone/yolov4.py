# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:yolov4.py
# software: PyCharm

from backbone.CSPDarknet53 import cspdarknet_body
import tensorflow.keras.layers as layers
import tensorflow.keras as keras


# ------------------------ #
# 单次卷积 无激活函数层和BN层
# ------------------------ #
def darknet_conv2d(inputs, filters, kernel_size, strides, use_bias=True):
    padding = 'valid' if strides == 2 else 'same'

    y = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                      kernel_regularizer=keras.regularizers.l2(5e-4))(inputs)

    return y


# -------------------- #
# 卷积 BN + Leaky
# -------------------- #
def darknet_con2d_bn_leaky(inputs, filters, kernel_size, strides):
    y = darknet_conv2d(inputs, filters, kernel_size, strides, use_bias=False)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(0.1)(y)

    return y


def yolo4_body(inputs, num_anchors, num_classes):
    """
        yolo4:
            spp:spatial pyramid pool
            spp论文中，输出是一维的向量，因为在早期backbone中都存在全连接层，所以在全卷积神经网络中不适用
            yolo中对齐进行改进，使用concatenate进行通道整合，使用不同的pool size来增加backbone feature的感受野
            k={1, 5, 9, 13}
    :param inputs:
    :param num_anchors:
    :param num_classes:
    :return:
    """
    cspdarknet = keras.Model(inputs, cspdarknet_body(inputs))
    x = cspdarknet.output
    y19 = darknet_con2d_bn_leaky(x, 512, 1, 1)
    y19 = darknet_con2d_bn_leaky(y19, 1024, 3, 1)
    y19 = darknet_con2d_bn_leaky(y19, 512, 1, 1)

    # spp
    # increase respective field
    pool1 = layers.MaxPool2D(13, 1, padding='same')(y19)
    pool2 = layers.MaxPool2D(9, 1, padding='same')(y19)
    pool3 = layers.MaxPool2D(5, 1, padding='same')(y19)

    y19 = layers.Concatenate()([pool1, pool2, pool3, y19])

    # bottleneck
    y19 = darknet_con2d_bn_leaky(y19, 512, 1, 1)
    y19 = darknet_con2d_bn_leaky(y19, 1024, 3, 1)
    y19 = darknet_con2d_bn_leaky(y19, 512, 1, 1)
    # (13, 13, 256)
    y19_up = darknet_con2d_bn_leaky(y19, 256, 1, 1)
    # (26, 26, 256)
    y19_up = layers.UpSampling2D(2)(y19_up)
    # 38*38 head
    y38 = darknet_con2d_bn_leaky(cspdarknet.layers[204].output, 256, 1, 1)
    y38 = layers.Concatenate()([y38, y19_up])
    # make last layer
    y38 = darknet_con2d_bn_leaky(y38, 256, (1, 1), 1)
    y38 = darknet_con2d_bn_leaky(y38, 512, (3, 3), 1)
    y38 = darknet_con2d_bn_leaky(y38, 256, (1, 1), 1)
    y38 = darknet_con2d_bn_leaky(y38, 512, (3, 3), 1)
    y38 = darknet_con2d_bn_leaky(y38, 256, (1, 1), 1)

    y38_up = darknet_con2d_bn_leaky(y38, 128, 1, 1)
    y38_up = layers.UpSampling2D(2)(y38_up)

    # PAN 76*76 head
    y76 = darknet_con2d_bn_leaky(cspdarknet.layers[131].output, 128, 1, 1)
    y76 = layers.Concatenate()([y76, y38_up])

    y76 = darknet_con2d_bn_leaky(y76, 128, (1, 1), 1)
    y76 = darknet_con2d_bn_leaky(y76, 256, (3, 3), 1)
    y76 = darknet_con2d_bn_leaky(y76, 128, (1, 1), 1)
    y76 = darknet_con2d_bn_leaky(y76, 256, (3, 3), 1)
    y76 = darknet_con2d_bn_leaky(y76, 128, (1, 1), 1)

    # 76*76 output
    output76 = darknet_con2d_bn_leaky(y76, 256, 3, 1)
    output76 = darknet_conv2d(output76, num_anchors * (num_classes + 5), 1, 1)

    # 38*38 output
    y76_down = layers.ZeroPadding2D(((1, 0), (1, 0)))(y76)
    y76_down = darknet_con2d_bn_leaky(y76_down, 256, 3, 2)
    y38 = layers.Concatenate()([y76_down, y38])
    # make last layer
    y38 = darknet_con2d_bn_leaky(y38, 256, 1, 1)
    y38 = darknet_con2d_bn_leaky(y38, 512, 3, 1)
    y38 = darknet_con2d_bn_leaky(y38, 256, 1, 1)
    y38 = darknet_con2d_bn_leaky(y38, 512, 3, 1)
    y38 = darknet_con2d_bn_leaky(y38, 256, 1, 1)
    # predict
    output38 = darknet_con2d_bn_leaky(y38, 512, 3, 1)
    output38 = darknet_conv2d(output38, num_anchors * (num_classes + 5), 1, 1)

    # 19*19 output
    y38_dowm = layers.ZeroPadding2D(((1, 0), (1, 0)))(y38)
    y38_dowm = darknet_con2d_bn_leaky(y38_dowm, 512, 3, 2)
    # make last year
    y19 = layers.Concatenate()([y38_dowm, y19])
    y19 = darknet_con2d_bn_leaky(y19, 512, 1, 1)
    y19 = darknet_con2d_bn_leaky(y19, 1024, 3, 1)
    y19 = darknet_con2d_bn_leaky(y19, 512, 1, 1)
    y19 = darknet_con2d_bn_leaky(y19, 1024, 3, 1)
    y19 = darknet_con2d_bn_leaky(y19, 512, 1, 1)
    # predict
    output19 = darknet_con2d_bn_leaky(y19, 1024, 3, 1)
    output19 = darknet_conv2d(output19, num_anchors * (num_classes + 5), 1, 1)

    yolo4 = keras.Model(inputs, [output19, output38, output76])

    return yolo4


if __name__ == '__main__':

    inputs_ = keras.Input((608, 608, 3))

    yolo4_model = yolo4_body(inputs_, 3, 80)

    yolo4_model.summary()

    try:
        yolo4_model.load_weights(r'F:\百度云下载\2019深度学习\2019图像处理\代码\keras-yolo4-master\yolo4_weight.h5')
    except Exception:
        print('模型参数载入失败，请重试')
    else:
        print('模型参数载入成功')

    print(len(yolo4_model.layers))
