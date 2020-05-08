# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:resnet50.py
# software: PyCharm

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

'''
    resnet50
    :param
        inputs:输入为keras.Input类型
    :return
        outputs:预测结果
    ----------------------------
    使用方式：
    resnet50 = keras.Model(inputs, outputs)
'''


def conv_block(inputs, kernel_size, filters, stage, block, strides=2):
    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'

    filter1, filter2, filter3 = filters

    x = layers.Conv2D(filter1, 1, strides=strides, name=conv_name + '2a')(inputs)
    x = layers.BatchNormalization(name=bn_name + '2a')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filter2, kernel_size, padding='same', name=conv_name + '2b')(x)
    x = layers.BatchNormalization(name=bn_name + '2b')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filter3, 1, name=conv_name + '2c')(x)
    x = layers.BatchNormalization(name=bn_name + '2c')(x)
    x = layers.ReLU()(x)

    y = layers.Conv2D(filter3, 1, strides=strides, name=conv_name + '1')(inputs)
    y = layers.BatchNormalization(name=bn_name + '1')(y)

    outputs = layers.Add()([x, y])
    outputs = layers.ReLU()(outputs)

    return outputs


def identity_block(inputs, kernel_size, filters, stage, block):
    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'

    filter1, filter2, filter3 = filters

    x = layers.Conv2D(filter1, 1, name=conv_name + '2a')(inputs)
    x = layers.BatchNormalization(name=bn_name + '2a')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filter2, kernel_size, name=conv_name + '2b', padding='same')(x)
    x = layers.BatchNormalization(name=bn_name + '2b')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filter3, 1, name=conv_name + '2c')(x)
    x = layers.BatchNormalization(name=bn_name + '2c')(x)
    x = layers.ReLU()(x)

    outputs = layers.Add()([x, inputs])
    outputs = layers.ReLU()(outputs)

    return outputs


def resnet50(inputs):
    x = layers.ZeroPadding2D((3, 3))(inputs)
    x = layers.Conv2D(64, 7, 2, name='conv1')(x)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.ReLU()(x)

    x = layers.MaxPool2D(3, strides=2, padding='same')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    return x


if __name__ == '__main__':
    inputs_ = keras.Input(shape=(600, 600, 3))

    outputs_ = resnet50(inputs_)
    res_model = keras.Model(inputs_, outputs_)
    # res_model.summary()
