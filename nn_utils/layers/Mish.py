# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:Mish.py
# software: PyCharm

import tensorflow.keras.layers as layers
import tensorflow as tf


class Mish(layers.Layer):

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        inputs = inputs * tf.tanh(tf.nn.softplus(inputs))
        return inputs

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


if __name__ == '__main__':
    data = tf.random.uniform(shape=(4, 32, 32, 3), minval=0, maxval=1)
    data = Mish()(data)
    print(data)
    mish = Mish()
    mish.trainable = False
    config_ = mish.get_config()
    print(config_)
