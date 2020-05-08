# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:cross_channel_pool.py
# software: PyCharm


import tensorflow as tf
import tensorflow.keras.layers as layers

"""
Maxout OP from https://arxiv.org/abs/1302.4389
Max pooling is performed in given filter/channel dimension. This can also be
used after fully-connected layers to reduce number of features.
Args:
    inputs: A Tensor on which maxout will be performed
    num_units: Specifies how many features will remain after max pooling at the
      channel dimension. This must be multiple of number of channels.
    axis: The dimension where max pooling will be performed. Default is the
      last dimension.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional scope for name_scope.
Returns:
    A `Tensor` representing the results of the pooling operation.
Raises:
    ValueError: if num_units is not multiple of number of features.
"""


def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


class CrossChannelPool2D(layers.Layer):

    def __init__(self, num_units, axis=None, **kwargs):
        super(CrossChannelPool2D, self).__init__(**kwargs)
        self.num_units = num_units
        self.axis = axis

    def call(self, inputs, **kwargs):
        num_units = self.num_units
        axis = self.axis

        shape = inputs.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1
        if axis is None:  # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError('number of features({}) is not '
                             'a multiple of num_units({})'.format(num_channels, num_units))
        shape[axis] = num_units
        shape += [num_channels // num_units]
        outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keepdims=False)
        return outputs

    def get_config(self):
        config = super(CrossChannelPool2D, self).get_config()
        return config


if __name__ == '__main__':
    data = tf.random.normal((4, 32, 32, 128))
    cross_channel_pool = CrossChannelPool2D(16)
    outputs = cross_channel_pool(data)
    print(outputs.shape)
