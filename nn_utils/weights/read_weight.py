# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:read_weight.py
# software: PyCharm

import h5py

# ------------------------------ #
# 查看weights.h5文件内容
# ------------------------------ #

# 读取h5py文件
# 其实就是个字典文件
weights = h5py.File('./resnet50_coco_best_v2.1.0.h5', 'r')
attrs = weights['model_weights']
print(attrs)

layer_names = attrs.attrs['layer_names']
print(layer_names)

key = weights.keys()
print(key)

model_weights = weights.get('model_weights')
print(model_weights)


def print_(name):
    print(name)


print(model_weights.visit(print_))

print(model_weights['conv1']['conv1'].keys())  # <KeysViewHDF5 ['kernel:0']>

import tensorflow.keras.models as models
models.Model.load_weights()