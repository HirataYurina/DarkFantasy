# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:split_dataset.py
# software: PyCharm

import numpy as np
import os

# #######################################################################
# 对dataset进行划分，并且进行shuffle（保证train, val, test数据分布相似）
# #######################################################################

# 数据集路径
imgs_path = r'F:\百度云下载\2019深度学习\2019图像处理\安全隐患数据集\VOC2028/JPEGImages'  # 替换为自己的annotations路径
split_ratio = {'train': 0.6, 'val': 0.2, 'test': 0.2}

img_id_list = [img_id for img_id in os.listdir(imgs_path)]
num_imgs = len(img_id_list)
num_train = int(num_imgs * split_ratio['train'])
num_val = int(num_imgs * split_ratio['val'])
num_test = num_imgs - num_train - num_val

print('数据集总共包含%d张图片' % num_imgs)
print('---------------开始shuffle---------------')
np.random.shuffle(img_id_list)
print('---------------完成shuffle---------------')
print('---------------开始split---------------')
train_list = img_id_list[0:num_train]
val_list = img_id_list[num_train:num_train + num_val]
test_list = img_id_list[num_train + num_val:]
print('---------------完成split---------------')
print('train数据集共有%d张图片' % len(train_list))
print('validate数据集共有%d张图片' % len(val_list))
print('test数据集共有%d张图片' % len(test_list))

# 开始写入
train_file = './VOC2020/ImageSets/Main/train.txt'
train_f = open(train_file, 'w')
for i in train_list:
    train_f.write(i + '\n')
train_f.close()

test_file = './VOC2020/ImageSets/Main/test.txt'
test_f = open(test_file, 'w')
for i in test_list:
    test_f.write(i + '\n')
test_f.close()

val_file = './VOC2020/ImageSets/Main/val.txt'
val_f = open(val_file, 'w')
for i in val_list:
    val_f.write(i + '\n')
val_f.close()
