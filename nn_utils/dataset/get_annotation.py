# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:get_annotation.py
# software: PyCharm

import xml.etree.ElementTree as et
import os


# ===============================================
# 把xml文件转化为img_id, x1, y1, x2, y2, class+id
# 文件件目录：./VOC2020/Annotations
# ===============================================

# 当前工作路径，绝对路径
wd = os.getcwd()

sets = [('2020', 'train'), ('2020', 'val'), ('2020', 'test')]
classes = []

with open('voc_classes_helmet.txt', 'r') as f:
    for line in f.readlines():
        if not line.isspace():
            line = line.strip()
            classes.append(line)

# print(classes)


# 读取xml文件
def convert_annotation(year, image_id, list_file):
    image_id = image_id.split('.')[0]
    in_file = open('./VOC%s/Annotations/%s.xml' % (year, image_id), encoding='utf-8')
    tree = et.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


# 开始读取并且写入text文件
for year, image_set in sets:
    image_ids = open('./VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
    for image_id in image_ids:
        list_file.write('%s/VOC%s/JPEGImages/%s' % (wd, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
