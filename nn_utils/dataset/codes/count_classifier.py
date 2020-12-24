# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:count_classifier.py
# software: PyCharm

import xml.etree.ElementTree as Tree
import os

files_list = os.listdir(r'F:\百度云下载\2019深度学习\2019图像处理\安全隐患数据集\VOC2012')

num_warning_sign = 0
num_no_reflective_cloth = 0
num_reflective_cloth = 0
num_staircase = 0
num_insulating_tool = 0
num_tool = 0

for file in files_list:
    file_name = os.path.join(r'F:\百度云下载\2019深度学习\2019图像处理\安全隐患数据集\VOC2012', file)
    root = Tree.parse(file_name).getroot()

    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'warning sign':
            num_warning_sign += 1
        elif name == 'no reflective cloth':
            num_no_reflective_cloth += 1
        elif name == 'reflective cloth':
            num_reflective_cloth += 1
        elif name == 'staircase':
            num_staircase += 1
        elif name == 'insulating tool':
            num_insulating_tool += 1
        elif name == 'tool':
            num_tool += 1

# -----------------------------------------------
# 将每个类别的数量写入日志
# -----------------------------------------------
log_name = './counter_log.txt'
with open(log_name, 'w') as f:
    f.write('The count of per classification:\n')
    f.write('the number of warning sign: ' + str(num_warning_sign) + '\n')
    f.write('the number of no_reflective_cloth: ' + str(num_no_reflective_cloth) + '\n')
    f.write('the number of reflective_cloth: ' + str(num_reflective_cloth) + '\n')
    f.write('the number of staircase: ' + str(num_staircase) + '\n')
    f.write('the number of insulating tool: ' + str(num_insulating_tool) + '\n')
    f.write('the number of tool: ' + str(num_tool) + '\n')
