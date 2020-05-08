# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:get_random_augdata.py
# software: PyCharm

import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def rand(a, b):
    return np.random.rand() * (b - a) + a


"""
    随机数据增强：
        1.resize
        2.水平翻转
        3.光学扭曲
"""


def get_random_data(annotation_line, input_shape, random=True, max_boxes=100,
                    jitter=0.3, hue=0.1, sat=1.5, val=1.5):

    # [img_name, box1, box2, ...]
    lines = annotation_line.split()
    image = Image.open(lines[0])
    iw, ih = image.size
    h, w = input_shape
    # 将box转化为nparray
    # (n, 4)
    boxes = np.array([np.array(list(map(int, box.split(',')))) for box in lines[1:]])

    # 不进行数据增强
    if not random:
        scale = min(h / ih, w / iw)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        # letter box
        image = image.resize((nw, nh), Image.BICUBIC)
        new_img = Image.new('RGB', (w, h), (128, 128, 128))
        new_img.paste(image, (dx, dy))
        # 归一化
        img_data = np.array(np.array(new_img) / 255, dtype='float32')
        # 校正box
        # (x1, y1, x2, y2)
        # 设置最多选择max_boxes个box
        box_data = np.zeros((max_boxes, 5))
        if len(boxes) > 0:
            # shuffle
            np.random.shuffle(boxes)
            if len(boxes) > max_boxes:
                boxes = boxes[:max_boxes]
            boxes[..., [0, 2]] = boxes[..., [0, 2]] * scale + dx
            boxes[..., [1, 3]] = boxes[..., [1, 3]] * scale + dy
            box_data[0:len(boxes)] = boxes
        return img_data, box_data

    # 进行数据增强
    # resize
    aspect_ratio = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(0.25, 2)
    resize_w = int(h * scale)
    resize_h = int(resize_w / aspect_ratio)
    image = image.resize((resize_w, resize_h), Image.BICUBIC)
    # place img
    # 具有剪切效果
    # place image
    dx = int(rand(0, w - resize_w))
    dy = int(rand(0, h - resize_h))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 翻转
    filp = rand(0, 1) < 0.5
    if filp:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 光学扭曲
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand(0, 1) < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand(0, 1) < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # 校正box
    box_data = np.zeros((max_boxes, 5))
    if len(boxes) > 0:
        np.random.shuffle(boxes)
        if len(boxes) > max_boxes:
            boxes = boxes[:max_boxes]
        boxes[..., [0, 2]] = boxes[..., [0, 2]] * resize_w / iw + dx
        boxes[..., [1, 3]] = boxes[..., [1, 3]] * resize_h / ih + dy
        if filp:
            boxes[..., [0, 2]] = w - boxes[..., [2, 0]]
        # 对超出图片范围的box进行校正
        # 对长宽小于1的box进行筛选
        boxes[boxes < 0] = 0
        boxes[boxes > w] = w
        boxes[boxes > h] = h
        boxes_wh = boxes[..., 2:4] - boxes[..., 0:2]
        boxes = boxes[np.logical_and(boxes_wh[:, 0] > 1, boxes_wh[:, 1] > 1)]
        box_data[:len(boxes)] = boxes

    return image_data, box_data
