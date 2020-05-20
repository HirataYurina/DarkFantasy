# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:mosaic_aug.py
# software: PyCharm

import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def rand(a=0.0, b=1.0):
    return np.random.rand() * (b - a) + a


# TODO:test this code
def get_random_mosaic_data(annotations, input_shape, max_boxes=50, hue=.1, sat=1.5, val=1.5):
    """
        数据增强：
        resize
        flip
        hsv transformation
        mosaic
    Args:
        annotations: 数据集标注信息
        input_shape: 统一尺寸，例如(416, 416)
        max_boxes: 每张图片最多检测数量
        hue: 色调变换
        sat: 饱和度变换
        val: 明度变换

    Returns: augment data with mosaic

    """
    h, w = input_shape
    min_x = 0.4
    min_y = 0.4
    scale_min = 1 - min(min_x, min_y)
    scale_max = scale_min + 0.2

    place_x = [0, int(min_x * w), 0, int(min_x * w)]
    place_y = [0, 0, int(min_y * h), int(min_y * h)]

    imgs = []
    boxes_data = []
    index = 0

    for line in annotations:
        contents = line.split()
        img = Image.open(contents[0])
        img = img.convert('RGB')

        iw, ih = img.size

        boxes = np.array([np.array(list(map(int, box.split(',')))) for box in contents[1:]])

        # 1.resize
        rand_scale = rand(scale_min, scale_max)
        new_h = int(h * rand_scale)
        new_w = int(new_h * (w / h))
        img = img.resize((new_w, new_h), Image.BICUBIC)

        # 2.flip
        if rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 3.hsv transform
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(img) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        img = hsv_to_rgb(x)
        img = Image.fromarray((img * 255).astype(np.uint8))

        # 4.place img
        new_img = Image.new('RGB', (w, h), (128, 128, 128))
        dx = place_x[index]
        dy = place_y[index]
        new_img.paste(img, (dx, dy))
        new_img = np.array(new_img) / 255
        imgs.append(new_img)
        index += 1

        # 5.correct box
        if len(boxes) > 0:
            # correct resize
            boxes[..., [0, 2]] = boxes[..., [0, 2]] * new_w / iw
            boxes[..., [1, 3]] = boxes[..., [1, 3]] * new_h / ih
            # correct flip
            boxes[..., [0, 2]] = new_w - boxes[..., [0, 2]]
            # correct place
            boxes[..., [0, 2]] = boxes[..., [0, 2]] + dx
            boxes[..., [1, 3]] = boxes[..., [1, 3]] + dy
            # pick valid boxes
            boxes[..., [0, 1]][boxes[..., [0, 1]] < 0] = 0
            boxes[..., 2][boxes[..., 2] > w] = w
            boxes[..., 3][boxes[..., 3] > h] = h
            boxes_w = boxes[..., 2] - boxes[..., 0]
            boxes_y = boxes[..., 3] - boxes[..., 1]
            boxes = boxes[np.logical_and(boxes_w > 1, boxes_y > 1)]

            boxes_data.append(boxes)

    # 6.crop imgs
    cropx = np.random.randint(int(w * min_x), int(w * (1 - min_x)))
    cropy = np.random.randint(int(h * min_y), int(h * (1 - min_y)))
    merge_img = np.zeros((h, w, 3))
    merge_img[:cropy, :cropx, :] = imgs[0][:cropy, :cropx, :]
    merge_img[:cropy, cropx:, :] = imgs[1][:cropy, cropx:, :]
    merge_img[cropy:, :cropx, :] = imgs[2][cropy:, :cropx, :]
    merge_img[cropy:, cropx:, :] = imgs[3][cropy:, cropx:, :]

    new_boxes = crop_boxes(boxes_data, cropx, cropy)
    num_boxes = len(new_boxes)
    if num_boxes > max_boxes:
        np.random.shuffle(new_boxes)
        new_boxes = new_boxes[:max_boxes]

    return merge_img, new_boxes


def crop_boxes(boxes, cropx, cropy):

    cropped_boxes = []

    for i, boxes in enumerate(boxes):
        if i == 0:
            boxes[..., [0, 2]] = np.minimum(cropx, boxes[..., [0, 2]])
            boxes[..., [1, 3]] = np.minimum(cropy, boxes[..., [1, 3]])
            boxes_w = boxes[..., 2] - boxes[..., 0]
            boxes_h = boxes[..., 3] - boxes[..., 1]
            valid_boxes = boxes[np.logical_and(boxes_w >= 5, boxes_h >= 5)]
            cropped_boxes.append(valid_boxes)

        if i == 1:
            boxes[..., [0, 2]] = np.maximum(cropx, boxes[..., [0, 2]])
            boxes[..., [1, 3]] = np.minimum(cropy, boxes[..., [1, 3]])
            boxes_w = boxes[..., 2] - boxes[..., 0]
            boxes_h = boxes[..., 3] - boxes[..., 1]
            valid_boxes = boxes[np.logical_and(boxes_w >= 5, boxes_h >= 5)]
            cropped_boxes.append(valid_boxes)

        if i == 2:
            boxes[..., [0, 2]] = np.minimum(cropx, boxes[..., [0, 2]])
            boxes[..., [1, 3]] = np.maximum(cropy, boxes[..., [1, 3]])
            boxes_w = boxes[..., 2] - boxes[..., 0]
            boxes_h = boxes[..., 3] - boxes[..., 1]
            valid_boxes = boxes[np.logical_and(boxes_w >= 5, boxes_h >= 5)]
            cropped_boxes.append(valid_boxes)

        if i == 3:
            boxes[..., [0, 2]] = np.maximum(cropx, boxes[..., [0, 2]])
            boxes[..., [1, 3]] = np.maximum(cropy, boxes[..., [1, 3]])
            boxes_w = boxes[..., 2] - boxes[..., 0]
            boxes_h = boxes[..., 3] - boxes[..., 1]
            valid_boxes = boxes[np.logical_and(boxes_w >= 5, boxes_h >= 5)]
            cropped_boxes.append(valid_boxes)

    return np.concatenate(cropped_boxes, axis=0)
