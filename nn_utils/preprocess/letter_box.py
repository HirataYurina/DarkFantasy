# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:2020/4/13 0013 21:44
# filename:letter_box.py
# software: PyCharm

"""
    如果直接进行resize，会导致数据失真，所以使用padding可以避免图片失真
"""

import cv2


# ======================================
# letter_box进行图片填充
# 输入为单张图片
# 通常用于detection阶段
# ======================================
def letter_box(img, target_size):
    h = img.shape[0]
    w = img.shape[1]

    th, tw = target_size
    scale = min(th / h, tw / w)
    h = int(h * scale)
    w = int(w * scale)
    # 填充0
    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, (th - h) // 2, (th - h) // 2,
                             (tw - w) // 2, (tw - w) // 2, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return img


# ======================================
# letter_box逆变换(单张图片处理）
# bxy:中心点坐标
# bwh:bbox的宽高
# ======================================
def correct_boxes(bx, by, bw, bh, input_shape, img_shape):
    # bx, by按比例还原
    bx = bx * input_shape[1]
    by = by * input_shape[0]

    ih = img_shape[0]
    iw = img_shape[1]
    h = input_shape[0]
    w = input_shape[1]

    scale = min(h / ih, w / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)

    bx = bx - (w - nw) // 2
    by = by - (h - nh) // 2
    bx = int(bx / nw * iw)
    by = int(by / nh * ih)
    bw = bw / nw * iw
    bh = bh / nh * ih

    return bx, by, bw, bh


if __name__ == '__main__':
    input_img = cv2.imread('./img/1.png')
    height = input_img.shape[0]
    widtn = input_img.shape[1]

    # img_resize = cv2.resize(input_img, (416, 416))
    # cv2.imshow('resize', img_resize)
    # cv2.waitKey(0)
    # 图片失真
    # cv2.imwrite('./img/resize.jpg', img_resize)

    # 使用letter_box
    img_pad = letter_box(input_img, (416, 416))
    # cv2.imshow('pad', img_pad)
    # cv2.waitKey(0)
    # img_pad = cv2.circle(img_pad, (32, 32), 5, (0, 255, 0), thickness=-1)
    # cv2.imshow('pad', img_pad)
    # cv2.waitKey(0)
    print(img_pad.shape[0], img_pad.shape[1])

    # 测试correct_box
    input_bx = 0.2
    input_by = 0.2
    input_bw = 50
    input_bh = 50
    input_shape_ = (416, 416)
    img_shape_ = (height, widtn)
    input_bx, input_by, input_bw, input_bh = correct_boxes(input_bx, input_by,
                                                           input_bw, input_bh, input_shape_, img_shape_)
    print(input_bx)
    print(input_by)
    print(input_bw)
    print(input_bh)
