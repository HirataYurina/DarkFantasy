# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:kmeans.py
# software: PyCharm

import numpy as np


"""
    anchor boxes聚类
    1.读取所有的true boxes
    2.对true boxes进行k-means聚类
        yolo中，输出三个stage，k=9
    3.得到 anchor boxes
"""


def k_means(inputs, k):
    # inputs: (n, m)
    # 初始化质心
    n, m = np.array(inputs).shape
    range_max = np.max(inputs, axis=0)
    range_min = np.min(inputs, axis=0)
    range_diff = np.expand_dims(range_max - range_min, axis=0)
    range_min = np.expand_dims(np.min(inputs, axis=0), axis=0)
    rand = np.random.rand(k, m)
    # (k, m)
    centriod = range_min + range_diff * rand
    # (n, 1)
    cluster = np.zeros(shape=(n, 1))

    changed = True

    while changed:
        changed = False
        for i in range(n):
            classifier = np.argmin(np.sum(np.square(inputs[i] - centriod), axis=-1))
            if not cluster[i][0] == classifier:
                changed = True
                cluster[i][0] = classifier
        # 重新计算centroid
        for k in range(k):
            # 判断是否有属于k类别的样本
            inputs_k = np.where(cluster == k)[0].tolist()
            if inputs_k:
                centriod[k] = np.sum(inputs[np.where(cluster == k)[0]], axis=0) / \
                              len(inputs[np.where(cluster == k)[0]])

    return centriod, cluster


class YoloKmeans():

    def __init__(self, cluster_k, fname):
        self.k = cluster_k
        self.filename = fname

    def __box_iou(self, boxes, cluster):
        """
            iou作为聚类中的距离标准，单个box对多个cluster
        :param boxes: (n, 2) w, h
        :param cluster: (k, 2) w, h
        :return: iou
        """
        # (n, 1, 2)
        boxes = np.expand_dims(boxes, axis=-2)
        # (1, k, 2)
        cluster = np.expand_dims(cluster, axis=0)
        # (n, 1)
        boxes_area = boxes[..., 0] * boxes[..., 1]
        # (1, k)
        cluster_area = cluster[..., 0] * cluster[..., 1]
        # (n, k)
        inter_area = np.minimum(boxes[..., 0], cluster[..., 0]) * np.minimum(boxes[..., 1], cluster[..., 1])
        iou = inter_area / (boxes_area + cluster_area - inter_area)
        return iou

    def kmeans(self, boxes, k):
        num_boxes = boxes.shape[0]
        # 随机初始化k个anchor boxes
        # 由于w, h为整数，不适合利用范围进行随机初始化，直接从true boxes中进行k个采样
        centriod = boxes[np.random.choice(num_boxes, k, replace=False)]

        cluster = np.zeros(shape=(num_boxes,))
        changed = True
        while changed:
            changed = False
            # (n,)
            classifier = np.argmax(self.__box_iou(boxes, centriod), axis=1)
            if not (classifier == cluster).all():
                cluster = classifier
                changed = True
            for i in range(k):
                box_belong_k = np.where(classifier == i)[0].tolist()
                if box_belong_k:
                    centriod[i] = np.median(boxes[box_belong_k], axis=0)
        return centriod

    def __txt2box(self):
        f = open(self.filename, 'r')
        boxes = []
        for line in f.readlines():
            splits = line.split(' ')
            len_info = len(splits)
            for i in range(1, len_info):
                box = splits[i]
                box = box.split(',')
                box_w = int(box[2]) - int(box[0])
                box_h = int(box[3]) - int(box[1])
                boxes.append([box_w, box_h])
        f.close()
        boxes = np.array(boxes)

        return boxes

    def __box2txt(self, centriod):
        f = open('../anchors.txt', 'w')
        num_boxes = centriod.shape[0]
        for i in range(num_boxes):
            f.write('%d, %d\n' % (centriod[i][0], centriod[i][1]))
        f.close()

    def accuracy(self, boxes, centriod):
        accuracy = np.mean(np.max(self.__box_iou(boxes, centriod), axis=-1))
        return accuracy

    def get_kmeans_anchor(self):
        boxes = self.__txt2box()
        centriod = self.kmeans(boxes, self.k)
        centriod = centriod[np.lexsort(centriod.T)]
        self.__box2txt(centriod)
        accuracy = self.accuracy(boxes, centriod)
        print('精确度为{}'.format(accuracy))


if __name__ == '__main__':
    # 测试kmeans
    a = np.random.randint(0, 10, size=(100, 8))
    # seed = np.random.seed(1)
    centriod_, _ = k_means(a, 5)
    # print(centriod_)

    # 测试yolo_kmean
    yolo_kmeans = YoloKmeans(9, './2020_train.txt')
    yolo_kmeans.get_kmeans_anchor()

