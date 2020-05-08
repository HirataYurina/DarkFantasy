# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:yolo4_loss.py
# software: PyCharm

from label.label_smooth import label_smooth
from losses.object_det.focal_iou_loss import *


# ---------------------------------------------------#
#   yolo_head
#   将预测结果进行解码
#   param:
#       features:网络预测结果
#       anchors:anchors
#       num_classes:number of classes
#       input_shape:输入图像的形状(n, h, w, c)
# ---------------------------------------------------#
def yolo_decode(features, anchors, num_classes, input_shape, cal_loss=False):

    # TODO 需要修改其中的代码 注意tf中的数据类型转化

    # (w,h)
    grid_shape = tf.shape(features)[1:3][::-1]
    num_anchors = len(anchors)
    # (1, 1, 1, 3, 2)
    anchors = tf.cast(tf.reshape(tf.constant(anchors), (1, 1, 1, num_anchors, 2)), features.dtype)

    features = tf.reshape(features, (-1, grid_shape[1], grid_shape[0], num_anchors, num_classes + 5))
    # (n, h, w, 3, 2)
    box_xy = features[..., :2]
    box_wh = features[..., 2:4]
    grid_x = tf.reshape(tf.keras.backend.arange(grid_shape[0]), (1, -1, 1, 1))
    # (h, w, 1, 2)
    grid_x = tf.tile(grid_x, (grid_shape[1], 1, 1, 1))
    grid_y = tf.reshape(tf.keras.backend.arange(grid_shape[1]), (-1, 1, 1, 1))
    grid_y = tf.tile(grid_y, (1, grid_shape[0], 1, 1))

    grid_xy = tf.concat([grid_x, grid_y], axis=-1)

    box_xy = 1 / (1 + tf.math.exp(-box_xy)) + grid_xy
    box_wh = anchors * tf.math.exp(box_wh)

    # 归一化
    box_xy = box_xy / grid_shape
    box_wh = box_wh / input_shape[::-1]

    confidence = 1 / (1 + tf.math.exp(-features[..., 4]))
    class_prob = 1 / (1 + tf.math.exp(-features[..., 5:]))

    if cal_loss:
        return grid_xy, features, box_xy, box_wh

    return box_xy, box_wh, confidence, class_prob


def yolo4_loss(args, anchors, num_classes, ignore_threshold=0.5,
               label_smoothing=0.05, use_obj_focal=True, use_giou_loss=False,
               use_diou_loss=True, use_class_focal_loss=True):
    """
        yolo4_loss:
            采用 focal loss 和 diou loss
            focal loss: 解决正负样本数量不平衡的问题，使得模型将更多的注意力放在区分困难样本上
            diou loss： 解决box regression中，方差损失函数和iou之间没有强关联的问题

        :param args: [y_pred, y_true]
        :param anchors: (n, 2)
        :param num_classes: number of classes
        :param ignore_threshold: minigate the unbalance between negative and positive examples
                when use_obj_focal=False
        :param label_smoothing: 进行标签平滑 防止模型预测结果过度自信，导致过拟合
        :param use_obj_focal: use focal loss
        :param use_giou_loss: use giou loss
        :param use_diou_loss: use diou_loss
        :param use_class_focal_loss: 在计算类别损失中，使用焦点损失函数

        :return: loss
    """
    num_layers = len(anchors) // 3
    y_pred = args[:num_layers]
    y_true = args[num_layers:]
    # input_shape grid_shape
    input_shape = tf.cast(tf.shape(y_pred[0])[1:3] * 32, tf.float32)
    grid_shape = [tf.cast(tf.shape(y_pred[i])[1:3], tf.float32) for i in range(num_layers)]
    # anchor mask
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]

    # loss
    loss = 0

    # bacth
    batch_size = tf.cast(tf.shape(y_pred[0])[0], tf.float32)
    # start compute loss
    for i in range(num_layers):
        # (batch, k, k, m, 5 + num_classes)
        y_pred_layer = y_pred[i]  # yolo_head input
        y_true_layer = y_true[i]

        location_loss_scale = 2 - y_true_layer[..., 2:3] * y_true_layer[..., 3:4]

        # object mask
        # (batch, k, k, m, 1)
        object_mask = y_true_layer[..., 4:5]
        true_classes = y_true_layer[..., 5:]

        if label_smoothing:
            true_classes = label_smooth(true_classes, label_smoothing)

        # grid:(k, k, 1, 2)
        # raw_pred: (batch, k, k, m, classes + 5) from logits
        # pred_xy: (batch, k, k, m, 2) 经过解码和归一化处理
        # pred_wh: (batch, k, k, m, 2)
        grid, raw_pred, pred_xy, pred_wh = yolo_decode(y_pred_layer,
                                                       anchors[anchor_mask[i]], num_classes, input_shape,
                                                       cal_loss=True)
        pred_box = tf.concat([pred_xy, pred_wh], axis=-1)
        # 对label进行编码
        # 不需要进行sigmoid解码，后面会对预测值进行sigmoid计算
        raw_ture_xy = y_true_layer[..., :2] * grid_shape[i][::-1] - grid
        # 需要log编码
        raw_true_wh = tf.math.log(y_true_layer[..., 2:4] * input_shape[::-1] / anchors[anchor_mask[i]])

        ignore_mask = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

        # 计算ignore_mask
        for b in range(int(batch_size)):
            # (k, k, m, 4)
            per_pred_box = pred_box[b]
            per_true_box = y_true_layer[b][..., :4]
            # (k, k, m)
            object_mask_bool = tf.cast(object_mask[b], tf.bool)[..., 0]
            # (n, 4)
            per_true_box = tf.boolean_mask(per_true_box, object_mask_bool)  # R - K + 1
            # (k, k, m ,n)
            iou = box_iou(per_pred_box, per_true_box)
            # (k, k, m)
            iou_max = tf.reduce_max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, tf.cast(iou_max < ignore_threshold, tf.float32))
        # (b, k, k, m)
        ignore_mask = ignore_mask.stack()
        # (b, k, k, m, 1)
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)

        # ------------------------ #
        #   正式计算loss
        # ------------------------ #
        if use_obj_focal:
            # (b, k, k, m, 1)
            confidence_loss = sigmoid_focal_loss(object_mask, y_pred_layer[..., 4:5])
        else:
            confidence_loss = object_mask * \
                              tf.nn.sigmoid_cross_entropy_with_logits(object_mask, y_pred_layer[..., 4:5]) + \
                              (1 - object_mask) * \
                              tf.nn.sigmoid_cross_entropy_with_logits(object_mask, y_pred_layer[..., 4:5]) * \
                              ignore_mask
        if use_giou_loss:
            box_loss = g_iou(pred_box, y_true_layer[..., 0:4])
            box_loss = object_mask * (1 - box_loss) * location_loss_scale
        elif use_diou_loss:
            box_loss = d_iou(pred_box, y_true_layer[..., 0:4])
            box_loss = object_mask * location_loss_scale * (1 - box_loss)
        else:
            box_loss = object_mask * location_loss_scale * \
                       tf.nn.sigmoid_cross_entropy_with_logits(raw_ture_xy, raw_pred[..., :2]) + \
                       object_mask * location_loss_scale * 0.5 * tf.math.square(raw_pred[..., 2:4] - raw_true_wh)
        if use_class_focal_loss:
            class_loss = sigmoid_focal_loss(y_pred_layer[..., 5:], true_classes)
        else:
            class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(true_classes, y_pred_layer[..., 5:])

        confidence_loss = tf.reduce_sum(confidence_loss) / batch_size
        box_loss = tf.reduce_sum(box_loss) / batch_size
        class_loss = tf.reduce_sum(class_loss) / batch_size

        loss += confidence_loss + box_loss + class_loss

    return loss
