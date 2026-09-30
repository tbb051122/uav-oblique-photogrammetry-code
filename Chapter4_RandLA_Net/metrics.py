# -*- coding: utf-8 -*-
"""
语义分割评价指标（对应论文 4.2.2）

* OA    总体准确率：正确分类点 / 总点数
* mAcc  平均类别准确率：各类别召回率的均值
* IoU   单类别交并比
* mIoU  各类别 IoU 的平均值
"""

import numpy as np


def confusion_matrix(preds, labels, num_classes, ignore_index=None):
    """
    计算混淆矩阵。

    参数:
        preds:  展平后的预测标签
        labels: 展平后的真实标签
        ignore_index: 需要忽略的类别（如“未分类”）
    返回:
        (K, K) numpy 混淆矩阵
    """
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    valid = labels >= 0
    if ignore_index is not None:
        valid &= (labels != ignore_index)
    preds = preds[valid]
    labels = labels[valid]
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    mask = (preds >= 0) & (preds < num_classes) & (labels >= 0) & (labels < num_classes)
    np.add.at(conf, (labels[mask], preds[mask]), 1)
    return conf


def compute_metrics(preds, labels, num_classes, ignore_index=None):
    """
    计算 OA / mAcc / mIoU 以及逐类别精度与 IoU。

    返回:
        dict:
          oa, macc, miou, class_acc (K,), class_iou (K,)
    """
    conf = confusion_matrix(preds, labels, num_classes, ignore_index)
    tp = np.diag(conf).astype(np.float64)
    n_gt = conf.sum(axis=1).astype(np.float64)   # 每类真实点数
    n_pred = conf.sum(axis=0).astype(np.float64)  # 每类预测点数

    oa = tp.sum() / max(conf.sum(), 1)

    class_acc = np.zeros(num_classes, dtype=np.float64)
    class_iou = np.zeros(num_classes, dtype=np.float64)
    present = n_gt > 0
    class_acc[present] = tp[present] / np.maximum(n_gt[present], 1.0)

    union = n_gt + n_pred - tp
    present_iou = union > 0
    class_iou[present_iou] = tp[present_iou] / union[present_iou]

    if ignore_index is not None and 0 <= ignore_index < num_classes:
        present[ignore_index] = False
        present_iou[ignore_index] = False

    macc = float(class_acc[present].mean()) if present.any() else 0.0
    miou = float(class_iou[present_iou].mean()) if present_iou.any() else 0.0

    return {
        "oa": float(oa),
        "macc": macc,
        "miou": miou,
        "class_acc": class_acc.tolist(),
        "class_iou": class_iou.tolist(),
    }
