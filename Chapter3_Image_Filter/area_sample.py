# -*- coding: utf-8 -*-
"""
测区空间网格划分（对应论文 2.6.1）

将研究区域 [xmin, xmax] x [ymin, ymax] 按边长 step 划分为规则网格，
以每个网格的中心点作为空间采样点。
"""

import numpy as np


def create_sample_points(xmin, xmax, ymin, ymax, step=10):
    """
    生成测区网格中心采样点。

    参数:
        xmin, xmax: X（东）方向范围（米）
        ymin, ymax: Y（北）方向范围（米）
        step:       网格尺寸 / 采样间距（米）

    返回:
        网格中心点列表，元素为 (x, y)
    """
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("测区范围无效：需要 xmax > xmin 且 ymax > ymin")
    if step <= 0:
        raise ValueError("网格尺寸 step 必须为正数")

    # 网格中心：xmin + step/2, xmin + step + step/2, ...
    x_centers = np.arange(xmin + step / 2.0, xmax, step)
    y_centers = np.arange(ymin + step / 2.0, ymax, step)
    if len(x_centers) == 0 or len(y_centers) == 0:
        raise ValueError("网格尺寸过大，测区内至少需要包含一个网格中心")

    points = []
    for x in x_centers:
        for y in y_centers:
            points.append((float(x), float(y)))
    return points


def sample_points_to_array(sample_points):
    """将采样点列表转换为 (N, 2) 数组，便于矢量计算。"""
    return np.asarray(sample_points, dtype=np.float64).reshape(-1, 2)
