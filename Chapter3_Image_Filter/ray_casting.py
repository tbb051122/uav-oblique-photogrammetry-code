# -*- coding: utf-8 -*-
"""
射线法（Ray Casting）空间包含关系判断（对应论文 2.5.3）

从待判断点向任意方向发射射线，统计射线与多边形边界的交点数量：
交点数为奇数 → 点在多边形内部；交点数为偶数 → 点在多边形外部。
"""

import numpy as np


def point_in_polygon(point, polygon):
    """
    判断单个点是否位于多边形内部（边界上的点视为内部）。

    参数:
        point:   (x, y)
        polygon: [(x1, y1), (x2, y2), ...]，多边形顶点按顺序排列

    返回:
        True / False
    """
    return bool(points_in_polygon([point], polygon)[0])


def points_in_polygon(points, polygon):
    """
    矢量化的点-多边形包含判断。

    参数:
        points:  (N, 2) 或 [(x, y), ...] 点集合
        polygon: [(x1, y1), (x2, y2), ...] 多边形顶点

    返回:
        (N,) bool 数组，True 表示点在多边形内（含边界）
    """
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 2)
    if pts.shape[1] != 2:
        raise ValueError("points 必须是 (N, 2) 数组")

    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError("polygon 必须是由 (x, y) 顶点组成的闭合多边形")

    xs = pts[:, 0]
    ys = pts[:, 1]
    n = len(poly)

    crossing = np.zeros(pts.shape[0], dtype=bool)
    on_boundary = np.zeros(pts.shape[0], dtype=bool)

    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        # 1) 边界判断：点在边上（叉积为 0 且在包围盒内）
        cross = (x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)
        in_bbox = (
            (xs >= min(x1, x2)) & (xs <= max(x1, x2))
            & (ys >= min(y1, y2)) & (ys <= max(y1, y2))
        )
        on_boundary |= np.isclose(cross, 0.0, atol=1e-9) & in_bbox

        # 2) 射线求交：边跨越采样点所在水平线时计算交点横坐标
        span = (y1 > ys) != (y2 > ys)
        denom = (y2 - y1)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = x1 + (ys - y1) * (x2 - x1) / denom
        crossing ^= span & (xs < x_intersect)

    inside = crossing | on_boundary
    if single:
        return inside.reshape(())
    return inside
