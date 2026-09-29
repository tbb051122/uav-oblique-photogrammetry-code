# -*- coding: utf-8 -*-
"""
生成合成点云分割数据（用于无真实 LAS 时验证代码流程）

场景由论文 4.3.1 的五类地物构成：
0 未分类 / 1 建筑物 / 2 道路 / 3 植被 / 4 裸地 / 5 其他人工设施

用法:
    python make_demo_data.py --clouds 6 --points 6000
"""

import argparse
import os

import numpy as np


RNG = np.random.default_rng(2026)


def add_bare_ground(cols, xmin, xmax, ymin, ymax, n):
    """裸地（带轻微起伏）。"""
    x = RNG.uniform(xmin, xmax, n)
    y = RNG.uniform(ymin, ymax, n)
    z = RNG.normal(0.0, 0.12, n)
    r = RNG.uniform(120, 160, n)
    g = RNG.uniform(110, 150, n)
    b = RNG.uniform(90, 130, n)
    cols.append(np.column_stack([x, y, z, r, g, b,
                                 np.full(n, 4, dtype=np.int64)]))


def add_road(cols, xmin, xmax, y0, width, n):
    """道路：灰色带状区域，沿 X 方向。"""
    x = RNG.uniform(xmin, xmax, n)
    y = RNG.uniform(y0, y0 + width, n)
    z = RNG.normal(0.01, 0.02, n)
    r = RNG.uniform(90, 120, n)
    g = RNG.uniform(90, 120, n)
    b = RNG.uniform(95, 125, n)
    cols.append(np.column_stack([x, y, z, r, g, b,
                                 np.full(n, 2, dtype=np.int64)]))


def add_building(cols, cx, cy, sx, sy, h, n):
    """矩形建筑：四个立面 + 屋顶。"""
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    parts = []
    m = max(int(n * 0.14), 50)

    # 南/北立面
    for yy in (y0, y1):
        xx = RNG.uniform(x0, x1, m)
        zz = RNG.uniform(0.2, h, m)
        shade = 0.85 if yy > cy else 1.0
        base = RNG.uniform(160, 220, m) * shade
        parts.append(np.column_stack([
            xx, np.full(m, yy) + RNG.normal(0, 0.02, m), zz,
            base, base * 0.92, base * 0.82,
            np.full(m, 1, dtype=np.int64)]))
    # 东/西立面
    for xx in (x0, x1):
        yy = RNG.uniform(y0, y1, m)
        zz = RNG.uniform(0.2, h, m)
        shade = 0.8 if xx < cx else 0.95
        base = RNG.uniform(170, 220, m) * shade
        parts.append(np.column_stack([
            np.full(m, xx) + RNG.normal(0, 0.02, m), yy, zz,
            base, base * 0.92, base * 0.82,
            np.full(m, 1, dtype=np.int64)]))

    # 屋顶
    m = max(int(n * 0.44), 100)
    parts.append(np.column_stack([
        RNG.uniform(x0, x1, m),
        RNG.uniform(y0, y1, m),
        RNG.normal(h, 0.03, m),
        RNG.uniform(130, 170, m), RNG.uniform(130, 170, m),
        RNG.uniform(140, 180, m),
        np.full(m, 1, dtype=np.int64)]))
    cols.append(np.concatenate(parts, axis=0))


def add_vegetation(cols, cx, cy, rx, ry, rh, n):
    """植被：椭球状绿色点簇（地表以上）。"""
    # 在单位球内均匀采样，再按三个半径拉伸
    v = RNG.normal(size=(int(n * 1.5), 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= RNG.uniform(0, 1, size=(len(v), 1)) ** (1.0 / 3.0)
    x = cx + v[:, 0] * rx
    y = cy + v[:, 1] * ry
    z = np.abs(v[:, 2]) * rh
    keep = z >= 0.05
    x, y, z = x[keep][:n], y[keep][:n], z[keep][:n]
    if len(x) < n:
        pad = n - len(x)
        x = np.concatenate([x, RNG.normal(cx, rx / 3, pad)])
        y = np.concatenate([y, RNG.normal(cy, ry / 3, pad)])
        z = np.concatenate([z, np.abs(RNG.normal(0, rh / 3, pad)) + 0.1])
    cols.append(np.column_stack([
        x, y, z,
        RNG.uniform(30, 80, n), RNG.uniform(90, 170, n),
        RNG.uniform(40, 90, n),
        np.full(n, 3, dtype=np.int64)]))


def add_facility(cols, x, y, h, n):
    """其他人工设施：路灯 / 杆状物。"""
    z = RNG.uniform(0.1, h, n)
    r = RNG.uniform(200, 240, n)
    g = RNG.uniform(190, 225, n)
    b = RNG.uniform(160, 200, n)
    cols.append(np.column_stack([
        x + RNG.normal(0, 0.04, n), y + RNG.normal(0, 0.04, n), z,
        r, g, b, np.full(n, 5, dtype=np.int64)]))


def make_scene(n_points=6000):
    cols = []
    xmin, xmax, ymin, ymax = 0.0, 120.0, 0.0, 120.0
    add_bare_ground(cols, xmin, xmax, ymin, ymax, int(n_points * 0.22))

    add_road(cols, xmin, xmax, 25.0, 5.0, int(n_points * 0.10))
    add_road(cols, xmin, xmax, 80.0, 5.0, int(n_points * 0.10))

    for _ in range(RNG.integers(2, 5)):
        add_building(cols,
                     RNG.uniform(15, 105), RNG.uniform(38, 72),
                     RNG.uniform(8, 18), RNG.uniform(8, 18),
                     RNG.uniform(8, 20), int(n_points * 0.12))

    for _ in range(RNG.integers(4, 8)):
        add_vegetation(cols,
                       RNG.uniform(5, 115), RNG.uniform(5, 115),
                       RNG.uniform(2, 4), RNG.uniform(2, 4),
                       RNG.uniform(1.5, 3.5), int(n_points * 0.025))

    for _ in range(5):
        add_facility(cols, RNG.uniform(3, 117), RNG.uniform(3, 117),
                     RNG.uniform(4, 7), max(int(n_points * 0.006), 30))

    cloud = np.concatenate(cols, axis=0)
    RNG.shuffle(cloud)
    cloud[:, :3] = cloud[:, :3] - cloud[:, :3].mean(axis=0)
    return cloud


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="./data")
    parser.add_argument("--clouds", type=int, default=8)
    parser.add_argument("--points", type=int, default=6000)
    parser.add_argument("--val", type=int, default=2)
    parser.add_argument("--test", type=int, default=2)
    args = parser.parse_args()

    for split, n in (("train", args.clouds), ("val", args.val), ("test", args.test)):
        split_dir = os.path.join(args.out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for i in range(n):
            cloud = make_scene(args.points)
            path = os.path.join(split_dir, "scene_{:03d}.npy".format(i))
            np.save(path, cloud)
            print("生成 {}（{} 点）".format(path, len(cloud)))


if __name__ == "__main__":
    main()
