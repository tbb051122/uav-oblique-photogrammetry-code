# -*- coding: utf-8 -*-
"""
对倾斜摄影点云做启发式预分类，输出带 Classification 的 LAS。

类别编号与论文一致：
    0 未分类  1 建筑物  2 道路  3 植被  4 裸地  5 其他人工设施

方法概要：
1. 用 1 m XY 网格内的低分位数高程拟合地面，再计算每个点的高程 h；
2. 用 0.5 m 体素的 PCA 特征描述局部平面性和粗糙度；
3. 用 RGB 绿度、亮度和高程把点分到上述类别；
4. 输出两份 LAS：一份保留原 RGB，一份用类别颜色着色，便于目视检查。

注意：这是预分类，不是人工真值。建议在 CloudCompare 中抽查并修正后再训练。
"""

import argparse
import os
import numpy as np
import laspy
from scipy import ndimage


CLASS_NAMES = {
    0: "未分类",
    1: "建筑物",
    2: "道路",
    3: "植被",
    4: "裸地",
    5: "其他人工设施",
}

CLASS_COLORS = {
    0: (255, 255, 255),
    1: (220, 60, 60),
    2: (90, 90, 90),
    3: (60, 180, 60),
    4: (185, 140, 90),
    5: (235, 200, 40),
}


def read_las(path):
    las = laspy.read(path)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    rgb = np.column_stack([las.red, las.green, las.blue]).astype(np.float64)
    if rgb.size:
        rgb /= 65535.0
    return xyz, rgb, las.header


def ground_grid(xyz, cell=1.0, percentile=5.0, min_points=5):
    """用 XY 网格低分位高程拟合地面，并返回最近邻补洞后的网格。"""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    x0, y0 = x.min(), y.min()
    ix = np.floor((x - x0) / cell).astype(np.int64)
    iy = np.floor((y - y0) / cell).astype(np.int64)
    nx, ny = ix.max() + 1, iy.max() + 1
    key = ix * ny + iy
    order = np.argsort(key)
    keys_sorted = key[order]
    z_sorted = z[order]
    boundaries = np.flatnonzero(np.diff(keys_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(keys_sorted)]])
    grid = np.full(nx * ny, np.nan, dtype=np.float64)
    for s, e in zip(starts, ends):
        if e - s >= min_points:
            grid[keys_sorted[s]] = np.percentile(z_sorted[s:e], percentile)
    grid = grid.reshape(nx, ny)
    valid = ~np.isnan(grid)
    if not valid.any():
        raise RuntimeError("地面网格为空")
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return grid[tuple(nearest)], ix, iy


def voxel_features(xyz, voxel=0.5):
    """返回每个点的局部平面性、粗糙度和法向竖直性。"""
    origin = xyz.min(axis=0)
    local = xyz - origin
    vi = np.floor((xyz - origin) / voxel).astype(np.int64)
    nx = vi[:, 0].max() + 1
    ny = vi[:, 1].max() + 1
    nz = vi[:, 2].max() + 1
    key = (vi[:, 0] * ny + vi[:, 1]) * nz + vi[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    n = inv.max() + 1
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    cnt = np.maximum(cnt, 1.0)

    x, y, z = local[:, 0], local[:, 1], local[:, 2]
    mx = np.bincount(inv, weights=x, minlength=n) / cnt
    my = np.bincount(inv, weights=y, minlength=n) / cnt
    mz = np.bincount(inv, weights=z, minlength=n) / cnt

    cov_xx = np.bincount(inv, weights=x * x, minlength=n) / cnt - mx * mx
    cov_yy = np.bincount(inv, weights=y * y, minlength=n) / cnt - my * my
    cov_zz = np.bincount(inv, weights=z * z, minlength=n) / cnt - mz * mz
    cov_xy = np.bincount(inv, weights=x * y, minlength=n) / cnt - mx * my
    cov_xz = np.bincount(inv, weights=x * z, minlength=n) / cnt - mx * mz
    cov_yz = np.bincount(inv, weights=y * z, minlength=n) / cnt - my * mz

    cov = np.zeros((n, 3, 3), dtype=np.float64)
    cov[:, 0, 0] = cov_xx
    cov[:, 1, 1] = cov_yy
    cov[:, 2, 2] = cov_zz
    cov[:, 0, 1] = cov[:, 1, 0] = cov_xy
    cov[:, 0, 2] = cov[:, 2, 0] = cov_xz
    cov[:, 1, 2] = cov[:, 2, 1] = cov_yz

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    # eigh 返回升序特征值：0 最小、2 最大
    planarity = (eigvals[:, 1] - eigvals[:, 0]) / np.maximum(eigvals[:, 2], 1e-12)
    roughness = np.sqrt(eigvals[:, 0])
    # 最小特征值对应法向，法向接近水平说明是竖直面
    verticality = 1.0 - np.abs(eigvecs[:, 2, 0])
    return planarity[inv], roughness[inv], verticality[inv]


def classify(xyz, rgb, args):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    exg = 2.0 * g - r - b
    green_dom = g - np.maximum(r, b)
    brightness = (r + g + b) / 3.0
    saturation = np.max(rgb, axis=1) - np.min(rgb, axis=1)

    ground_z, ix, iy = ground_grid(xyz, cell=args.ground_cell,
                                   percentile=args.ground_percentile,
                                   min_points=args.ground_min_points)
    h = z - ground_z[ix, iy]

    planarity, roughness, verticality = voxel_features(xyz, voxel=args.voxel)

    labels = np.zeros(len(xyz), dtype=np.uint8)

    green = (exg > args.exg_threshold) & (green_dom > args.green_dom_threshold)
    vegetation = green | ((h > args.veg_height) & (exg > args.veg_exg) &
                          (roughness > args.veg_roughness))

    building = ((h > args.building_height) & (planarity > args.building_planarity) &
                (roughness < args.building_roughness) & (~vegetation))
    building |= ((h > args.building_height) & (verticality > args.building_verticality) &
                 (~vegetation) & (saturation < args.building_saturation))

    ground = h <= args.ground_height
    ground_green = ground & green
    ground_other = ground & (~green)
    road = (ground_other & (brightness < args.road_brightness) &
            (roughness < args.road_roughness) & (saturation < args.road_saturation))
    bare = (ground_other & (~road) & (r - g > args.bare_rg) &
            (brightness > args.bare_brightness))
    other_ground = ground_other & (~road) & (~bare)
    other = (~ground) & (~vegetation) & (~building) & (h > args.other_height)

    labels[vegetation] = 3
    labels[ground_green & (labels == 0)] = 3
    labels[building & (labels == 0)] = 1
    labels[road & (labels == 0)] = 2
    labels[bare & (labels == 0)] = 4
    labels[other_ground & (labels == 0)] = 5
    labels[other & (labels == 0)] = 5

    # 低矮但不属于任何明显类别的点保持未分类，避免把噪声强行归类
    ambiguous = (labels == 5) & (~ground) & (~other)
    labels[ambiguous] = 0
    return labels, {
        "height": h,
        "exg": exg,
        "brightness": brightness,
        "planarity": planarity,
        "roughness": roughness,
        "verticality": verticality,
    }


def save_outputs(xyz, rgb, labels, header, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)

    def new_header():
        h = laspy.LasHeader(point_format=header.point_format.id, version=header.version)
        h.scales = header.scales
        h.offsets = header.offsets
        h.vlrs = header.vlrs
        return h

    out = laspy.LasData(new_header())
    out.x, out.y, out.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out.red = np.clip(rgb[:, 0] * 65535.0, 0, 65535).astype(np.uint16)
    out.green = np.clip(rgb[:, 1] * 65535.0, 0, 65535).astype(np.uint16)
    out.blue = np.clip(rgb[:, 2] * 65535.0, 0, 65535).astype(np.uint16)
    out.classification = labels
    path1 = os.path.join(out_dir, stem + "_preclassified.las")
    out.write(path1)

    colored = laspy.LasData(new_header())
    colored.x, colored.y, colored.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    colors = np.array([CLASS_COLORS[int(v)] for v in labels], dtype=np.float64) / 255.0
    colored.red = np.clip(colors[:, 0] * 65535.0, 0, 65535).astype(np.uint16)
    colored.green = np.clip(colors[:, 1] * 65535.0, 0, 65535).astype(np.uint16)
    colored.blue = np.clip(colors[:, 2] * 65535.0, 0, 65535).astype(np.uint16)
    colored.classification = labels
    path2 = os.path.join(out_dir, stem + "_class_colors.las")
    colored.write(path2)
    return path1, path2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--ground-cell", type=float, default=1.0)
    parser.add_argument("--ground-percentile", type=float, default=5.0)
    parser.add_argument("--ground-min-points", type=int, default=5)
    parser.add_argument("--ground-height", type=float, default=0.25)
    parser.add_argument("--voxel", type=float, default=0.5)
    parser.add_argument("--exg-threshold", type=float, default=0.03)
    parser.add_argument("--green-dom-threshold", type=float, default=0.01)
    parser.add_argument("--veg-height", type=float, default=0.4)
    parser.add_argument("--veg-exg", type=float, default=0.01)
    parser.add_argument("--veg-roughness", type=float, default=0.06)
    parser.add_argument("--building-height", type=float, default=1.5)
    parser.add_argument("--building-planarity", type=float, default=0.35)
    parser.add_argument("--building-roughness", type=float, default=0.15)
    parser.add_argument("--building-verticality", type=float, default=0.25)
    parser.add_argument("--building-saturation", type=float, default=0.18)
    parser.add_argument("--road-brightness", type=float, default=0.45)
    parser.add_argument("--road-roughness", type=float, default=0.05)
    parser.add_argument("--road-saturation", type=float, default=0.10)
    parser.add_argument("--bare-rg", type=float, default=0.015)
    parser.add_argument("--bare-brightness", type=float, default=0.25)
    parser.add_argument("--other-height", type=float, default=0.20)
    args = parser.parse_args()

    xyz, rgb, header = read_las(args.input)
    labels, feats = classify(xyz, rgb, args)

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.input), "preclassify")
    stem = os.path.splitext(os.path.basename(args.input))[0]
    path1, path2 = save_outputs(xyz, rgb, labels, header, out_dir, stem)

    counts = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    total = len(labels)
    report = [f"input={args.input}", f"points={total}", ""]
    for cid in range(6):
        n = counts.get(cid, 0)
        report.append(f"{cid} {CLASS_NAMES[cid]}: {n} ({n / total * 100:.2f}%)")
    report += ["", "feature quantiles:"]
    for name, values in feats.items():
        qs = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
        report.append(name + ": " + ", ".join(f"{v:.4f}" for v in qs))
    report += ["", "parameters:"]
    for key, value in sorted(vars(args).items()):
        report.append(f"{key}={value}")
    report_path = os.path.join(out_dir, stem + "_preclassify_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report[:10]))
    print("saved:", path1)
    print("saved:", path2)
    print("report:", report_path)


if __name__ == "__main__":
    main()
