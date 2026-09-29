# -*- coding: utf-8 -*-
"""
建筑目标提取（对应论文 4.2.4）

流程：
1. 从语义分割结果中筛选建筑类别点云；
2. 使用 DBSCAN 对建筑点云按空间距离聚类，分离不同建筑目标；
3. 逐目标计算水平投影范围、二维边界与建筑高度。

用法:
    python building_extract.py --cloud scene.npy --eps 2.0 \
        --min-samples 20 --building-class 1 --output-dir buildings

输入文件列：
    .npy/.txt: x y z [r g b] [label]
    若含标签列（第 4 或第 7 列），直接筛选 building-class；
    也可通过 --pred-file 单独提供预测标签。
"""

import argparse
import os

import numpy as np

from data_utils import read_point_cloud, voxel_downsample


def dbscan_2d(points_xy, eps=2.0, min_samples=20):
    """
    DBSCAN（论文 4.2.4 步骤 5）。
    使用 scipy.spatial.cKDTree 加速邻域查询。
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise RuntimeError("DBSCAN 需要 scipy：pip install scipy")

    tree = cKDTree(points_xy)
    n = len(points_xy)
    labels = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.asarray(tree.query_ball_point(points_xy[i], eps))
        if len(neighbors) < min_samples:
            continue  # 噪声点
        labels[i] = cluster_id
        queue = list(neighbors)
        pos = 0
        while pos < len(queue):
            j = queue[pos]
            pos += 1
            if visited[j]:
                continue
            visited[j] = True
            seed = np.asarray(tree.query_ball_point(points_xy[j], eps))
            if len(seed) >= min_samples:
                queue.extend(seed.tolist())
            if labels[j] < 0:
                labels[j] = cluster_id
        cluster_id += 1
    return labels


def extract_buildings(xyz, labels, building_class=1, eps=2.0, min_samples=20,
                      voxel=None):
    """筛选建筑点并按 DBSCAN 聚类，返回簇统计。"""
    mask = labels == building_class
    if not mask.any():
        print("警告：没有找到类别 {} 的建筑点".format(building_class))
        return None
    bxyz = xyz[mask]
    if voxel:
        bxyz, _, _ = voxel_downsample(bxyz, voxel)

    cluster_labels = dbscan_2d(bxyz[:, :2], eps=eps, min_samples=min_samples)
    results = []
    for cid in np.unique(cluster_labels):
        if cid < 0:
            continue
        pts = bxyz[cluster_labels == cid]
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        results.append({
            "cluster_id": int(cid),
            "points": int(len(pts)),
            "xmin": float(x0), "ymin": float(y0),
            "xmax": float(x1), "ymax": float(y1),
            "height": float(max(pts[:, 2].max() - pts[:, 2].min(), 0.0)),
            "area": float((x1 - x0) * (y1 - y0)),
            "zmin": float(pts[:, 2].min()),
            "zmax": float(pts[:, 2].max()),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="建筑目标提取（论文 4.2.4）")
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--pred-file", default=None,
                        help="预测标签文件（与点云等长）；缺省读取点云自带标签列")
    parser.add_argument("--building-class", type=int, default=1)
    parser.add_argument("--eps", type=float, default=2.0)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--voxel", type=float, default=None,
                        help="聚类前体素下采样尺寸")
    parser.add_argument("--output-dir", default="./buildings")
    args = parser.parse_args()

    xyz, _, labels = read_point_cloud(args.cloud)
    if args.pred_file:
        _, _, pred_labels = read_point_cloud(args.pred_file)
        labels = pred_labels
    if labels is None:
        raise ValueError("点云没有标签列，请提供 --pred-file")
    if len(labels) != len(xyz):
        raise ValueError("预测标签数量与点云不一致")

    results = extract_buildings(
        xyz, labels, args.building_class, args.eps, args.min_samples,
        voxel=args.voxel)
    if results is None:
        return
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n建筑目标 {} 个：".format(len(results)))
    header = "cluster_id,points,xmin,ymin,xmax,ymax,height,zmin,zmax,area"
    print(header)
    rows = []
    for r in sorted(results, key=lambda d: d["xmin"]):
        rows.append("{cluster_id},{points},{xmin:.3f},{ymin:.3f},"
                    "{xmax:.3f},{ymax:.3f},{height:.3f},"
                    "{zmin:.3f},{zmax:.3f},{area:.2f}".format(**r))
        print(rows[-1])

    csv_path = os.path.join(args.output_dir, "building_targets.csv")
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")
    print("\n建筑目标信息已保存:", csv_path)


if __name__ == "__main__":
    main()
