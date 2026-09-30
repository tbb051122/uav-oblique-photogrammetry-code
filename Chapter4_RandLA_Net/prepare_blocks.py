# -*- coding: utf-8 -*-
"""
大规模点云训练样本构建（对应论文 4.2.1“空间划分”与“样本构建”）

将完整 LAS/点云场景按 XY 网格切成可训练的分块，并分为 train/val/test。
流程：
1. 读入点云与标签（LAS classification 或 .npy 第 7 列）；
2. 体素下采样 / 统计离群点剔除；
3. 按 block_size 划分空间块；
4. 保存为 .npy [x,y,z,r,g,b,label]。

用法:
    python prepare_blocks.py --input scene1.las scene2.las --out-dir data \
        --block-size 100 --voxel 0.1
"""

import argparse
import os

import numpy as np

from data_utils import (remove_statistical_outliers, read_point_cloud,
                        voxel_downsample)


def split_blocks(xyz, block_size, overlap=0.0):
    """XY 网格分块，返回每个块的索引列表。"""
    if block_size <= 0:
        return [np.arange(len(xyz))]
    stride = block_size * (1.0 - overlap)
    xs = np.arange(xyz[:, 0].min(), xyz[:, 0].max(), stride)
    ys = np.arange(xyz[:, 1].min(), xyz[:, 1].max(), stride)
    blocks = []
    for bx in xs:
        for by in ys:
            mask = ((xyz[:, 0] >= bx) & (xyz[:, 0] < bx + block_size) &
                    (xyz[:, 1] >= by) & (xyz[:, 1] < by + block_size))
            idx = np.flatnonzero(mask)
            if len(idx) >= 100:
                blocks.append(idx)
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+",
                        help="一个或多个输入 LAS/.npy/.txt 场景")
    parser.add_argument("--out-dir", default="./data")
    parser.add_argument("--block-size", type=float, default=100.0,
                        help="分块尺寸（米）")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="分块重叠比例")
    parser.add_argument("--voxel", type=float, default=None,
                        help="体素下采样尺寸（米）")
    parser.add_argument("--outlier", action="store_true",
                        help="是否执行统计离群点剔除")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    xyz_list, colors_list, labels_list = [], [], []
    for path in args.input:
        xyz, colors, labels = read_point_cloud(path)
        if labels is None:
            raise ValueError("输入点云不包含标签，无法用于训练: " + path)
        xyz_list.append(xyz)
        colors_list.append(colors)
        labels_list.append(labels)
    xyz = np.concatenate(xyz_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    colors = (np.concatenate(colors_list, axis=0)
              if all(c is not None for c in colors_list) else None)
    if args.voxel:
        xyz, colors, labels = voxel_downsample(xyz, args.voxel, colors, labels)
    if args.outlier:
        xyz, colors, labels = remove_statistical_outliers(
            xyz, colors=colors, labels=labels)

    blocks = split_blocks(xyz, args.block_size, args.overlap)
    rng = np.random.default_rng(0)
    rng.shuffle(blocks)
    n_val = int(len(blocks) * args.val_ratio)
    n_test = int(len(blocks) * args.test_ratio)
    assignments = (["val"] * n_val + ["test"] * n_test +
                   ["train"] * (len(blocks) - n_val - n_test))

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(args.out_dir, split), exist_ok=True)

    saved = 0
    for bi, (split, idx) in enumerate(zip(assignments, blocks)):
        arr = [xyz[idx]]
        if colors is not None:
            arr.append(colors[idx])
            if labels is not None:
                arr.append(labels[idx].reshape(-1, 1))
        elif labels is not None:
            arr.append(labels[idx].reshape(-1, 1))
        out = os.path.join(args.out_dir, split, "block_{:05d}.npy".format(bi))
        np.save(out, np.concatenate(arr, axis=1))
        saved += 1
    print("完成：保存 {} 个分块到 {}".format(saved, args.out_dir))


if __name__ == "__main__":
    main()
