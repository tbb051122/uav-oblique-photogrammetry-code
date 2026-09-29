# -*- coding: utf-8 -*-
"""
用预分类 LAS 渲染论文用点云语义标注图。

输出：
1. 四块瓦片标注总览（2x2 俯视图，共享图例）
2. 单块瓦片三维标注视图
"""

import os
import numpy as np
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CLASS_NAMES = ["建筑物", "道路", "植被", "裸地", "其他人工设施"]
CLASS_IDS = [1, 2, 3, 4, 5]
CLASS_COLORS = ["#D83C3C", "#5A5A5A", "#3CB43C", "#B98C5A", "#EBC828"]


def load_points(path, max_per_class=120000):
    las = laspy.read(path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    labels = np.asarray(las.classification, dtype=np.int64)
    rng = np.random.default_rng(0)
    xs, ys, zs, cs = [], [], [], []
    for cid, color in zip(CLASS_IDS, CLASS_COLORS):
        idx = np.flatnonzero(labels == cid)
        if len(idx) == 0:
            continue
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        xs.append(x[idx])
        ys.append(y[idx])
        zs.append(z[idx])
        cs.append(np.full(len(idx), color, dtype=object))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs),
            np.concatenate(cs))


def top_panel(ax, path, title, max_per_class=80000):
    x, y, _, colors = load_points(path, max_per_class=max_per_class)
    ax.scatter(x, y, c=colors, s=0.12, linewidths=0, alpha=0.75,
               rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
        spine.set_linewidth(0.8)


def add_legend(fig):
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
               markersize=8, label=n)
        for n, c in zip(CLASS_NAMES, CLASS_COLORS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.01))


def render_overview(items, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, (title, path) in zip(axes.ravel(), items):
        top_panel(ax, path, title)
    add_legend(fig)
    fig.suptitle("四块代表性瓦片点云语义标注结果", fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out_path)


def render_3d(path, title, out_path, max_per_class=90000):
    x, y, z, colors = load_points(path, max_per_class=max_per_class)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=colors, s=0.12, linewidths=0, alpha=0.8,
               depthshade=False, rasterized=True)
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect((np.ptp(x), np.ptp(y), max(np.ptp(z) * 1.8, 1.0)))
    ax.set_title(title, fontsize=16, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    add_legend(fig)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out_path)


def main():
    base = r"C:\Users\tongb\Desktop\佟邦本\毕业设计"
    out_dir = os.path.join(base, "第四章_标注截图")
    os.makedirs(out_dir, exist_ok=True)

    items = [
        ("Tile_37", os.path.join(base, "第四章_预分类结果",
                                 "Tile_37_annotation_005m_preclassified.las")),
        ("Tile_27", os.path.join(base, "第四章_多样本训练", "预分类", "Tile_27",
                                 "Tile_27_sample_005m_preclassified.las")),
        ("Tile_53", os.path.join(base, "第四章_多样本训练", "预分类", "Tile_53",
                                 "Tile_53_sample_005m_preclassified.las")),
        ("Tile_74", os.path.join(base, "第四章_多样本训练", "预分类", "Tile_74",
                                 "Tile_74_sample_005m_preclassified.las")),
    ]

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    render_overview(items, os.path.join(out_dir, "四瓦片标注总览.png"))
    render_3d(items[0][1], "Tile_37 点云语义标注三维视图",
              os.path.join(out_dir, "Tile_37_标注三维图.png"))
    render_3d(items[3][1], "Tile_74 点云语义标注三维视图",
              os.path.join(out_dir, "Tile_74_标注三维图.png"))


if __name__ == "__main__":
    main()
