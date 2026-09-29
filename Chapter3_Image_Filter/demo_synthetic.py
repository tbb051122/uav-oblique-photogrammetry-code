# -*- coding: utf-8 -*-
"""
合成数据演示：在没有真实 POS 数据时，验证第三章影像筛选流程可运行。

生成 3 条航线 x 8 个摄站 + 斜摄影像（共 48 幅），其中部分影像覆盖测区
以外或与相邻影像高度重叠，运行筛选后输出优化影像集合。

用法:
    python demo_synthetic.py
"""

import csv
import os

import numpy as np

from main import parse_args, run_pipeline


def make_demo_pos(out_dir="demo_data"):
    os.makedirs(out_dir, exist_ok=True)
    pos_path = os.path.join(out_dir, "pos.csv")
    rows = []
    img_id = 0

    # 测区范围约 [0, 800] x [0, 800]，航高 120 m
    ref_lon, ref_lat = 120.3000, 31.2000
    for line, y0 in enumerate(np.arange(0, 801, 120)):   # 7 条航线
        for x in np.arange(0, 901, 60):                  # 每线 16 个摄站
            y = y0 + (30 if line % 2 else -30)
            img_id += 1
            rows.append({
                "name": "IMG_{:05d}".format(img_id),
                "lon": ref_lon + x / 111320.0,
                "lat": ref_lat + y / 110540.0,
                "alt": 120.0,
                "omega": 0.0, "phi": 0.0, "kappa": 0.0,
            })
            # 每幅下视影像附带一幅 25 度斜视影像（朝东），形成冗余
            img_id += 1
            rows.append({
                "name": "IMG_{:05d}".format(img_id),
                "lon": ref_lon + (x - 20) / 111320.0,
                "lat": ref_lat + y / 110540.0,
                "alt": 120.0,
                "omega": 0.0, "phi": 25.0, "kappa": 90.0,
            })

    # 加入 6 幅严重偏离测区的影像（低贡献）
    for k in range(6):
        img_id += 1
        rows.append({
            "name": "FAR_{:02d}".format(k),
            "lon": ref_lon + (1200 + 80 * k) / 111320.0,
            "lat": ref_lat + (900 + 100 * k) / 110540.0,
            "alt": 120.0,
            "omega": 0.0, "phi": 0.0, "kappa": 0.0,
        })

    with open(pos_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "lon", "lat", "alt", "omega", "phi", "kappa"])
        writer.writeheader()
        writer.writerows(rows)
    print("已生成合成 POS: {}（{} 幅影像）".format(pos_path, len(rows)))
    return pos_path


if __name__ == "__main__":
    pos_file = make_demo_pos()
    args = parse_args([
        "--images-dir", "demo_data",
        "--pos-file", pos_file,
        "--focal", "35", "--sensor-width", "36", "--sensor-height", "24",
        "--image-width", "8192", "--image-height", "5460",
        "--grid-size", "20", "--ref-elevation", "0",
        "--threshold", "0.08",
        "--output-dir", "demo_data/output",
        "--save-fig",
    ])
    run_pipeline(args)
