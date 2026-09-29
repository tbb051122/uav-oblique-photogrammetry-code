# -*- coding: utf-8 -*-
"""
对已切分的点云块做逐点预测，保存预测标签。

用法示例：
python predict_blocks.py \
    --data-dir ./data --split test \
    --checkpoint ./models/best_model.pth \
    --out-dir ./predictions --num-points 8192 --device cuda

输出：
    predictions/pred_<原文件名>.npy
    列为 x y z r g b pred_label
"""

import argparse
import glob
import os

import numpy as np
import torch

from data_utils import normalize_xyz, read_point_cloud
from test import load_model


@torch.no_grad()
def predict_block(model, xyz, colors, num_points=8192, device="cuda"):
    """按固定点数分块推理，返回每个点的预测类别。"""
    n = len(xyz)
    preds = np.zeros(n, dtype=np.int64)
    xyz_n, _, _ = normalize_xyz(xyz)
    rng = np.random.default_rng(0)

    for start in range(0, n, num_points):
        end = min(start + num_points, n)
        x = xyz_n[start:end]
        c = colors[start:end] if colors is not None else None
        m = len(x)
        if m < num_points:
            idx = rng.choice(m, size=num_points, replace=True)
            x = x[idx]
            c = c[idx] if c is not None else None

        if c is not None:
            c_n = np.clip(c.astype(np.float64) / 255.0, 0.0, 1.0)
            feats = np.concatenate([x, c_n], axis=-1).astype(np.float32)
        else:
            feats = x.astype(np.float32)

        t_xyz = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
        t_feats = torch.from_numpy(feats).unsqueeze(0).to(device)
        outputs = model(t_xyz, t_feats)
        pred = outputs.argmax(1).squeeze(0).cpu().numpy()
        preds[start:end] = pred[:m]
    return preds


def main():
    parser = argparse.ArgumentParser(description="分块点云逐点预测")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, num_classes=args.num_classes,
                       in_channels=6, num_points=args.num_points, device=device)
    os.makedirs(args.out_dir, exist_ok=True)

    split_dir = os.path.join(args.data_dir, args.split)
    files = sorted(glob.glob(os.path.join(split_dir, "*.npy")))
    if not files:
        raise SystemExit("没有找到 npy 块: " + split_dir)

    for path in files:
        xyz, colors, _ = read_point_cloud(path)
        pred = predict_block(model, xyz, colors,
                             num_points=args.num_points, device=device)
        out = np.column_stack([xyz, colors, pred]) if colors is not None else \
            np.column_stack([xyz, pred])
        out_path = os.path.join(args.out_dir, "pred_" + os.path.basename(path))
        np.save(out_path, out)
        values, counts = np.unique(pred, return_counts=True)
        print(os.path.basename(path), dict(zip(values.tolist(), counts.tolist())))
        print("saved:", out_path)


if __name__ == "__main__":
    main()
