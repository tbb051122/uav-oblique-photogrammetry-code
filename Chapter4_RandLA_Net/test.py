# -*- coding: utf-8 -*-
"""
模型测试与推理（论文 4.4 实验）

* 数据集整体评估：OA / mAcc / mIoU / 逐类 IoU；
* 单帧文件预测：输出每点类别并保存结果。
"""

import os

import numpy as np
import torch

from data_utils import (CLASS_NAMES, NUM_CLASSES_DEFAULT, get_data_loader,
                        normalize_xyz, random_sample_cloud,
                        read_point_cloud, split_xyz_rgb)
from metrics import compute_metrics
from model import RandLANet


def load_model(checkpoint_path, num_classes=NUM_CLASSES_DEFAULT,
               in_channels=6, num_points=40960, device="cpu",
               use_multi_scale=True):
    """从检查点恢复模型。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RandLANet(
        num_classes=checkpoint.get("num_classes", num_classes),
        in_channels=in_channels, num_points=num_points,
        use_multi_scale=use_multi_scale)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate_loader(model, loader, num_classes=NUM_CLASSES_DEFAULT, device="cuda"):
    model.eval()
    all_preds, all_labels = [], []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        xyz, feats = split_xyz_rgb(features, use_color=features.shape[-1] > 3)
        outputs = model(xyz, feats)
        all_preds.append(outputs.argmax(1).cpu().numpy().reshape(-1))
        all_labels.append(labels.cpu().numpy().reshape(-1))
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return compute_metrics(preds, labels, num_classes)


@torch.no_grad()
def predict_cloud(model, xyz, colors, num_points=40960, use_color=True,
                  device="cuda"):
    """
    对单帧点云执行预测。

    注意：当点数超过 num_points 时只预测随机采样出的子集，返回的索引
    可用来把预测结果映射回原文件。

    返回:
        pred_labels (M,), sample_idx (M,)
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if colors is not None:
        colors = np.clip(np.asarray(colors, dtype=np.float64), 0, 255) / 255.0
    xyz_s, colors_s, idx = random_sample_cloud(xyz, num_points, colors)
    xyz_s, _, _ = normalize_xyz(xyz_s)
    if use_color and colors_s is not None:
        feats = np.concatenate([xyz_s, colors_s.astype(np.float32)], axis=-1)
    else:
        feats = xyz_s.astype(np.float32)
    t_xyz = torch.from_numpy(xyz_s.astype(np.float32)).unsqueeze(0).to(device)
    t_feats = torch.from_numpy(feats).unsqueeze(0).to(device)
    outputs = model(t_xyz, t_feats)
    preds = outputs.argmax(1).squeeze(0).cpu().numpy()
    return preds, idx


def test(data_dir, checkpoint_path, num_classes=NUM_CLASSES_DEFAULT,
         num_points=40960, batch_size=4, use_color=True,
         device="cuda", class_names=None, use_multi_scale=True):
    """测试主函数。"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("检查点不存在: " + checkpoint_path)
    model = load_model(checkpoint_path, num_classes=num_classes,
                       in_channels=6 if use_color else 3,
                       num_points=num_points, device=device,
                       use_multi_scale=use_multi_scale)
    loader = get_data_loader(data_dir=data_dir, batch_size=batch_size,
                             num_points=num_points,
                             num_classes=num_classes, split="test",
                             use_color=use_color, drop_last=False)
    metrics = evaluate_loader(model, loader, num_classes, device)

    names = class_names or CLASS_NAMES
    print("\n===== 测试结果（论文 4.3.2 指标） =====")
    print("总体准确率 OA   : {:.4f}".format(metrics["oa"]))
    print("平均类别精度 mAcc: {:.4f}".format(metrics["macc"]))
    print("平均交并比 mIoU : {:.4f}".format(metrics["miou"]))
    for i, iou in enumerate(metrics["class_iou"]):
        acc = metrics["class_acc"][i]
        label = names[i] if i < len(names) else str(i)
        print("  类别 {:>6}: IoU {:.4f}  精度 {:.4f}".format(
            label, iou, acc))
    return metrics


if __name__ == "__main__":
    test(data_dir="./data",
         checkpoint_path="./models/best_model.pth",
         num_classes=NUM_CLASSES_DEFAULT,
         num_points=4096, batch_size=2, use_color=True, device="cpu")
