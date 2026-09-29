# -*- coding: utf-8 -*-
"""
RandLA-Net 训练器（论文 4.4 实验）

评价指标采用论文 4.3.2 定义的 OA / mAcc / mIoU，并根据验证集 mIoU
保存最优模型。
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from data_utils import (CLASS_NAMES, NUM_CLASSES_DEFAULT, get_data_loader,
                        split_xyz_rgb)
from metrics import compute_metrics
from model import RandLANet


class Trainer:
    def __init__(self, model, device, num_classes=NUM_CLASSES_DEFAULT,
                 lr=0.001, weight_decay=0.0001, ignore_index=None,
                 class_names=None):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=-100 if ignore_index is None else int(ignore_index))
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.epoch = 0
        self.best_miou = 0.0

    def _feed(self, features, labels):
        """前向并返回 logits 与损失。"""
        xyz, feats = split_xyz_rgb(features, use_color=features.shape[-1] > 3)
        outputs = self.model(xyz, feats)
        loss = self.criterion(outputs, labels)
        return outputs, loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        start = time.time()
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs, loss = self._feed(features, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.append(preds.reshape(-1))
            all_labels.append(labels.cpu().numpy().reshape(-1))
            if batch_idx % 10 == 0:
                oa = (preds == labels.cpu().numpy()).mean()
                print("  批次 [{}/{}] 损失 {:.4f} 精度 {:.4f}".format(
                    batch_idx, len(train_loader), loss.item(), oa))

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        metrics = compute_metrics(preds, labels, self.num_classes)
        return total_loss / max(len(train_loader), 1), metrics

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        for features, labels in val_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            outputs, loss = self._feed(features, labels)
            total_loss += loss.item()
            all_preds.append(outputs.argmax(1).cpu().numpy().reshape(-1))
            all_labels.append(labels.cpu().numpy().reshape(-1))
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        metrics = compute_metrics(preds, labels, self.num_classes)
        return total_loss / max(len(val_loader), 1), metrics

    def save_checkpoint(self, path, is_best=False):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_miou": self.best_miou,
            "num_classes": self.num_classes,
        }
        torch.save(state, path)
        if is_best:
            torch.save(state, os.path.join(
                os.path.dirname(path), "best_model.pth"))
            print("已保存最优模型:", os.path.join(os.path.dirname(path), "best_model.pth"))

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.epoch = state.get("epoch", 0)
        self.best_miou = state.get("best_miou", 0.0)
        print("从 {} 恢复训练，epoch = {}".format(path, self.epoch))


def train(data_dir, model_dir, num_classes=NUM_CLASSES_DEFAULT,
          num_points=40960, batch_size=4, num_epochs=100, lr=0.001,
          use_color=True, device="cuda", resume=None, voxel_size=None,
          log_file=None, use_multi_scale=True):
    """训练主函数。"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    in_channels = 6 if use_color else 3
    model = RandLANet(num_classes=num_classes,
                      in_channels=in_channels,
                      num_points=num_points,
                      use_multi_scale=use_multi_scale)

    trainer = Trainer(model=model, device=device, num_classes=num_classes,
                      lr=lr)
    if resume and os.path.exists(resume):
        trainer.load_checkpoint(resume)

    train_loader = get_data_loader(
        data_dir=data_dir, batch_size=batch_size, num_points=num_points,
        num_classes=num_classes, split="train", use_color=use_color,
        voxel_size=voxel_size)
    val_loader = get_data_loader(
        data_dir=data_dir, batch_size=batch_size, num_points=num_points,
        num_classes=num_classes, split="val", use_color=use_color,
        voxel_size=voxel_size, drop_last=False)

    log_f = open(log_file, "w", encoding="utf-8") if log_file else None

    def log(msg):
        print(msg)
        if log_f:
            log_f.write(str(msg) + "\n")
            log_f.flush()

    os.makedirs(model_dir, exist_ok=True)
    log("===== 开始训练改进 RandLA-Net =====")
    log("训练集: {} 帧, 验证集: {} 帧, 点数量: {}, 类别: {}".format(
        len(train_loader.dataset), len(val_loader.dataset),
        num_points, num_classes))

    for epoch in range(trainer.epoch, num_epochs):
        trainer.epoch = epoch
        log("\n=== Epoch {}/{} ===".format(epoch + 1, num_epochs))
        log("学习率: {:.6f}".format(trainer.optimizer.param_groups[0]["lr"]))
        t0 = time.time()
        train_loss, train_metrics = trainer.train_one_epoch(train_loader)
        val_loss, val_metrics = trainer.validate(val_loader)
        trainer.scheduler.step()

        log("训练: 损失 {:.4f}  OA {:.4f}  mAcc {:.4f}  mIoU {:.4f}".format(
            train_loss, train_metrics["oa"], train_metrics["macc"],
            train_metrics["miou"]))
        log("验证: 损失 {:.4f}  OA {:.4f}  mAcc {:.4f}  mIoU {:.4f}  "
            "耗时 {:.1f}s".format(
                val_loss, val_metrics["oa"], val_metrics["macc"],
                val_metrics["miou"], time.time() - t0))
        log("逐类 IoU: " + " ".join(
            "{:.4f}".format(x) for x in val_metrics["class_iou"]))

        is_best = val_metrics["miou"] > trainer.best_miou
        if is_best:
            trainer.best_miou = val_metrics["miou"]
        checkpoint_path = os.path.join(model_dir,
                                       "checkpoint_epoch_{}.pth".format(epoch + 1))
        trainer.save_checkpoint(checkpoint_path, is_best=is_best)
        if epoch > 0 and (epoch + 1) % 10 == 0:
            prev = os.path.join(model_dir,
                                "checkpoint_epoch_{}.pth".format(epoch - 8))
            if os.path.exists(prev):
                os.remove(prev)  # 仅保留最近 10 个周期的检查点

    log("\n训练完成，最优验证 mIoU: {:.4f}".format(trainer.best_miou))
    if log_f:
        log_f.close()


if __name__ == "__main__":
    train(data_dir="./data", model_dir="./models", num_classes=NUM_CLASSES_DEFAULT,
          num_points=4096, batch_size=2, num_epochs=3, lr=0.001,
          use_color=True, device="cpu")
