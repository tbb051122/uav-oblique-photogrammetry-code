# -*- coding: utf-8 -*-
"""
改进 RandLA-Net 点云语义分割入口
================================

用法:
    python main.py --mode train --data_dir ./data --model_dir ./models
    python main.py --mode test  --data_dir ./data \
        --checkpoint ./models/best_model.pth

数据目录结构:
    data/
      train/  *.npy|*.txt|*.las
      val/    ...
      test/   ...
"""

import argparse

from data_utils import CLASS_NAMES, NUM_CLASSES_DEFAULT
from train import train
from test import test


def parse_args():
    parser = argparse.ArgumentParser(description="改进 RandLA-Net 点云语义分割")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--checkpoint", type=str,
                        default="./models/best_model.pth")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES_DEFAULT)
    parser.add_argument("--num_points", type=int, default=40960)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    parser.add_argument("--voxel_size", type=float, default=None,
                        help="数据预处理：体素下采样尺寸（米）")
    parser.add_argument("--use_color", action="store_true", default=True)
    parser.add_argument("--no_color", action="store_true")
    parser.add_argument("--no_multi_scale", action="store_true",
                        help="消融：关闭多尺度局部特征融合模块（4.2.6）")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--log_file", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    use_color = not args.no_color
    print("类别:", CLASS_NAMES)
    print("类别数:", args.num_classes)
    if args.mode == "train":
        train(data_dir=args.data_dir, model_dir=args.model_dir,
              num_classes=args.num_classes, num_points=args.num_points,
              batch_size=args.batch_size, num_epochs=args.num_epochs,
              lr=args.lr, use_color=use_color, device=args.device,
              resume=args.resume, voxel_size=args.voxel_size,
              log_file=args.log_file,
              use_multi_scale=not args.no_multi_scale)
    else:
        test(data_dir=args.data_dir, checkpoint_path=args.checkpoint,
             num_classes=args.num_classes, num_points=args.num_points,
             batch_size=args.batch_size, use_color=use_color,
             device=args.device)


if __name__ == "__main__":
    main()
