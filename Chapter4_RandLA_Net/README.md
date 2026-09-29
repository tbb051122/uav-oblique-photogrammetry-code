# 第四章：改进 RandLA-Net 点云语义分割与建筑目标提取

本目录代码对应论文第 4 章：

| 论文章节 | 模块 |
| --- | --- |
| 4.2.1 RandLA-Net 基本结构（随机采样 / LFA / 编解码） | `model.py`、`utils.py` |
| 4.2.2 输入特征优化（XYZRGB 六维联合特征） | `data_utils.py`、`model.py` |
| 4.2.3 多尺度局部特征融合模块（K=8/16/32，仅编码器第一层） | `model.py` |
| 4.3.1 数据预处理（异常点剔除/归一化/采样/增强） | `data_utils.py` |
| 4.3.2 评价指标（OA/mAcc/mIoU） | `metrics.py` |
| 4.4 实验训练与测试 | `train.py`、`test.py`、`main.py` |
| 4.2.4 建筑目标提取（语义筛选 + DBSCAN + 边界/高度） | `building_extract.py` |

## 数据格式

目录结构：

```text
data/
  train/  *.npy | *.txt | *.las
  val/    ...
  test/   ...
```

`.npy` / `.txt` 列格式：

```text
x y z r g b label        # 7 列（论文采用 XYZRGB 特征）
x y z label              # 4 列（无颜色）
```

`.las` / `.laz` 使用 laspy 读取（需要 `pip install laspy[lazrs]`），
坐标为米，RGB 通道按 16 位自动转换为 0-255，标签取自 Classification。

默认类别（论文 4.3.1 的 5 类地物 + 未分类）：

```text
0 未分类   1 建筑物   2 道路   3 植被   4 裸地   5 其他人工设施
```

## 使用流程

### 1) 生成合成演示数据（可选）

```bash
python make_demo_data.py --clouds 8 --points 6000
```

### 2) 由真实 LAS 构建训练分块（论文 4.3.1 空间划分）

```bash
python prepare_blocks.py --input scene.las --out-dir data \
    --block-size 100 --voxel 0.1
```

### 3) 训练

```bash
python main.py --mode train --data_dir ./data --model_dir ./models \
    --num_points 8192 --batch_size 4 --num_epochs 100 --device cuda
```

### 4) 测试

```bash
python main.py --mode test --data_dir ./data \
    --checkpoint ./models/best_model.pth --num_points 8192
```

输出测试集的 OA、mAcc、mIoU 与逐类别 IoU（论文 4.3.2）。

## 建筑目标提取（论文 4.2.4）

```bash
python building_extract.py \
    --cloud data/test/scene_000.npy \
    --building-class 1 --eps 2.0 --min-samples 20 \
    --output-dir buildings
```

输出每个建筑目标的二维边界、高度与水平投影面积，对应论文
“建筑目标二维边界范围”与“建筑高度”的计算。

## 说明

- 输入特征融合按论文 4.2.2 实现：坐标归一化到单位尺度、RGB 归一化到
  [0,1]，输入网络为 XYZRGB 六维特征；
- 多尺度局部特征融合按论文 4.2.3 实现：K=8/16/32，各尺度共享 MLP +
  最大池化，拼接后压缩，**仅在编码器第一层使用**；
- 随机采样会改变点序，推理时返回采样索引，便于把预测结果映射回原文件；
- 显存有限时可减小 `--num_points`（如 8192/4096），或用
  `--voxel_size` 在数据端下采样。
