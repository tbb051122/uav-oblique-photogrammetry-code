# 基于影像优化与点云语义分割的无人机倾斜摄影三维场景方法研究

本仓库是上述毕业论文的**配套代码与数据**，包含第三章"基于空间覆盖分析的倾斜摄影低贡献影像筛选方法"和第四章"基于改进 RandLA-Net 的倾斜摄影点云语义分割与建筑目标提取方法"的完整实现、实验数据统计结果与论文插图。

## 一、仓库结构

```
.
├── Chapter3_Image_Filter/      第三章：低贡献影像筛选算法（含合成数据示例）
├── Chapter4_RandLA_Net/        第四章：改进 RandLA-Net 语义分割与建筑目标提取
├── data/                       论文实验的小体积统计结果（可直接对照论文表格）
│   ├── image_screening/        影像筛选结果：覆盖率报告、保留/剔除影像清单
│   ├── pointcloud_comparison/  剔除冗余影像前后点云规模与语义分割指标
│   ├── ablation/               RGB 颜色特征、多尺度融合模块消融结果
│   ├── training/               训练日志、测试与消融评估输出
│   ├── building_extraction/    建筑目标提取结果与参考标签对比
│   └── preclassification/      点云预分类报告、训练样本说明
├── docs/                       数据说明文档
├── figures/                    论文插图（PNG，可直接引用）
├── sample_data/                原始影像抽样示例与 POS 示例（演示数据格式）
├── DATA.md                     完整数据集说明与获取方式
└── requirements.txt            Python 依赖
```

## 二、运行环境

- Python 3.8 及以上（开发环境为 Python 3.13）
- 依赖安装：

```bash
pip install -r requirements.txt
```

主要依赖：`numpy`、`pandas`、`Pillow`、`scipy`、`scikit-learn`、`matplotlib`、`laspy`、`torch`。

## 三、快速开始

### 1. 第三章  影像筛选（无需真实数据即可运行）

```bash
cd Chapter3_Image_Filter
python demo_synthetic.py          # 生成模拟 POS 与影像清单
python main.py --help             # 查看完整参数
```

用真实数据运行时，输入为航摄影像目录 + POS/EXIF 信息，输出为
`coverage_report.csv`、`optimized_image_list.txt`、`removed_image_list.txt`，
与 `data/image_screening/` 中的论文结果同格式。

### 2. 第四章  点云语义分割

```bash
cd Chapter4_RandLA_Net
python make_demo_data.py --clouds 8 --points 6000      # 生成模拟点云
python main.py --mode train --data_dir ./data --model_dir ./models \
    --num_points 8192 --batch_size 2 --num_epochs 100
python main.py --mode test  --data_dir ./data --model_dir ./models
python building_extract.py --input <LAS 点云> --out_dir <输出目录>
```

## 四、论文与代码的对应关系

| 论文章节 | 内容 | 代码位置 |
| --- | --- | --- |
| 3.2 节 | 无人机影像成像模型、地面覆盖范围计算 | `Chapter3_Image_Filter/camera_model.py`、`ray_casting.py` |
| 3.3 节 | 空间网格划分、覆盖贡献度与低贡献影像筛选 | `Chapter3_Image_Filter/image_filter.py`、`area_sample.py`、`main.py` |
| 4.2.2 节 | XYZRGB 六维输入与归一化 | `Chapter4_RandLA_Net/data_utils.py` |
| 4.2.3 节 | 多尺度局部特征融合模块（K=8/16/32） | `Chapter4_RandLA_Net/model.py` |
| 4.2.4 节 | 建筑语义点提取 + DBSCAN 聚类 + 边界/高度统计 | `Chapter4_RandLA_Net/building_extract.py` |
| 4.3 节 | 训练、测试与消融实验 | `Chapter4_RandLA_Net/train.py`、`test.py`、`metrics.py` |
| 4.3.3 节 | 影像优化对点云语义分割的影响分析 | `Chapter4_RandLA_Net/predict_blocks.py`、`prepare_blocks.py` |

论文中的统计数字与 `data/` 目录下的 JSON/CSV 一一对应，例如
`data/pointcloud_comparison/summary.json` 对应表 4-3、表 4-4，
`data/ablation/ablation_metrics.json` 对应表 4-7、表 4-8。

## 五、原始数据

完整原始数据（无人机原始影像 1905 张约 15.5 GB、剔除冗余影像后的影像集 800 张约 6.5 GB、
点云瓦片与训练样本约 4.5 GB）体积超出 GitHub 仓库限制，未直接放入本仓库，
获取方式与文件清单见 [DATA.md](DATA.md)。

`sample_data/` 中给出了原始影像与 POS 文件的抽样示例，可用于了解数据格式。

## 六、说明

- 本仓库代码与数据仅供论文复现与学术交流使用。
- 论文题目：基于影像优化与点云语义分割的无人机倾斜摄影三维场景方法研究。
