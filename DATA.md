# 原始数据与获取方式

论文使用的完整数据体积较大，未直接提交到本仓库（GitHub 单文件上限 100 MB、
仓库体积不宜超过数 GB）。本文件说明数据内容、规模与还原方法。

## 一、数据清单

| 数据集 | 内容 | 文件数 | 体积 |
| --- | --- | --- | --- |
| `SURVEY` | 无人机原始航摄影像（大疆 Phantom 4 RTK，FC6310R，含 EXIF-POS 与 XMP 姿态） | 1905 | ≈15.5 GB |
| `SURVEY_optimized_T0.005` | 剔除冗余影像后用于重建的影像集（T=0.005，1865 幅） | 800 | ≈6.5 GB |
| `chapter4_training_data` | 四块代表性瓦片（Tile_27/37/53/74）5 cm 采样点云、预分类结果、训练样本 | — | ≈2.1 GB |
| `chapter4_results` | 训练/测试结果、预测结果、剔除前后对比点云（npz/las/npy） | — | ≈2.5 GB |
| `chapter4_others` | 消融实验、建筑提取结果、标注截图、实验过程截图 | — | ≈0.5 GB |

说明：`SURVEY` 与 `SURVEY_optimized_T0.005` 中的影像文件一一对应，
`data/image_screening/removed_image_list.txt` 给出了被剔除影像的文件名清单。

## 二、分卷压缩包

完整数据打包为分卷归档（tar 分卷，每卷 < 1.9 GB），已上传至百度网盘同步空间，
可通过下面的分享链接下载：

> 分享链接：https://pan.baidu.com/s/1zvsGOOs4C9MRJ_jw2kjyoA?pwd=z57x
> 提取码：`z57x`
> 网盘位置：同步空间 → `毕业设计_数据分卷归档`

分卷包命名与内容：

```
SURVEY_raw_images.part.001 … .009              无人机原始航摄影像 1905 张（15.5 GB）
SURVEY_optimized_T0.005.part.001 … .004        剔除冗余影像后的影像集 800 张（6.5 GB）
chapter4_training_data.part.001 … .002         四瓦片 5cm 点云、预分类、训练样本与模型（2.0 GB）
chapter4_results.part.001 … .002               训练/测试、预测、剔除前后对比数据（2.4 GB）
chapter4_others_and_figures.part.001           消融、建筑提取、截图与论文配图（0.1 GB）
```

全部 18 卷合计约 26.6 GB，各卷按序号顺序合并即可还原。

## 三、还原方法

分卷包为顺序切分的 tar 归档，还原时按顺序合并即可：

```bash
# Linux / macOS
cat SURVEY_raw.tar.part.* > SURVEY_raw.tar
tar -xf SURVEY_raw.tar

# Windows（命令提示符）
copy /b SURVEY_raw.tar.part.001+SURVEY_raw.tar.part.002 SURVEY_raw.tar
```

## 四、数据格式

- 原始影像：JPEG，内嵌 EXIF-GPS、相机参数与 XMP 云台姿态（俯仰、横滚、航向、高程）。
- 点云：LAS 1.2（含 RGB），瓦片 5 cm 体素采样后用于网络训练。
- 语义标签：0 未分类、1 建筑物、2 道路、3 植被、4 裸地、5 其他人工设施。
- POS 示例：`sample_data/pos_demo.csv`（列名与正式 POS 数据一致）。
