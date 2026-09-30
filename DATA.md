# 原始数据与获取方式

论文使用的完整数据体积较大，未直接提交到本仓库（GitHub 单文件上限 100 MB、
仓库体积不宜超过数 GB）。本文件说明数据内容、规模、分卷包命名与还原方法。

## 一、数据清单

| 数据集 | 内容 | 文件数 | 体积 |
| --- | --- | --- | --- |
| 原始航摄影像 `SURVEY`（磁盘位置 E:\SURVEY） | 大疆 Phantom 4 RTK（FC6310R）单镜头多架次多航向影像，含 EXIF-POS 与 XMP 姿态，8 个架次，2779 张 JPG | 2819 | ≈23.1 GB |
| 剔除冗余影像后用于重建的影像集（本地目录名 `SURVEY`，即桌面工作副本） | T=0.005 筛选中保留的影像，1865 张 JPG，每一张都能在原始影像集中找到同名原图 | 1905 | ≈15.5 GB |
| 被剔除的冗余影像（本地目录名 `SURVEY_剔除影像_T0.005`） | T=0.005 的筛选共剔除 914 张，该目录是其中较早一次运行输出的副本，799 张 JPG | 800 | ≈6.5 GB |
| `chapter4_training_data` | 四块代表性瓦片（Tile_27/37/53/74）5 cm 采样点云、预分类结果、训练样本 | — | ≈2.1 GB |
| `chapter4_results` | 训练/测试结果、预测结果、剔除前后对比点云（npz/las/npy） | — | ≈2.5 GB |
| `chapter4_others` | 消融实验、建筑提取结果、标注截图、实验过程截图 | — | ≈0.5 GB |
| `osgb_before_2779` | 剔除冗余影像前导出的 OSGB 三维模型（111 个切块，含 Data 下各 Tile_* 分块） | 5136 | ≈4.35 GB |
| `osgb_after_1865` | 剔除冗余影像后导出的 OSGB 三维模型（74 个切块，含 Data 下各 Tile_* 分块） | 4769 | ≈3.84 GB |

说明：

- 论文中「原始影像 2779 张 → 保留 1865 张（减少 914 张，32.89%）」对应上表前两行；
  被剔除影像的完整 914 张清单见 `data/image_screening/removed_image_list.txt`。
- 保留影像集所在目录沿用 `SURVEY` 这个名字，与磁盘上的原始影像集目录同名但内容不同：
  随包/桌面副本中是剔除后的 1865 张，磁盘 E:\SURVEY 下是原始 2779 张。
- `SURVEY` 目录另有 40 个非影像文件（8 个架次 × 5 个：`*_EVENTLOG.bin`、
  `*_PPKRAW.bin`、`*_PPKRAW.sig`、`*_Rinex.obs`、`*_Timestamp.MRK`），因此目录文件数比影像数多 40。
- 影像筛选的完整结果见 `data/image_screening/`：`optimized_image_list.txt`（保留 1865 张）、
  `removed_image_list.txt`（剔除 914 张），以及对应的 2779 张原始影像一轮的清单
  （`*_2779.txt`）与逐影像覆盖度统计 `coverage_report_1865images.csv`、
  `coverage_report_2779images.csv`。
- 原始影像集（≈23.1 GB）体积过大，未随分卷上传。

## 二、分卷压缩包

数据打包为分卷归档（tar 顺序切分，每卷 < 1.9 GB），已上传至百度网盘同步空间，
可通过下面的分享链接下载：

> 分享链接：https://pan.baidu.com/s/1zvsGOOs4C9MRJ_jw2kjyoA?pwd=z57x
> 提取码：`z57x`
> 网盘位置：同步空间 → `毕业设计_数据分卷归档`

分卷文件名与内容一一对应（2026-09 整理时已将影像类分卷改为与内容一致的名字）：

| 分卷文件 | 内容 | 卷内目录 |
| --- | --- | --- |
| `SURVEY_optimized_1865.part.001 … .009` | 剔除冗余影像后用于重建的影像集，1865 张 JPG（≈15.5 GB） | `SURVEY/` |
| `SURVEY_removed_799.part.001 … .004` | 被剔除的冗余影像，799 张 JPG（≈6.5 GB） | `SURVEY_剔除影像_T0.005/` |
| `chapter4_training_data.part.001 … .002` | 四瓦片 5 cm 点云、预分类结果、训练样本与模型（≈2.0 GB） | `第四章_多样本训练/` |
| `chapter4_results.part.001 … .002` | 训练/测试、预测、剔除前后对比数据（≈2.4 GB） | `第四章_训练测试结果/` |
| `chapter4_others_and_figures.part.001` | 消融实验、建筑提取、截图与论文配图（≈0.1 GB） | `第四章_消融实验/` |
| `osgb_before_2779.part.001 … .003` | 剔除冗余影像前导出的 OSGB 模型，111 个切块（≈4.35 GB） | `OSGB_剔除冗余影像前_111切块/` |
| `osgb_after_1865.part.001 … .003` | 剔除冗余影像后导出的 OSGB 模型，74 个切块（≈3.84 GB） | `OSGB_剔除冗余影像后_74切块/` |

全部 24 卷合计约 34.8 GB，各卷按序号顺序合并即可还原。

历史命名提示：早期打包版本中这两组影像卷曾名为 `SURVEY_raw_images.part.001 … .009`（实为保留的
1865 张）与 `SURVEY_optimized_T0.005.part.001 … .004`（实为剔除的 799 张），名称与内容相反；
若你手上有旧名文件，其内容对应关系同上表前两行。

## 三、还原方法

分卷包为顺序切分的 tar 归档，还原时按顺序合并后解包即可：

```bash
# Linux / macOS：保留影像集（1865 张）
cat SURVEY_optimized_1865.part.* > SURVEY_1865.tar
tar -xf SURVEY_1865.tar          # 解出 SURVEY/ 目录

# 被剔除影像集（799 张）
cat SURVEY_removed_799.part.* > SURVEY_removed.tar
tar -xf SURVEY_removed.tar       # 解出 SURVEY_剔除影像_T0.005/ 目录
```

```bat
:: Windows（命令提示符），按序号顺序拼接
copy /b SURVEY_optimized_1865.part.001+SURVEY_optimized_1865.part.002+...+SURVEY_optimized_1865.part.009 SURVEY_1865.tar
tar -xf SURVEY_1865.tar
```

## 四、数据格式

- 原始影像：JPEG，内嵌 EXIF-GPS、相机参数与 XMP 云台姿态（俯仰、横滚、航向、高程）。
- 点云：LAS 1.2（含 RGB），瓦片 5 cm 体素采样后用于网络训练。
- 语义标签：0 未分类、1 建筑物、2 道路、3 植被、4 裸地、5 其他人工设施。
- POS 示例：`sample_data/pos_demo.csv`（列名与正式 POS 数据一致）。
