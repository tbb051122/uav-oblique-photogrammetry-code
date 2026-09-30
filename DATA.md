# 原始数据与获取方式

论文使用的完整数据体积较大，未直接提交到本仓库（GitHub 单文件上限 100 MB、
仓库体积不宜超过数 GB）。本文件说明数据内容、规模、还原方法，以及分卷包
文件名与实际内容之间的对应关系。

## 一、数据清单

| 数据集（目录名） | 内容 | 文件数 | 体积 |
| --- | --- | --- | --- |
| `SURVEY` | 无人机原始航摄影像（大疆 Phantom 4 RTK，FC6310R，含 EXIF-POS 与 XMP 姿态），8 个架次，2779 张 JPG | 2819 | ≈23.1 GB |
| `SURVEY_optimized_T0.005` | 剔除冗余影像后用于三维重建的影像集（T=0.005），1865 张 JPG | 1905 | ≈15.5 GB |
| `SURVEY_剔除影像_T0.005` | 被剔除的冗余影像，799 张 JPG（T=0.005 的筛选共剔除 914 张，该目录为其中较早一次运行输出的副本） | 800 | ≈6.5 GB |
| `chapter4_training_data` | 四块代表性瓦片（Tile_27/37/53/74）5 cm 采样点云、预分类结果、训练样本 | — | ≈2.1 GB |
| `chapter4_results` | 训练/测试结果、预测结果、剔除前后对比点云（npz/las/npy） | — | ≈2.5 GB |
| `chapter4_others` | 消融实验、建筑提取结果、标注截图、实验过程截图 | — | ≈0.5 GB |
| `osgb_before_2779` | 剔除冗余影像前导出的 OSGB 三维模型（111 个切块，含 Data 下各 Tile_* 分块） | 5136 | ≈4.35 GB |
| `osgb_after_1865` | 剔除冗余影像后导出的 OSGB 三维模型（74 个切块，含 Data 下各 Tile_* 分块） | 4769 | ≈3.84 GB |

说明：

- `SURVEY` 是原始航摄影像集（2779 张）；`SURVEY_optimized_T0.005` 是它剔除冗余影像后
  的子集（1865 张），其中每一张都能在 `SURVEY` 中找到同名原图；`SURVEY_剔除影像_T0.005`
  是这一轮筛选中被剔除的影像。
- `SURVEY` 目录中另有 40 个非影像文件（8 个架次 × 5 个：`*_EVENTLOG.bin`、`*_PPKRAW.bin`、
  `*_PPKRAW.sig`、`*_Rinex.obs`、`*_Timestamp.MRK`），因此目录文件数比影像数多 40。
- 影像筛选的完整结果见 `data/image_screening/`：`optimized_image_list.txt`（保留 1865 张）、
  `removed_image_list.txt`（剔除 914 张），以及对应的 2779 张原始影像一轮的清单
  （`*_2779.txt`）与逐影像覆盖度统计 `coverage_report_1865images.csv`、
  `coverage_report_2779images.csv`。
- 原始影像集 `SURVEY`（≈23.1 GB）体积过大，未随分卷上传。

## 二、分卷压缩包

数据打包为分卷归档（tar 顺序切分，每卷 < 1.9 GB），已上传至百度网盘同步空间，
可通过下面的分享链接下载：

> 分享链接：https://pan.baidu.com/s/1zvsGOOs4C9MRJ_jw2kjyoA?pwd=z57x
> 提取码：`z57x`
> 网盘位置：同步空间 → `毕业设计_数据分卷归档`

**分卷文件名的实际内容对照（重要）**：影像类分卷的文件名沿用打包时的早期命名，
与实际内容不一致，请以下表为准。

| 分卷文件 | 实际内容 |
| --- | --- |
| `SURVEY_raw_images.part.001 … .009` | **剔除冗余影像后用于重建的影像集**（T=0.005，1865 张 JPG，≈15.5 GB）。文件名中的 `raw_images` 不是内容含义，卷内目录名为 `SURVEY/` |
| `SURVEY_optimized_T0.005.part.001 … .004` | **被剔除的冗余影像**（799 张 JPG，≈6.5 GB）。卷内目录名为 `SURVEY_剔除影像_T0.005/` |
| `chapter4_training_data.part.001 … .002` | 四瓦片 5 cm 点云、预分类结果、训练样本与模型（≈2.0 GB），卷内目录 `第四章_多样本训练/` |
| `chapter4_results.part.001 … .002` | 训练/测试、预测、剔除前后对比数据（≈2.4 GB），卷内目录 `第四章_训练测试结果/` |
| `chapter4_others_and_figures.part.001` | 消融实验、建筑提取、截图与论文配图（≈0.1 GB），卷内目录 `第四章_消融实验/` |
| `osgb_before_2779.part.001 … .003` | 剔除冗余影像前导出的 OSGB 模型，111 个切块（≈4.35 GB），卷内目录 `OSGB_剔除冗余影像前_111切块/` |
| `osgb_after_1865.part.001 … .003` | 剔除冗余影像后导出的 OSGB 模型，74 个切块（≈3.84 GB），卷内目录 `OSGB_剔除冗余影像后_74切块/` |

全部 24 卷合计约 34.8 GB，各卷按序号顺序合并即可还原。

## 三、还原方法

分卷包为顺序切分的 tar 归档，还原时按顺序合并后解包即可：

```bash
# Linux / macOS：以保留影像集为例
cat SURVEY_raw_images.part.* > SURVEY_1865.tar
tar -xf SURVEY_1865.tar          # 解出 SURVEY/ 目录（1865 张保留影像）

# 被剔除影像集
cat SURVEY_optimized_T0.005.part.* > SURVEY_removed.tar
tar -xf SURVEY_removed.tar       # 解出 SURVEY_剔除影像_T0.005/ 目录（799 张）
```

```bat
:: Windows（命令提示符），按序号顺序拼接
copy /b SURVEY_raw_images.part.001+SURVEY_raw_images.part.002+...+SURVEY_raw_images.part.009 SURVEY_1865.tar
tar -xf SURVEY_1865.tar
```

## 四、数据格式

- 原始影像：JPEG，内嵌 EXIF-GPS、相机参数与 XMP 云台姿态（俯仰、横滚、航向、高程）。
- 点云：LAS 1.2（含 RGB），瓦片 5 cm 体素采样后用于网络训练。
- 语义标签：0 未分类、1 建筑物、2 道路、3 植被、4 裸地、5 其他人工设施。
- POS 示例：`sample_data/pos_demo.csv`（列名与正式 POS 数据一致）。
