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

完整数据打包为分卷压缩包（每卷 < 2 GB），存放于百度网盘（链接与提取码见下），
也可按需从原始工程目录直接拷贝。

> 网盘链接：`（待补充）`  提取码：`（待补充）`

分卷包命名示例：

```
SURVEY_raw.tar.z01, SURVEY_raw.tar.z02, ...
```

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
