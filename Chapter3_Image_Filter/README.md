# 第三章：基于空间覆盖分析的倾斜摄影低贡献影像筛选

本目录代码对应论文第 3 章：

| 论文章节 | 模块 |
| --- | --- |
| 3.2.1 无人机影像成像模型 | `camera_model.py` |
| 3.2.2 影像地面覆盖范围计算 | `camera_model.py` |
| 3.2.3 射线法空间包含关系 | `ray_casting.py` |
| 3.3.1 测区空间网格划分 | `area_sample.py` |
| 3.3.2 影像覆盖贡献计算 | `image_filter.py` |
| 3.3.3 低贡献影像筛选策略 | `image_filter.py` |
| 3.3.4 影像筛选流程 / 3.4 实验 | `main.py` |

## 数据格式

影像筛选依赖无人机 POS 信息（位置 + 姿态角）。支持两种方式：

### 方式 1：DJI JPG 自带全部元数据（无需 pos.csv）

大疆影像的 EXIF 中含有 GPS、相机焦距/影像尺寸，XMP（APP1）中含有
云台俯仰/横滚/偏航与高程。直接把影像文件夹交给程序即可：

```bash
python main.py --images-dir images \
    --ref-elevation 0 --grid-size 20 --threshold 0.10 --save-fig
```

姿态换算约定（`read_exif.py`）：

- 大疆云台俯仰角默认 -90° = 垂直下视，论文的 phi（0°=下视）按
  `phi = GimbalPitchDegree - nadir_pitch` 换算；
- 相机航向默认直接使用 `GimbalYawDegree`；若你的机型该角度是相对机身的，
  加 `--gimbal-yaw-relative`，程序会叠加 `FlightYawDegree`；
- 如果相机不是大疆或下视时俯仰角不是 -90，用 `--nadir-pitch` 覆盖。

程序会自动从第一张完整 EXIF 读取焦距、影像尺寸并推算传感器尺寸；
也可以显式覆盖：`--focal 8.8 --image-width 5280 --image-height 3956`。

### 方式 2：POS CSV

```csv
name,lon,lat,alt,omega,phi,kappa
IMG_0001,120.30123,31.20015,120.0,0.0,0.0,0.0
```

列名兼容中英文（`longitude/lon/经度`、`latitude/lat/纬度`、
`altitude/alt/高程`、`omega/roll/横滚角`、`phi/pitch/俯仰角`、
`kappa/yaw/heading/航向角`）。若 CSV 已经给出局部平面坐标
`x,y,z`（米），会直接使用，不再做经纬度转换。

角度约定与换算见 `camera_model.py` 文件头注释。实际飞行平台记录的
姿态角定义不同时，请先换算到代码约定的 omega / phi / kappa。

## 运行

方式 1（JPG 全部自带，最常用）：

```bash
python main.py --images-dir images --grid-size 20 \
    --ref-elevation 0 --threshold 0.10 --save-fig
```

筛选后直接把低贡献影像移出 `images`，保留高贡献影像供 ContextCapture 使用：

```bash
python main.py --images-dir images --recursive --grid-size 20 \
    --ref-elevation 0 --threshold 0.10 --save-fig \
    --move-removed --removed-dir removed_images
```

执行后：
- `images/` 中只保留高贡献影像，可直接整文件夹或使用 `output/optimized_image_list.txt` 导入 CC；
- 低贡献影像按原目录结构剪切到 `removed_images/`；
- `output/removed_image_list.txt` 记录被剪切的影像清单，方便与原数据对比。

注意：`--move-removed` 是真实剪切操作，正式处理前建议先备份原始影像
（或先不加该参数跑一遍，确认覆盖图后再正式分拣）。

方式 2（外部 POS CSV）：

```bash
python main.py \
  --images-dir images \
  --pos-file pos.csv \
  --focal 35 --sensor-width 36 --sensor-height 24 \
  --image-width 8192 --image-height 5460 \
  --grid-size 20 --ref-elevation 0 \
  --threshold 0.10 --save-fig
```

主要参数：

- `--grid-size`：测区网格尺寸 d（米），对应论文 3.3.1；
- `--ref-elevation`：摄影光线求交的参考地面高程；
- `--alpha/--beta`：综合评分权重，论文取 0.5 / 0.5；
- `--threshold`：评分阈值 T，低于该值的影像被判为低贡献；
- `--allow-gaps`：默认保证优化后测区完整覆盖，如不要求可加该参数；
- `--area`：手动指定测区局部坐标范围 `xmin,ymin,xmax,ymax`；
- `--area-lonlat`：直接按经纬度圈定测区
  `lon_min,lat_min,lon_max,lat_max`，程序会自动换算到局部坐标；
- 都不给时，程序默认由所有影像覆盖范围外扩 50 m 生成测区（仅适合快速试验，
  正式实验应明确圈定真实建模范围）。

## 输出

- `output/optimized_image_list.txt`：保留影像清单（供 ContextCapture 使用）；
- `output/coverage_report.csv`：每幅影像的覆盖网格数、基础覆盖率、
  新增覆盖率、综合评分与保留标记；
- `output/coverage_map.png`：优化前后地面覆盖范围示意图（可选）。

## 无真实数据快速验证

```bash
python demo_synthetic.py
```

该脚本会生成 3 条航线 × 16 摄站的下视影像、对应斜视影像以及若干远离
测区的影像，并自动运行完整筛选流程。
