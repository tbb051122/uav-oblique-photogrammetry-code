# -*- coding: utf-8 -*-
"""
基于空间覆盖分析的倾斜摄影低贡献影像筛选（论文第 3 章）

流程（对应论文 2.6.4）：
    1. 获取无人机影像及 POS 信息；
    2. 读取相机内外参数；
    3. 计算每幅影像的地面投影范围；
    4. 建立研究区域空间网格（2.6.1）；
    5. 用射线法判断影像覆盖关系（2.5.3、2.6.1）；
    6. 计算影像覆盖贡献评分（2.6.2）；
    7. 按阈值筛选低贡献影像并保证测区完整覆盖（2.6.3）；
    8. 输出优化影像集合，供 ContextCapture 三维重建使用。

示例:
    python main.py --images-dir images --pos-file pos.csv \
        --focal 35 --sensor-width 36 --sensor-height 24 \
        --image-width 8192 --image-height 5460 --grid-size 20
"""

import argparse
import csv
import os
import shutil
import sys

import numpy as np

from area_sample import create_sample_points
from camera_model import (Camera, Image, footprint_bbox, ground_footprint,
                          lonlat_to_local)
from image_filter import build_coverage_matrix, coverage_score, optimize_image_set
from read_exif import read_image_metadata, read_pos_csv


IMAGE_EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="倾斜摄影低贡献影像筛选（论文第 3 章）")

    # 数据输入
    parser.add_argument("--images-dir", type=str, default="images",
                        help="影像文件夹（使用 EXIF-GPS 时需要）")
    parser.add_argument("--pos-file", type=str, default=None,
                        help="POS CSV 文件；未提供时尝试读取影像 EXIF-GPS")
    parser.add_argument("--recursive", action="store_true",
                        help="递归扫描 images-dir 子目录中的影像")

    # 相机参数（内方位元素）；缺省时自动从 JPG EXIF 读取，读不到再用默认值
    parser.add_argument("--focal", type=float, default=None,
                        help="相机焦距 mm（缺省自动从 EXIF 读取）")
    parser.add_argument("--sensor-width", type=float, default=None,
                        help="传感器宽度 mm（缺省自动推算）")
    parser.add_argument("--sensor-height", type=float, default=None,
                        help="传感器高度 mm（缺省自动推算）")
    parser.add_argument("--image-width", type=int, default=None,
                        help="影像宽度像素（缺省自动从 EXIF 读取）")
    parser.add_argument("--image-height", type=int, default=None,
                        help="影像高度像素（缺省自动从 EXIF 读取）")

    # 大疆 XMP 姿态约定
    parser.add_argument("--nadir-pitch", type=float, default=-90.0,
                        help="大疆云台俯仰角为垂直下视时的数值（默认 -90）")
    parser.add_argument("--gimbal-yaw-relative", action="store_true",
                        help="GimbalYawDegree 为相对机身角度时叠加 FlightYawDegree")

    # 测区与投影参数
    parser.add_argument("--area", type=str, default=None,
                        help="测区范围 'xmin,ymin,xmax,ymax'；默认由影像覆盖范围外扩生成")
    parser.add_argument("--area-lonlat", type=str, default=None,
                        help="按经纬度圈定矩形测区 "
                             "'lon_min,lat_min,lon_max,lat_max'（与影像同一坐标系）")
    parser.add_argument("--area-margin", type=float, default=50.0,
                        help="由影像范围自动生成测区时外扩的边距（米）")
    parser.add_argument("--grid-size", type=float, default=20.0,
                        help="测区网格尺寸 d（米），论文 2.6.1")
    parser.add_argument("--ref-elevation", type=float, default=0.0,
                        help="参考地面高程 z_ref（米），用于摄影光线求交")
    parser.add_argument("--ref-lon", type=float, default=None,
                        help="局部坐标原点经度（默认取 POS 均值）")
    parser.add_argument("--ref-lat", type=float, default=None,
                        help="局部坐标原点纬度（默认取 POS 均值）")

    # 筛选参数（论文 2.6.2 / 2.6.3）
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="覆盖范围权重 alpha（论文默认 0.5）")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="新增贡献权重 beta（论文默认 0.5）")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="贡献评分阈值 T")
    parser.add_argument("--allow-gaps", action="store_true",
                        help="允许优化后测区出现未覆盖网格（默认保证完整覆盖）")

    # 输出
    parser.add_argument("--output-dir", type=str, default="output",
                        help="结果输出目录")
    parser.add_argument("--save-fig", action="store_true",
                        help="保存覆盖范围示意图（需要 matplotlib）")
    parser.add_argument("--move-removed", action="store_true",
                        help="筛选后将低贡献影像移动到 --removed-dir")
    parser.add_argument("--removed-dir", type=str, default="removed_images",
                        help="低贡献影像输出文件夹（配合 --move-removed）")
    return parser.parse_args(argv)


def _image_files_in_dir(images_dir, recursive=False):
    if not os.path.isdir(images_dir):
        return []
    files = []
    if recursive:
        for root, dirs, names in os.walk(images_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in names:
                if f.lower().endswith(IMAGE_EXTS) and not f.startswith("."):
                    files.append(os.path.join(root, f))
    else:
        files = [os.path.join(images_dir, f)
                 for f in os.listdir(images_dir)
                 if f.lower().endswith(IMAGE_EXTS) and not f.startswith(".")]
    return sorted(files)


def _match_pos_to_images(records, images_dir, recursive=False):
    """将 POS 记录与影像文件建立对应关系。"""
    images = _image_files_in_dir(images_dir, recursive=recursive)
    if not images:
        return records
    by_basename = {}
    for img in images:
        stem = os.path.splitext(os.path.basename(img))[0]
        by_basename.setdefault(stem, img)
        by_basename.setdefault(img, img)
    for rec in records:
        rec["file"] = by_basename.get(rec.get("name"),
                                      by_basename.get(os.path.basename(rec.get("name", ""))))
    return records


def _load_records(args):
    """载入 POS 记录（CSV 优先；否则逐张读取 JPG 的 EXIF-GPS + XMP 姿态）。"""
    if args.pos_file:
        records = read_pos_csv(args.pos_file)
        records = _match_pos_to_images(records, args.images_dir,
                                       recursive=args.recursive)
        # 若有 name 但没有匹配到文件，输出列表仍以 POS 名称为准
        for rec in records:
            rec.setdefault("file", rec.get("name", ""))
        return records

    images = _image_files_in_dir(args.images_dir, recursive=args.recursive)
    if not images:
        raise FileNotFoundError(
            "--pos-file 未提供且 {} 中没有找到影像文件".format(args.images_dir))
    records = []
    missing_gps = []
    for img in images:
        meta = read_image_metadata(img, nadir_pitch=args.nadir_pitch,
                                   gimbal_yaw_relative=args.gimbal_yaw_relative)
        if "lon" not in meta or "lat" not in meta:
            missing_gps.append(img)
            continue
        records.append({
            "name": os.path.splitext(os.path.basename(img))[0],
            "file": img,
            "lon": meta["lon"],
            "lat": meta["lat"],
            "alt": meta.get("alt", 0.0),
            "omega": meta.get("omega", 0.0),
            "phi": meta.get("phi", 0.0),
            "kappa": meta.get("kappa", 0.0),
            "camera": meta.get("camera"),
            "make": meta.get("make", ""),
            "model": meta.get("model", ""),
        })
    if missing_gps:
        print("警告：以下影像没有 EXIF-GPS 信息，已跳过：")
        for p in missing_gps[:20]:
            print("   ", p)
    if not records:
        raise RuntimeError("没有可用的 POS 信息，请提供 --pos-file 或带 GPS 的影像")
    return records


def _resolve_camera(records, args):
    """
    确定相机内方位元素：
    1. 命令行显式给出全部参数时直接使用；
    2. 否则从第一张含 EXIF 相机信息的 JPG 自动读取；
    3. 均不可用时使用典型默认值并在终端提示。
    """
    cli_vals = [args.focal, args.sensor_width, args.sensor_height,
                args.image_width, args.image_height]
    if any(v is not None for v in cli_vals):
        if not all(v is not None for v in cli_vals):
            raise ValueError(
                "--focal/--sensor-width/--sensor-height/--image-width/"
                "--image-height 必须同时提供，或全部省略自动从 EXIF 读取")
        camera = Camera(focal_mm=args.focal, sensor_width_mm=args.sensor_width,
                        sensor_height_mm=args.sensor_height,
                        image_width_px=args.image_width,
                        image_height_px=args.image_height)
        print("使用命令行相机参数: 焦距 {} mm, 传感器 {:.2f}x{:.2f} mm, "
              "影像 {}x{} px".format(camera.focal_mm, camera.sensor_width_mm,
                                     camera.sensor_height_mm,
                                     camera.image_width_px,
                                     camera.image_height_px))
        return camera

    for rec in records:
        cam = rec.get("camera") or {}
        if (cam.get("focal_mm") and cam.get("width") and cam.get("height")):
            sensor_w = (cam.get("sensor_width_mm")
                        or _DEFAULT_SENSOR_W)
            sensor_h = (cam.get("sensor_height_mm")
                        or _DEFAULT_SENSOR_H)
            camera = Camera(
                focal_mm=cam["focal_mm"],
                sensor_width_mm=sensor_w,
                sensor_height_mm=sensor_h,
                image_width_px=cam["width"],
                image_height_px=cam["height"])
            print("已从 {} 的 EXIF 自动读取相机参数: 焦距 {:.2f} mm, "
                  "传感器 {:.2f}x{:.2f} mm, 影像 {}x{} px".format(
                      rec.get("file"), camera.focal_mm,
                      camera.sensor_width_mm, camera.sensor_height_mm,
                      camera.image_width_px, camera.image_height_px))
            return camera

    camera = Camera(focal_mm=_DEFAULT_FOCAL,
                    sensor_width_mm=_DEFAULT_SENSOR_W,
                    sensor_height_mm=_DEFAULT_SENSOR_H,
                    image_width_px=_DEFAULT_W,
                    image_height_px=_DEFAULT_H)
    print("未找到完整 EXIF 相机信息，使用默认参数: 焦距 {} mm, "
          "传感器 {}x{} mm, 影像 {}x{} px（如不合适请用命令行参数覆盖）".format(
              camera.focal_mm, camera.sensor_width_mm,
              camera.sensor_height_mm, camera.image_width_px,
              camera.image_height_px))
    return camera


_DEFAULT_FOCAL = 35.0
_DEFAULT_SENSOR_W = 36.0
_DEFAULT_SENSOR_H = 24.0
_DEFAULT_W = 8192
_DEFAULT_H = 5460


def _to_local(records, args):
    """
    将 POS 记录转换为局部 ENU 坐标。
    若 CSV 含 x/y（局部平面坐标）则直接使用；否则由经纬度转换。
    """
    if records and "x" in records[0] and "y" in records[0]:
        for rec in records:
            rec["pos"] = np.array([rec["x"], rec["y"], rec.get("alt", 0.0)])
        return

    lons = [r["lon"] for r in records if "lon" in r]
    lats = [r["lat"] for r in records if "lat" in r]
    if not lons or not lats:
        raise ValueError("POS 记录缺少经纬度或局部坐标 x/y")
    ref_lon = args.ref_lon if args.ref_lon is not None else float(np.mean(lons))
    ref_lat = args.ref_lat if args.ref_lat is not None else float(np.mean(lats))
    print("局部坐标原点: lon={:.7f}, lat={:.7f}".format(ref_lon, ref_lat))
    for rec in records:
        x, y = lonlat_to_local(rec["lon"], rec["lat"], ref_lon, ref_lat)
        rec["pos"] = np.array([x, y, rec.get("alt", 0.0)])


def _study_area(records, footprints, args):
    """确定测区范围与网格采样点。"""
    if args.area_lonlat:
        try:
            lon1, lat1, lon2, lat2 = (
                float(v) for v in args.area_lonlat.split(","))
        except ValueError:
            raise ValueError(
                "--area-lonlat 格式应为 'lon_min,lat_min,lon_max,lat_max'")
        lons = [r["lon"] for r in records if "lon" in r]
        lats = [r["lat"] for r in records if "lat" in r]
        ref_lon = args.ref_lon if args.ref_lon is not None else float(np.mean(lons))
        ref_lat = args.ref_lat if args.ref_lat is not None else float(np.mean(lats))
        xmin, ymin = lonlat_to_local(lon1, lat1, ref_lon, ref_lat)
        xmax, ymax = lonlat_to_local(lon2, lat2, ref_lon, ref_lat)
        if xmax < xmin or ymax < ymin:
            xmin, xmax = min(xmin, xmax), max(xmin, xmax)
            ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        print("由经纬度圈定的测区已换算为局部坐标: "
              "x[{:.1f},{:.1f}], y[{:.1f},{:.1f}]".format(
                  xmin, xmax, ymin, ymax))
        if xmax == xmin or ymax == ymin:
            raise ValueError("经纬度测区退化为一条线，请检查范围")
    else:
        if args.area:
            try:
                xmin, ymin, xmax, ymax = (
                    float(v) for v in args.area.split(","))
            except ValueError:
                raise ValueError("--area 格式应为 'xmin,ymin,xmax,ymax'")
        else:
            boxes = [footprint_bbox(p) for p in footprints]
            xmin = min(b[0] for b in boxes) - args.area_margin
            ymin = min(b[1] for b in boxes) - args.area_margin
            xmax = max(b[2] for b in boxes) + args.area_margin
            ymax = max(b[3] for b in boxes) + args.area_margin
    print("测区范围: x[{:.1f}, {:.1f}], y[{:.1f}, {:.1f}]".format(
        xmin, xmax, ymin, ymax))
    sample_points = create_sample_points(xmin, xmax, ymin, ymax, args.grid_size)
    print("网格采样点数量: {}".format(len(sample_points)))
    return xmin, ymin, xmax, ymax, sample_points


def _compute_footprints(records, camera, args):
    """逐幅影像计算地面覆盖多边形（论文 2.5.2）。"""
    footprints = []
    valid_records = []
    invalid = 0
    for rec in records:
        img = Image(file_path=rec.get("file") or rec["name"],
                    position=rec["pos"],
                    omega_deg=float(rec.get("omega", 0.0)),
                    phi_deg=float(rec.get("phi", 0.0)),
                    kappa_deg=float(rec.get("kappa", 0.0)))
        try:
            poly = ground_footprint(img, camera, z_ref=args.ref_elevation)
        except ValueError as e:
            print("警告: {}".format(e))
            invalid += 1
            continue
        footprints.append(poly)
        rec["footprint"] = poly
        valid_records.append(rec)
    print("有效影像数量: {}（无法投影到参考面: {}）".format(
        len(valid_records), invalid))
    if not footprints:
        raise RuntimeError("没有可用的影像地面覆盖范围")
    return valid_records, footprints


def _save_results(records, keep_flags, report, args, bounds):
    """保存优化影像集合与筛选报告。"""
    os.makedirs(args.output_dir, exist_ok=True)

    # 影像列表（ContextCapture 可直接使用的输入清单）
    list_path = os.path.join(args.output_dir, "optimized_image_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for rec, keep in zip(records, keep_flags):
            if keep:
                path = rec.get("file")
                if path and os.path.exists(path):
                    f.write("{}\n".format(os.path.abspath(path)))
                else:
                    f.write("{}\n".format(rec.get("file") or rec["name"]))

    removed_list_path = os.path.join(args.output_dir, "removed_image_list.txt")
    with open(removed_list_path, "w", encoding="utf-8") as f:
        for rec, keep in zip(records, keep_flags):
            if not keep:
                path = rec.get("file")
                if path and os.path.exists(path):
                    f.write("{}\n".format(os.path.abspath(path)))
                else:
                    f.write("{}\n".format(rec.get("file") or rec["name"]))

    # 覆盖贡献报告
    csv_path = os.path.join(args.output_dir, "coverage_report.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["影像名称", "覆盖网格数", "网格总数", "基础覆盖率",
                         "新增覆盖率", "综合评分", "是否保留"])
        for i in range(len(records)):
            name = records[i].get("file") or records[i]["name"]
            writer.writerow([
                name,
                report["covered_grids"][i],
                report["total_grids"][i],
                "{:.6f}".format(report["base_ratio"][i]),
                "{:.6f}".format(report["new_ratio"][i]),
                "{:.6f}".format(report["score"][i]),
                "是" if keep_flags[i] else "否",
            ])

    print("\n优化影像清单已保存: {}".format(list_path))
    print("低贡献影像清单已保存: {}".format(removed_list_path))
    print("覆盖贡献报告已保存: {}".format(csv_path))


def _relocate_removed(records, keep_flags, args):
    """
    将低贡献影像从原目录剪切到 removed_dir。

    - images_dir 中的保留影像保持原位；
    - 低贡献影像按原相对路径结构移动到 removed_dir，
      避免五镜头等子目录中同名文件互相覆盖；
    - 移动前检查目标位置，不覆盖已有文件。
    """
    images_dir = args.images_dir
    removed_dir = args.removed_dir
    os.makedirs(removed_dir, exist_ok=True)
    base = os.path.abspath(images_dir)
    moved = 0
    moved_list = []
    skipped = []
    for rec, keep in zip(records, keep_flags):
        if keep:
            continue
        src = rec.get("file")
        if not src or not os.path.exists(src):
            continue
        src_abs = os.path.abspath(src)
        try:
            inside = os.path.commonpath([base, src_abs]) == base
        except ValueError:
            inside = False
        if not inside:
            # 影像不在 images_dir 内（例如外部 POS 路径），直接复制式移动即可
            rel = os.path.basename(src_abs)
        else:
            rel = os.path.relpath(src_abs, base)
        dst = os.path.join(removed_dir, rel)
        if os.path.exists(dst):
            skipped.append(src_abs)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src_abs, dst)
        moved += 1
        moved_list.append(os.path.abspath(dst))
    if skipped:
        print("警告：以下影像目标位置已存在，未移动（请人工处理）：")
        for s in skipped:
            print("   ", s)
    print("已剪切 {} 幅低贡献影像到 {}".format(moved, removed_dir))
    if moved_list:
        list_path = os.path.join(removed_dir, "removed_image_list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(moved_list) + "\n")
        print("剔除影像目标清单: {}".format(list_path))
    return moved


def _print_summary(records, keep_flags):
    n_all = len(records)
    n_keep = int(keep_flags.sum())
    n_drop = n_all - n_keep
    ratio = n_drop / n_all * 100.0 if n_all else 0.0
    print("\n===== 影像筛选结果（论文 4.1.2 指标） =====")
    print("原始影像数量 N = {}".format(n_all))
    print("优化后影像数量 M = {}".format(n_keep))
    print("减少数量 N - M = {}".format(n_drop))
    print("减少比例 (N-M)/N*100% = {:.2f}%".format(ratio))


def _plot(records, footprints, keep_flags, bounds, out_path):
    """绘制优化前后影像覆盖范围（可选）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        font_names = {f.name for f in font_manager.fontManager.ttflist}
        for font_name in ("Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"):
            if font_name in font_names:
                plt.rcParams["font.sans-serif"] = [font_name]
                break
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("未安装 matplotlib，跳过示意图输出")
        return
    xmin, ymin, xmax, ymax = bounds
    fig, ax = plt.subplots(figsize=(9, 9))
    for poly, keep in zip(footprints, keep_flags):
        px = [p[0] for p in poly] + [poly[0][0]]
        py = [p[1] for p in poly] + [poly[0][1]]
        color = "#2e86de" if keep else "#d63031"
        ax.plot(px, py, color=color, lw=0.6, alpha=0.7)
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, edgecolor="black", lw=1.5))
    ax.set_aspect("equal")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("影像地面覆盖范围（蓝=保留, 红=剔除）")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("覆盖范围示意图已保存: {}".format(out_path))


def run_pipeline(args):
    """执行完整筛选流程，返回 (保留标记, 影像记录)。"""
    records = _load_records(args)
    camera = _resolve_camera(records, args)
    _to_local(records, args)
    records, footprints = _compute_footprints(records, camera, args)

    bounds_xy = _study_area(records, footprints, args)
    _, _, _, _, sample_points = bounds_xy
    cover = build_coverage_matrix(footprints, sample_points)
    print("覆盖关系矩阵: 影像 {} x 网格 {}".format(*cover.shape))

    names = [r.get("file") or r["name"] for r in records]
    keep_flags, report = optimize_image_set(
        cover,
        names=names,
        alpha=args.alpha,
        beta=args.beta,
        threshold=args.threshold,
        ensure_full_coverage=not args.allow_gaps,
    )
    _save_results(records, keep_flags, report, args, bounds_xy[:4])
    _print_summary(records, keep_flags)
    if args.move_removed:
        _relocate_removed(records, keep_flags, args)
    if args.save_fig:
        _plot(records, footprints, keep_flags, bounds_xy[:4],
              os.path.join(args.output_dir, "coverage_map.png"))
    return keep_flags, records


def main(argv=None):
    args = parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
