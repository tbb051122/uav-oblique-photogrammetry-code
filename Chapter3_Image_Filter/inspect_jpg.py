# -*- coding: utf-8 -*-
"""
检查单张大疆 JPG 中可用于覆盖计算的元数据。

用法:
    python inspect_jpg.py DJI_0001.JPG
    python inspect_jpg.py DJI_0001.JPG DJI_0002.JPG

打印：
* EXIF-GPS 经纬度/高程；
* EXIF 相机焦距与影像尺寸；
* XMP 中所有大疆自定义标签原文及解析结果；
* 程序最终换算出的 omega / phi / kappa。
"""

import argparse
import re
import sys

from read_exif import (_read_xmp_text, read_camera_exif, read_dji_xmp,
                       read_exif_gps, read_image_metadata)


def dump_xmp_tags(xmp_text):
    """提取 XMP 中 <ns:Tag>value</ns:Tag> 或 ns:Tag="value" 形式的全部标签。"""
    pairs = re.findall(
        r"<([A-Za-z0-9_.\-]+:[A-Za-z0-9_.\-]+)[^>]*>([^<]+)</"
        r"[A-Za-z0-9_.\-]+:[A-Za-z0-9_.\-]+>", xmp_text)
    # 属性形式（大疆 XMP 实际采用）：<rdf:Description ... ns:Tag="value" ...>
    attrs = re.findall(
        r"\b([A-Za-z0-9_.\-]+:[A-Za-z0-9_.\-]+)\s*=\s*\"([^\"]*)\"",
        xmp_text)
    return pairs + attrs


def inspect(path, nadir_pitch=-90.0):
    print("\n========== {} ==========".format(path))
    gps = read_exif_gps(path)
    print("EXIF-GPS      :", gps)
    camera = read_camera_exif(path)
    print("EXIF 相机参数 :", camera)

    xmp_text = _read_xmp_text(path)
    print("XMP 标签原文  :")
    if xmp_text:
        for tag, value in dump_xmp_tags(xmp_text):
            print("   {} = {}".format(tag, value.strip()))
    else:
        print("   （未找到 XMP 段）")
    xmp = read_dji_xmp(path)
    print("XMP 解析结果  :", xmp)

    meta = read_image_metadata(path, nadir_pitch=nadir_pitch)
    print("换算后 omega/phi/kappa: {:.3f} / {:.3f} / {:.3f}".format(
        meta.get("omega", 0.0), meta.get("phi", 0.0), meta.get("kappa", 0.0)))
    print("厂商/型号     : {} {}".format(meta.get("make", ""),
                                        meta.get("model", "")))


def main():
    parser = argparse.ArgumentParser(description="检查大疆 JPG 元数据")
    parser.add_argument("images", nargs="+", help="一张或多张 JPG")
    parser.add_argument("--nadir-pitch", type=float, default=-90.0,
                        help="垂直下视时的云台俯仰角（默认 -90）")
    args = parser.parse_args()
    for img in args.images:
        inspect(img, args.nadir_pitch)


if __name__ == "__main__":
    main()
