# -*- coding: utf-8 -*-
"""
无人机影像 POS 信息读取
========================
对应论文 2.6.4 步骤 (1)：获取无人机影像及 POS 信息。

支持两种来源：
1. JPEG/TIFF 的 EXIF-GPS 标签（位置信息，通常不含姿态角）；
2. 大疆等无人机 JPG 内嵌的 XMP 元数据（GimbalPitch/Roll/Yaw、绝对/相对
   高程等，位于 APP1 XMP 段）；
3. 外部 POS 文本（CSV），记录影像名、经纬度、高程以及
   横滚 omega / 俯仰 phi / 航向 kappa（度）。

注意：不同无人机平台记录的姿态角定义与参考系可能不同，读入后请根据
飞行平台说明换算到 camera_model 中约定的 omega/phi/kappa。
"""

import csv
import io
import os
import re

try:
    import exifread
except ImportError:  # 未安装时仅 CSV 方式可用
    exifread = None


def _as_values(value):
    """兼容 exifread 3.x / 2.x：统一取标签的值列表。"""
    if hasattr(value, "values"):
        return list(value.values)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _num(value):
    """把 EXIF 数值（int / IFDRational / Ratio / IfdTag）转为 float。"""
    if value is None:
        return None
    vals = _as_values(value)
    if not vals:
        return None
    v = vals[0]
    try:
        return float(v.num) / float(v.den)
    except AttributeError:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None


def dms_to_decimal(dms) -> float:
    """将 EXIF 中的度/分/秒 Rational 值转换为十进制度。"""
    vals = _as_values(dms)
    if len(vals) < 3:
        raise ValueError("GPS 度分秒至少需要 3 个数值: {}".format(vals))
    d = _num(vals[0])
    m = _num(vals[1])
    s = _num(vals[2])
    if None in (d, m, s):
        raise ValueError("GPS 度分秒解析失败: {}".format(vals))
    return d + m / 60.0 + s / 3600.0


def _ratio_value(value) -> float:
    """旧接口：数值标签转 float（兼容新旧 exifread）。"""
    return _num(value)


def get_gps(image_path: str):
    """
    读取影像 EXIF-GPS 位置（经纬度），返回 dict 或 None。
    兼容旧版接口。
    """
    if exifread is None:
        raise RuntimeError("读取 EXIF 需要安装 exifread：pip install exifread")
    info = read_exif_gps(image_path)
    if info is None:
        return None
    return {"latitude": info["lat"], "longitude": info["lon"]}


def _process_exif(image_path=None, data=None):
    """读取并解析 EXIF 标签；data 已给出时避免重复读文件。"""
    if exifread is None:
        raise RuntimeError("读取 EXIF 需要安装 exifread：pip install exifread")
    if data is not None:
        return exifread.process_file(io.BytesIO(data), details=False)
    if image_path is None:
        raise ValueError("需要 image_path 或 data")
    with open(image_path, "rb") as f:
        return exifread.process_file(f, details=False)


def _gps_from_tags(tags):
    """从已解析的 EXIF 标签中提取 GPS 信息。"""
    lat = tags.get("GPS GPSLatitude")
    lon = tags.get("GPS GPSLongitude")
    if not lat or not lon:
        return None

    latitude = dms_to_decimal(lat)
    longitude = dms_to_decimal(lon)
    if str(tags.get("GPS GPSLatitudeRef", "")).strip().upper() == "S":
        latitude = -latitude
    if str(tags.get("GPS GPSLongitudeRef", "")).strip().upper() == "W":
        longitude = -longitude

    info = {"lon": longitude, "lat": latitude}
    alt_tag = tags.get("GPS GPSAltitude")
    if alt_tag:
        alt_ref = str(tags.get("GPS GPSAltitudeRef", "")).strip()
        alt = _ratio_value(alt_tag)
        if alt_ref == "1":  # 海平面以下
            alt = -alt
        info["alt"] = alt
    return info


def read_exif_gps(image_path: str, data=None, tags=None):
    """
    读取影像 EXIF-GPS 信息。

    返回:
        {"lon": 度, "lat": 度, "alt": 米(可选)} 或 None（无 GPS 标签）
    """
    if tags is None:
        tags = _process_exif(image_path, data)
    return _gps_from_tags(tags)


def _camera_from_tags(tags):
    """从已解析的 EXIF 标签中提取相机内方位信息。"""
    def first_tag(*candidates):
        for name in candidates:
            if name in tags:
                return tags[name]
        return None

    width = _num(first_tag("EXIF ExifImageWidth", "Image ExifImageWidth",
                          "EXIF PixelXDimension"))
    height = _num(first_tag("EXIF ExifImageLength", "Image ExifImageLength",
                            "EXIF PixelYDimension"))
    if not width or not height:
        width = _num(first_tag("Image ImageWidth"))
        height = _num(first_tag("Image ImageLength"))

    focal = _num(first_tag("EXIF FocalLength"))
    focal35 = _num(first_tag("EXIF FocalLengthIn35mmFilm"))

    sensor_w = None
    sensor_h = None
    if focal and focal35:
        crop = focal35 / focal
        sensor_w = 36.0 / crop
        sensor_h = 24.0 / crop
    if sensor_w is None and width:
        fpx = _num(first_tag("EXIF FocalPlaneXResolution"))
        fpy = _num(first_tag("EXIF FocalPlaneYResolution"))
        if fpx and height and fpy:
            sensor_w = width / fpx * 25.4
            sensor_h = height / fpy * 25.4

    return {
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "focal_mm": float(focal) if focal else None,
        "focal_35": float(focal35) if focal35 else None,
        "sensor_width_mm": float(sensor_w) if sensor_w else None,
        "sensor_height_mm": float(sensor_h) if sensor_h else None,
    }


def read_camera_exif(image_path: str, data=None, tags=None):
    """
    从 EXIF 读取相机内方位信息。

    返回 dict（缺失项为 None）：
        width / height         影像像素尺寸
        focal_mm               实际焦距（mm）
        focal_35               35mm 等效焦距（mm）
        sensor_width_mm / sensor_height_mm
                               （由 35mm 等效焦距或 FocalPlane 分辨率推算）
    """
    if tags is None:
        tags = _process_exif(image_path, data)
    return _camera_from_tags(tags)


# ---------------------------------------------------------------------------
# 大疆 JPG 内嵌 XMP（APP1）解析
# ---------------------------------------------------------------------------
_XMP_FLOAT_TAGS = {
    # DJI 常用姿态 / 高程标签（不同机型名称略有差异）
    "pitch": ("GimbalPitchDegree", "GimbalPitch"),
    "roll": ("GimbalRollDegree", "GimbalRoll"),
    "gimbal_yaw": ("GimbalYawDegree", "GimbalYaw"),
    "flight_yaw": ("FlightYawDegree", "FlightYaw", "Yaw"),
    "absolute_alt": ("AbsoluteAltitude",),
    "relative_alt": ("RelativeAltitude",),
    "lat": ("Latitude", "GpsLatitude"),
    "lon": ("Longitude", "GpsLongitude"),
}


def _read_xmp_text(image_path=None, data=None):
    """从 JPEG 二进制中截取 <x:xmpmeta>...</x:xmpmeta> 文本段。"""
    if data is None:
        if image_path is None:
            raise ValueError("需要 image_path 或 data")
        with open(image_path, "rb") as f:
            data = f.read()
    start = data.find(b"<x:xmpmeta")
    if start < 0:
        start = data.find(b"<xmpmeta")
    if start < 0:
        return ""
    end = data.find(b"</x:xmpmeta>", start)
    if end < 0:
        end = data.find(b"</xmpmeta>", start)
    if end < 0:
        return ""
    return data[start:end + 16].decode("utf-8", errors="ignore")


def _xmp_float(xmp_text: str, *candidates):
    """在 XMP 文本中查找带命名空间的标签数值（支持元素和属性两种写法）。"""
    for tag in candidates:
        # 写法 1：<ns:Tag>value</ns:Tag>
        element_pattern = re.compile(
            r"<[A-Za-z0-9_\-]+:{}[^>]*>\s*([-+0-9.eE]+)\s*</"
            r"[A-Za-z0-9_\-]+:{}[^>]*>".format(re.escape(tag), re.escape(tag)))
        m = element_pattern.search(xmp_text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        # 写法 2（大疆常见）：<rdf:Description ... ns:Tag="+86.30" ...>
        attr_pattern = re.compile(
            r"\b[A-Za-z0-9_\-]+:{}\s*=\s*[\"']"
            r"([-+0-9.eE]+)[\"']".format(re.escape(tag)))
        m = attr_pattern.search(xmp_text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def parse_dji_xmp_text(xmp_text: str):
    """解析 DJI XMP 文本中的姿态/高程/GPS 数值。"""
    out = {}
    for key, candidates in _XMP_FLOAT_TAGS.items():
        value = _xmp_float(xmp_text, *candidates)
        if value is not None:
            out[key] = value
    return out


def read_dji_xmp(image_path=None, data=None):
    """读取 DJI JPG 的 XMP 姿态与高程信息（无 XMP 时返回空 dict）。"""
    return parse_dji_xmp_text(_read_xmp_text(image_path, data))


# ---------------------------------------------------------------------------
# 综合影像元数据
# ---------------------------------------------------------------------------
def _normalize_yaw(yaw):
    """把航向角归一化到 [0, 360)，0 表示正北。"""
    if yaw is None:
        return None
    return float(yaw % 360.0)


def read_image_metadata(image_path: str, nadir_pitch=-90.0,
                        gimbal_yaw_relative=False):
    """
    读取单张无人机 JPG 中用于覆盖计算的完整元数据。

    大疆姿态约定：
        GimbalPitchDegree = -90 表示垂直下视（nadir_pitch 默认 -90）；
        因此论文约定的俯仰角 phi（0 = 垂直下视） = pitch - nadir_pitch。
        GimbalRollDegree 直接作为 omega。
        相机航向优先取 GimbalYawDegree；若该角度是相对机身的角度，需要把
        gimbal_yaw_absolute 设为 False，由调用方叠加 FlightYawDegree。

    返回 dict：
        lon/lat/alt、omega/phi/kappa（度）、
        camera（read_camera_exif 的结果）、make/model（诊断用）。
    """
    with open(image_path, "rb") as f:
        data = f.read()
    tags = _process_exif(data=data)

    meta = {}
    gps = _gps_from_tags(tags)
    if gps:
        meta["lon"] = gps["lon"]
        meta["lat"] = gps["lat"]
        if "alt" in gps:
            meta["alt"] = gps["alt"]
    camera = _camera_from_tags(tags)
    if camera:
        meta["camera"] = camera

    xmp = parse_dji_xmp_text(_read_xmp_text(data=data))
    if xmp:
        if "lat" in xmp and "lat" not in meta:
            meta["lat"] = xmp["lat"]
        if "lon" in xmp and "lon" not in meta:
            meta["lon"] = xmp["lon"]
        if "alt" not in meta:
            meta["alt"] = (xmp.get("absolute_alt")
                           if xmp.get("absolute_alt") is not None
                           else xmp.get("relative_alt", 0.0))

        pitch = xmp.get("pitch")
        meta["omega"] = xmp.get("roll", 0.0)
        # pitch -> 论文 phi：nadir_pitch（如 -90）对应 0
        meta["phi"] = (pitch - nadir_pitch) if pitch is not None else 0.0
        meta["kappa"] = combine_gimbal_and_flight_yaw(
            xmp.get("gimbal_yaw"), xmp.get("flight_yaw"),
            relative=gimbal_yaw_relative)
        meta["xmp"] = xmp
    else:
        meta.setdefault("omega", 0.0)
        meta.setdefault("phi", 0.0)
        meta.setdefault("kappa", 0.0)

    meta["make"] = str(tags.get("Image Make", ""))
    meta["model"] = str(tags.get("Image Model", ""))
    return meta


_COLUMN_ALIASES = {
    "name": ["name", "file", "image", "filename", "影像名", "文件名", "影像"],
    "lon": ["lon", "longitude", "经度", "x_lon"],
    "lat": ["lat", "latitude", "纬度", "y_lat"],
    "alt": ["alt", "altitude", "height", "z", "高程", "飞行高度"],
    "x": ["x", "east", "easting", "东坐标", "x_local"],
    "y": ["y", "north", "northing", "北坐标", "y_local"],
    "omega": ["omega", "roll", "横滚角", "侧滚角"],
    "phi": ["phi", "pitch", "俯仰角"],
    "kappa": ["kappa", "yaw", "heading", "航向角", "偏航角"],
}


def _normalize_header(name: str) -> str:
    return str(name).strip().replace(" ", "").lower().replace("(", "").replace(")", "")


def _find_column(header) -> str:
    key = _normalize_header(header)
    for field, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if key == _normalize_header(alias):
                return field
    return None


def read_pos_csv(pos_path: str):
    """
    读取 POS CSV 文件。

    支持的列名（中英文均可）：
        name/file/image  影像名（可省略，省略时按文件顺序编号）
        lon/longitude/经度, lat/latitude/纬度, alt/altitude/高程
        x/y（局部平面坐标，米；若给出则优先使用，不再转换经纬度）
        omega/roll/横滚角, phi/pitch/俯仰角, kappa/yaw/heading/航向角（度）

    返回:
        [{"name":..., "lon":..., "lat":..., "alt":...,
          "omega":..., "phi":..., "kappa":...}, ...]
    """
    if not os.path.exists(pos_path):
        raise FileNotFoundError("POS 文件不存在: " + pos_path)
    with open(pos_path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        columns = [_find_column(h) for h in header]
        if not any(columns):
            raise ValueError(
                "无法识别 POS 文件表头 {}，请使用 name/lon/lat/alt/"
                "omega/phi/kappa 等列名".format(header))

        records = []
        for row in reader:
            if not row or not any(c.strip() for c in row):
                continue
            data = {}
            for field, cell in zip(columns, row):
                cell = cell.strip()
                if field is None or cell == "":
                    continue
                try:
                    data[field] = float(cell) if field != "name" else cell
                except ValueError:
                    if field == "name":
                        data[field] = cell
                    else:
                        raise ValueError("POS 文件数值解析失败: {} -> {}"
                                         .format(header[columns.index(field)], cell))
            records.append(data)

    # 默认补齐字段
    for i, rec in enumerate(records):
        rec.setdefault("name", "image_{:06d}".format(i + 1))
        rec.setdefault("omega", 0.0)
        rec.setdefault("phi", 0.0)
        rec.setdefault("kappa", 0.0)
        rec.setdefault("alt", 0.0)
    return records


def combine_gimbal_and_flight_yaw(gimbal_yaw, flight_yaw, relative=False):
    """
    综合相机航向：
    GimbalYawDegree 若为绝对方位角（relative=False）则直接使用；
    若为相对机身角度（relative=True），叠加机身航向 FlightYawDegree。
    """
    if gimbal_yaw is None:
        return _normalize_yaw(flight_yaw)
    if relative and flight_yaw is not None:
        return _normalize_yaw(gimbal_yaw + flight_yaw)
    return _normalize_yaw(gimbal_yaw)
