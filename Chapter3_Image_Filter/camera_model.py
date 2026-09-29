# -*- coding: utf-8 -*-
"""
无人机影像成像模型与影像地面覆盖范围计算
========================================
对应论文 3.2.1（无人机影像成像模型）与 3.2.2（影像地面覆盖范围计算）。

坐标与角度约定
--------------
* 局部测区坐标系采用 ENU：X 向东、Y 向北、Z 向上，单位为米；
* 相机采用针孔（中心投影）模型。光轴为相机坐标系 +Z，任意像素 (u, v)
  对应的成像射线为：
      d_cam = ((u - cx) / fx, (v - cy) / fy, 1)
* 姿态角（单位：度），用于描述相机主光轴相对测区坐标系的指向：
      kappa  航向角：影像“上方”所指方向的方位角，从正北顺时针起算，0 为正北朝上；
      phi    俯仰角：主光轴自天底（垂直向下）向影像上方方向的倾斜角，0 为垂直下视；
      omega  横滚角：绕主光轴（视轴）的旋转角，0 为水平。
  相机坐标系到测区坐标系的旋转矩阵为
      R = Rz(kappa) @ R_base @ Rx(phi) @ Rz(omega)
  其中 R_base = diag(1, -1, -1) 将“正北朝上、垂直下视”的相机坐标转到 ENU。

实际无人机 POS 若直接记录机身 omega/phi/kappa（摄影测量惯例），可在读入
POS 后通过镜头安装角/机身坐标系换算得到上述角度；本模块只负责几何投影计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 角度工具
# ---------------------------------------------------------------------------
def deg2rad(deg: float) -> float:
    return float(np.deg2rad(deg))


def _rot_x(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def _rot_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def rotation_matrix_from_angles(omega_deg: float,
                                phi_deg: float,
                                kappa_deg: float) -> np.ndarray:
    """
    根据横滚角 omega、俯仰角 phi、航向角 kappa 构建相机→测区旋转矩阵 R。

    返回:
        (3, 3) 旋转矩阵，满足 d_enu = R @ d_cam
    """
    o = deg2rad(omega_deg)
    p = deg2rad(phi_deg)
    k = deg2rad(kappa_deg)
    r_base = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    # 航向角 kappa 为“从正北顺时针起算”（大疆/惯导约定），
    # 而 _rot_z 是逆时针数学角，因此对 kappa 取负。
    return _rot_z(-k) @ r_base @ _rot_x(p) @ _rot_z(o)


# ---------------------------------------------------------------------------
# 相机与影像模型
# ---------------------------------------------------------------------------
@dataclass
class Camera:
    """相机内方位元素（单位：毫米 / 像素）。"""
    focal_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    image_width_px: int
    image_height_px: int
    principal_x_px: Optional[float] = None
    principal_y_px: Optional[float] = None

    def __post_init__(self) -> None:
        if self.principal_x_px is None:
            self.principal_x_px = self.image_width_px / 2.0
        if self.principal_y_px is None:
            self.principal_y_px = self.image_height_px / 2.0
        if self.focal_mm <= 0 or self.sensor_width_mm <= 0 or self.sensor_height_mm <= 0:
            raise ValueError("相机参数必须为正")

    def focal_length_px(self) -> Tuple[float, float]:
        """
        计算 x / y 方向等效焦距（像素）。
        依据像素尺寸与传感器尺寸的对应关系：
            fx = focal_mm * image_width_px / sensor_width_mm
            fy = focal_mm * image_height_px / sensor_height_mm
        """
        fx = self.focal_mm * self.image_width_px / self.sensor_width_mm
        fy = self.focal_mm * self.image_height_px / self.sensor_height_mm
        return fx, fy

    def pixel_size_mm(self) -> Tuple[float, float]:
        return (self.sensor_width_mm / self.image_width_px,
                self.sensor_height_mm / self.image_height_px)


@dataclass
class Image:
    """单幅影像：外方位元素（位置 + 姿态）。"""
    file_path: str
    position: np.ndarray          # 摄影中心 ENU 坐标 [x, y, z]，单位米
    omega_deg: float = 0.0        # 横滚角（度）
    phi_deg: float = 0.0          # 俯仰角（度），0 表示垂直下视
    kappa_deg: float = 0.0        # 航向角（度）

    @property
    def xyz(self) -> np.ndarray:
        return np.asarray(self.position, dtype=np.float64).reshape(3)

    def rotation_matrix(self) -> np.ndarray:
        return rotation_matrix_from_angles(self.omega_deg, self.phi_deg, self.kappa_deg)


# ---------------------------------------------------------------------------
# 地面投影（摄影光线与参考高程面的交点）
# ---------------------------------------------------------------------------
def ground_point_from_pixel(image: Image,
                            camera: Camera,
                            u: float,
                            v: float,
                            z_ref: float = 0.0) -> Optional[Tuple[float, float]]:
    """
    将影像上的像点 (u, v) 沿摄影光线投影至参考高程面 z = z_ref。

    实现论文 3.2.2 的步骤：
    1) 像素坐标经内方位元素转换到相机坐标系，得到光线方向 d_cam；
    2) 经姿态旋转矩阵转换到测区坐标系，得到光线方向 d_enu；
    3) 计算光线与平面 z = z_ref 的交点（射线方程求交）。

    返回:
        地面交点 (x, y)；当光线与参考面不相交（例如朝向天顶）时返回 None
    """
    fx, fy = camera.focal_length_px()
    cx = float(camera.principal_x_px)
    cy = float(camera.principal_y_px)

    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    r = image.rotation_matrix()
    d_enu = r @ d_cam

    zs = image.xyz[2]
    dz = d_enu[2]
    if abs(dz) < 1e-9:
        return None
    t = (z_ref - zs) / dz
    if t < 0 or not np.isfinite(t):
        # 交点在相机后方（光轴指向参考面之上），该像点不能投影到参考面
        return None

    x = image.xyz[0] + t * d_enu[0]
    y = image.xyz[1] + t * d_enu[1]
    return float(x), float(y)


def ground_footprint(image: Image,
                     camera: Camera,
                     z_ref: float = 0.0) -> list:
    """
    计算影像四个角点在地面参考面 z = z_ref 上的投影，返回地面覆盖多边形。

    多边形顶点按 左上、右上、右下、左下 的顺序排列，可直接用于
    射线法包含关系判断与覆盖贡献计算。

    返回:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    抛出:
        ValueError: 当影像四角中存在无法投影到参考面的角点时
    """
    w = int(camera.image_width_px)
    h = int(camera.image_height_px)
    corners_px = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    footprint = []
    for u, v in corners_px:
        xy = ground_point_from_pixel(image, camera, u, v, z_ref)
        if xy is None:
            raise ValueError(
                "影像 {} 存在无法投影到参考高程面 z={} 的角点，"
                "请检查姿态角或参考高程设置".format(image.file_path, z_ref))
        footprint.append(xy)

    # 消除退化的重叠/共线多边形（面积过小视为无效）
    area = polygon_area(footprint)
    if area < 1e-6:
        raise ValueError(
            "影像 {} 的地面覆盖多边形面积过小（{:.4f} m^2），"
            "请检查 POS 与相机参数".format(image.file_path, area))
    return footprint


def polygon_area(polygon) -> float:
    """鞋带公式计算多边形面积（平面坐标）。"""
    p = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    x = p[:, 0]
    y = p[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def footprint_bbox(polygon) -> Tuple[float, float, float, float]:
    """多边形轴对齐包围盒 (xmin, ymin, xmax, ymax)。"""
    p = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    return (float(p[:, 0].min()), float(p[:, 1].min()),
            float(p[:, 0].max()), float(p[:, 1].max()))


# ---------------------------------------------------------------------------
# 经纬度 -> 局部平面坐标（等距近似）
# ---------------------------------------------------------------------------
def lonlat_to_local(lon: float, lat: float,
                    ref_lon: float, ref_lat: float) -> Tuple[float, float]:
    """
    将 WGS-84 经纬度近似转换为以 (ref_lon, ref_lat) 为原点的 ENU 平面坐标。

    测区范围较小时，采用等距圆柱近似：
        x = (lon - ref_lon) * 111320 * cos(ref_lat)
        y = (lat - ref_lat) * 110540
    """
    x = (lon - ref_lon) * 111320.0 * np.cos(deg2rad(ref_lat))
    y = (lat - ref_lat) * 110540.0
    return float(x), float(y)


# ---------------------------------------------------------------------------
# 兼容旧版简单模型（正射近似）
# ---------------------------------------------------------------------------
def calculate_ground_size(height: float, sensor: float, focal: float) -> float:
    """
    垂直下视近似：地面覆盖边长 = 航高 * 传感器边长 / 焦距。

    该函数仅适用于近似水平（phi、omega≈0）的快速估算；
    严格倾斜投影请使用 ground_footprint()。
    """
    if focal <= 0 or height < 0:
        raise ValueError("焦距必须为正且航高不能为负")
    return height * sensor / focal


def image_corner(x: float, y: float, size: float) -> list:
    """
    兼容旧版：以 (x, y) 为中心构造边长为 size 的矩形覆盖范围
    （垂直下视近似，未考虑倾斜与姿态）。
    """
    half = size / 2.0
    return [(x - half, y - half), (x + half, y - half),
            (x + half, y + half), (x - half, y + half)]
