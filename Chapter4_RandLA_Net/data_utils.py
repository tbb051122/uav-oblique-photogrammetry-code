# -*- coding: utf-8 -*-
"""
点云数据读取、预处理与数据集类
================================
对应论文 4.3.1 数据预处理流程：
1. 异常点剔除（统计离群点）；
2. 坐标归一化（统一到零均值单位尺度，保证训练稳定）；
3. 点云采样（调整到网络输入数量 num_points）；
4. 数据增强（随机旋转 / 缩放 / 抖动等）。

支持数据格式：
* .npy / .txt：列为 x,y,z,r,g,b,label（可缺省部分列）；
* .las / .laz：需要安装 laspy（laspy[lazrs]）。

论文将 XYZ 与 RGB 融合为 XYZRGB 六维联合特征（4.2.2），
其中颜色通道归一到 [0, 1]，坐标归一到零均值单位尺度。
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


CLASS_NAMES = [
    "未分类",     # 0
    "建筑物",     # 1
    "道路",       # 2
    "植被",       # 3
    "裸地",       # 4
    "其他人工设施",  # 5
]
NUM_CLASSES_DEFAULT = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# 点云读取
# ---------------------------------------------------------------------------
def _read_las(path):
    """使用 laspy 读取 LAS/LAZ 点云（坐标已按 scale/offset 转为米）。"""
    try:
        import laspy
    except ImportError:
        raise RuntimeError(
            "读取 LAS 需要安装 laspy：pip install laspy[lazrs]")
    las = laspy.read(path)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    colors = None
    if all(hasattr(las, attr) for attr in ("red", "green", "blue")):
        rgb = np.column_stack([las.red, las.green, las.blue]).astype(np.float64)
        if rgb.max() > 255:      # 16-bit LAS 颜色
            rgb = rgb / 257.0
        colors = np.clip(rgb, 0, 255)
    labels = None
    if hasattr(las, "classification"):
        labels = np.asarray(las.classification, dtype=np.int64)
    return xyz, colors, labels


def read_point_cloud(path):
    """
    读取点云文件。

    返回:
        (xyz, colors, labels)
        xyz    (N,3) float
        colors (N,3) float，值域 0-255；无颜色时为 None
        labels (N,)  int；无标签时为 None
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".las", ".laz"):
        return _read_las(path)
    if ext == ".npy":
        data = np.load(path)
    elif ext in (".txt", ".csv"):
        data = np.loadtxt(path)
    else:
        raise ValueError("不支持的点云格式: {}".format(ext))
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("点云文件必须为二维数组，当前形状 {}".format(data.shape))

    xyz = data[:, :3]
    colors = None
    labels = None
    if data.shape[1] >= 6:
        colors = np.clip(data[:, 3:6], 0, 255)
    if data.shape[1] >= 7:
        labels = data[:, 6].astype(np.int64)
    elif data.shape[1] == 4:
        labels = data[:, 3].astype(np.int64)
    return xyz, colors, labels


# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------
def remove_statistical_outliers(xyz, k=16, std_ratio=2.0, colors=None, labels=None):
    """
    统计离群点剔除（论文 4.3.1 步骤 1）：
    计算每个点与其 k 个近邻的平均距离，距离超过全局均值 ± std_ratio*标准差
    的点视为异常点。
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise RuntimeError("离群点剔除需要 scipy：pip install scipy")
    n = len(xyz)
    k = min(k, n - 1)
    if k <= 0:
        return xyz, colors, labels
    tree = cKDTree(xyz)
    dist, _ = tree.query(xyz, k=k + 1)
    mean_dist = dist[:, 1:].mean(axis=1)
    threshold = mean_dist.mean() + std_ratio * mean_dist.std()
    keep = mean_dist <= threshold
    if not keep.all():
        xyz = xyz[keep]
        if colors is not None:
            colors = colors[keep]
        if labels is not None:
            labels = labels[keep]
    return xyz, colors, labels


def voxel_downsample(xyz, voxel_size=0.5, colors=None, labels=None):
    """
    体素下采样：每个体素内保留所有点的均值，用于降低稠密点云规模。
    """
    if voxel_size is None or voxel_size <= 0:
        return xyz, colors, labels
    keys = np.floor(xyz / voxel_size)
    # 用结构化的 (x,y,z) 键构造唯一体素
    kx = (keys[:, 0] - keys[:, 0].min()).astype(np.int64)
    ky = (keys[:, 1] - keys[:, 1].min()).astype(np.int64)
    kz = (keys[:, 2] - keys[:, 2].min()).astype(np.int64)
    widths = (kx.max() + 1, ky.max() + 1, kz.max() + 1)
    flat = kx * (widths[1] * widths[2]) + ky * widths[2] + kz
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    split = np.flatnonzero(np.diff(flat_sorted) != 0) + 1
    groups = np.split(order, split)

    out = []
    out_c = []
    out_l = []
    for g in groups:
        out.append(xyz[g].mean(axis=0))
        if colors is not None:
            out_c.append(colors[g].mean(axis=0))
        if labels is not None:
            vals, counts = np.unique(labels[g], return_counts=True)
            out_l.append(vals[np.argmax(counts)])
    return (np.asarray(out),
            np.asarray(out_c) if colors is not None else None,
            np.asarray(out_l, dtype=np.int64) if labels is not None else None)


def normalize_xyz(xyz, centroid=None, scale=None):
    """
    坐标归一化（论文 4.3.1 步骤 2 / 4.2.2）：
        坐标减去质心，再除以最大点到质心距离，使点云落在单位球内。
    返回归一化坐标及可逆参数。
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if centroid is None:
        centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    if scale is None:
        scale = float(np.sqrt((centered ** 2).sum(axis=1)).max())
    if scale and scale > 0:
        centered = centered / scale
    return centered, np.asarray(centroid, dtype=np.float64), scale


def random_sample_cloud(xyz, n_target, colors=None, labels=None, rng=None):
    """随机采样到固定点数量（不足时有放回采样）。"""
    n = len(xyz)
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(n, size=n_target, replace=(n < n_target))
    return xyz[idx], colors[idx] if colors is not None else None, \
        labels[idx] if labels is not None else None


def augment_cloud(xyz, colors=None):
    """
    数据增强（论文 4.3.1 步骤 4）：绕竖直轴随机旋转、随机缩放与抖动。
    colors 通道不变（旋转不影响 RGB）。
    """
    rng = np.random.default_rng()
    angle = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    xyz = xyz @ r
    scale = rng.uniform(0.9, 1.1)
    xyz = xyz * scale
    xyz = xyz + rng.normal(0.0, 0.01, size=xyz.shape)
    if colors is not None:
        colors = colors + rng.normal(0.0, 0.01, size=colors.shape)
        colors = np.clip(colors, 0.0, 1.0)
    return xyz, colors


def cloud_to_features(xyz, colors, use_color=True):
    """
    构造网络输入特征（论文 4.2.2）：
    use_color=True  -> [x,y,z,r,g,b]（XYZRGB 六维联合特征）
    use_color=False -> [x,y,z]
    颜色已归一到 [0,1]（调用方保证）。
    """
    if use_color and colors is not None:
        return np.concatenate([xyz, colors], axis=-1).astype(np.float32)
    return xyz.astype(np.float32)


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
class PointCloudDataset(Dataset):
    """
    点云语义分割数据集。

    目录结构：
        data_dir/
          train/  001.npy ...
          val/    ...
          test/   ...
    每帧为一个 .npy/.txt/.las 场景（或已切分的空间块），采样到 num_points。
    """
    def __init__(self, data_dir, num_points=40960, num_classes=NUM_CLASSES_DEFAULT,
                 split="train", use_color=True, augment=False,
                 remove_outliers=True, outlier_k=16, voxel_size=None,
                 seed=None):
        super().__init__()
        self.num_points = num_points
        self.num_classes = num_classes
        self.split = split
        self.use_color = use_color
        self.augment = augment and split == "train"
        self.remove_outliers = remove_outliers
        self.outlier_k = outlier_k
        self.voxel_size = voxel_size
        # seed 不为空时，每个样本的随机采样点固定，便于多次评估/消融对比
        self.seed = seed
        self.file_list = self._file_list(data_dir, split)

    def _file_list(self, data_dir, split):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            split_dir = data_dir
        files = []
        for f in os.listdir(split_dir):
            if f.lower().endswith((".npy", ".txt", ".las", ".laz")):
                files.append(os.path.join(split_dir, f))
        return sorted(files)

    def __len__(self):
        return len(self.file_list)

    def _load(self, idx):
        path = self.file_list[idx]
        xyz, colors, labels = read_point_cloud(path)
        if labels is None:
            raise ValueError("文件 {} 不包含标签（需要第 4 列或第 7 列）".format(path))
        return xyz, colors, labels

    def __getitem__(self, idx):
        xyz, colors, labels = self._load(idx)

        if self.voxel_size:
            xyz, colors, labels = voxel_downsample(
                xyz, self.voxel_size, colors, labels)
        if self.remove_outliers:
            xyz, colors, labels = remove_statistical_outliers(
                xyz, k=self.outlier_k, colors=colors, labels=labels)
        rng = (None if self.seed is None
               else np.random.default_rng(self.seed + idx))
        xyz, colors, labels = random_sample_cloud(
            xyz, self.num_points, colors, labels, rng=rng)

        # 坐标归一化：零均值、单位尺度
        xyz_n, _, _ = normalize_xyz(xyz)
        if colors is not None:
            colors_n = (np.asarray(colors, dtype=np.float64) / 255.0).astype(np.float32)
            colors_n = np.clip(colors_n, 0.0, 1.0)
        else:
            colors_n = None

        if self.augment:
            xyz_n, colors_n = augment_cloud(xyz_n, colors_n)

        features = cloud_to_features(xyz_n, colors_n, self.use_color)
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(np.asarray(labels, dtype=np.int64)).long()
        return features, labels


def get_data_loader(data_dir, batch_size=4, num_points=40960,
                    num_classes=NUM_CLASSES_DEFAULT, split="train",
                    num_workers=0, use_color=True, augment=None,
                    voxel_size=None, shuffle=None, drop_last=True,
                    seed=None):
    """
    创建数据加载器。

    参数与旧版保持一致，并新增论文预处理选项。
    """
    dataset = PointCloudDataset(
        data_dir=data_dir,
        num_points=num_points,
        num_classes=num_classes,
        split=split,
        use_color=use_color,
        augment=(split == "train") if augment is None else augment,
        voxel_size=voxel_size,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return loader


def split_xyz_rgb(features, use_color=True):
    """从批次特征中分离坐标与颜色/原始特征。"""
    xyz = features[:, :, :3]
    feats = features if (use_color and features.shape[-1] > 3) else xyz
    return xyz, feats


if __name__ == "__main__":
    # 快速自检
    import tempfile
    tmp = tempfile.mkdtemp()
    for name in ("train", "val"):
        os.makedirs(os.path.join(tmp, name), exist_ok=True)
        n = 2000
        xyz = np.random.randn(n, 3)
        rgb = np.random.randint(0, 255, size=(n, 3))
        lab = np.random.randint(0, NUM_CLASSES_DEFAULT, size=n)
        np.save(os.path.join(tmp, name, "cloud.npy"),
                np.column_stack([xyz, rgb, lab]))
    ds = PointCloudDataset(tmp, num_points=1024, split="train")
    f, l = ds[0]
    print("features:", f.shape, "labels:", l.shape, "unique:", torch.unique(l).tolist())
