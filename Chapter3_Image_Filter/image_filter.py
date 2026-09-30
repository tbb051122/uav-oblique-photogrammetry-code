# -*- coding: utf-8 -*-
"""
影像覆盖贡献评价与低贡献影像筛选（对应论文 2.6.2、2.6.3）

1. 基础覆盖率 bi = 影像覆盖的网格数 / 测区网格总数；
2. 新增覆盖率 di = 处理顺序中该影像“新覆盖”的网格数 / 网格总数
   （与已保留影像重叠越多，新增贡献越低）；
3. 综合贡献评分 Si = alpha * bi + beta * di，论文取 alpha = beta = 0.5；
4. 按评分降序排序，评分低于阈值 T 的影像标记为低贡献候选影像；
   为保证“优化后影像集合仍保持测区完整覆盖”（论文 2.6.3），
   必要时从低贡献候选中按新增覆盖最大原则补回影像。
"""

import numpy as np

from ray_casting import points_in_polygon


def coverage_score(image_polygon, sample_points):
    """
    单幅影像覆盖比例：
        score = 落在影像覆盖多边形内的采样点数 / 采样点总数
    """
    sample = np.asarray(sample_points, dtype=np.float64).reshape(-1, 2)
    if len(sample) == 0:
        raise ValueError("sample_points 为空")
    inside = points_in_polygon(sample, image_polygon)
    return float(inside.mean())


def build_coverage_matrix(footprints, sample_points) -> np.ndarray:
    """
    构建影像-网格覆盖关系矩阵（论文 2.6.1 的 m_ij 矩阵）。

    参数:
        footprints:    每幅影像的地面覆盖多边形列表
        sample_points: 测区网格中心采样点（(x, y) 列表）

    返回:
        (影像数, 网格数) bool 数组，True 表示该影像覆盖该网格
    """
    sample = np.asarray(sample_points, dtype=np.float64).reshape(-1, 2)
    rows = []
    for polygon in footprints:
        rows.append(points_in_polygon(sample, polygon))
    if not rows:
        return np.zeros((0, len(sample)), dtype=bool)
    return np.stack(rows, axis=0)


def compute_contribution_scores(cover_matrix: np.ndarray,
                                alpha: float = 0.5,
                                beta: float = 0.5):
    """
    计算每幅影像的覆盖率、新增覆盖率与综合贡献评分。

    新增覆盖率的“已有影像”集合按基础覆盖率降序逐个加入：
    覆盖率高的影像先加入，后续影像若与已加入影像高度重叠，其新增贡献自动降低。

    返回:
        base_ratio: (I,) 基础覆盖率
        new_ratio:  (I,) 新增覆盖率（相对处理顺序中已保留影像）
        scores:     (I,) 综合贡献评分 S_i = alpha*base + beta*new
    """
    cover = np.asarray(cover_matrix, dtype=bool)
    n_img, n_grid = cover.shape
    if n_grid == 0:
        raise ValueError("覆盖矩阵没有网格列")
    if alpha < 0 or beta < 0:
        raise ValueError("权重 alpha、beta 必须非负")

    base_ratio = cover.sum(axis=1) / float(n_grid)
    order = np.argsort(-base_ratio, kind="stable")

    new_ratio = np.zeros(n_img, dtype=np.float64)
    covered = np.zeros(n_grid, dtype=bool)
    for i in order:
        newly = cover[i] & (~covered)
        new_ratio[i] = float(newly.sum()) / float(n_grid)
        covered |= cover[i]

    scores = alpha * base_ratio + beta * new_ratio
    return base_ratio, new_ratio, scores


def optimize_image_set(cover_matrix: np.ndarray,
                       names,
                       alpha: float = 0.5,
                       beta: float = 0.5,
                       threshold: float = 0.1,
                       ensure_full_coverage: bool = True):
    """
    低贡献影像筛选（论文 2.6.3）。

    参数:
        cover_matrix:          影像-网格覆盖矩阵 (I, J)
        names:                 I 个影像名称（用于输出）
        alpha, beta:           综合评分权重，论文默认均为 0.5
        threshold:             评分阈值 T，低于 T 的影像标记为低贡献
        ensure_full_coverage:  True 时在筛选后补回必要影像，保证测区网格全覆盖

    返回:
        keep_flags:  (I,) bool，True 表示保留
        report:      dict，包含 base/new/score/keep 等字段，可直接转为 CSV
    """
    cover = np.asarray(cover_matrix, dtype=bool)
    n_img, n_grid = cover.shape
    if len(names) != n_img:
        raise ValueError("names 数量必须与覆盖矩阵行数一致")

    base_ratio, new_ratio, scores = compute_contribution_scores(
        cover, alpha=alpha, beta=beta)

    keep_flags = scores >= threshold

    if ensure_full_coverage:
        # 若阈值筛选后存在未覆盖网格，则按“新增未覆盖网格最多”原则补回影像
        if n_img > 0:
            uncovered = ~cover[keep_flags].any(axis=0) if keep_flags.any() else np.ones(n_grid, bool)
            while uncovered.any() and (~keep_flags).any():
                candidate_ids = np.where(~keep_flags)[0]
                gains = cover[candidate_ids][:, uncovered].sum(axis=1)
                best_pos = int(np.argmax(gains))
                if gains[best_pos] <= 0:
                    break  # 剩余影像已无法补充任何未覆盖网格
                best = int(candidate_ids[best_pos])
                keep_flags[best] = True
                uncovered &= ~cover[best]

    report = {
        "name": list(names),
        "covered_grids": cover.sum(axis=1).tolist(),
        "total_grids": [n_grid] * n_img,
        "base_ratio": base_ratio.tolist(),
        "new_ratio": new_ratio.tolist(),
        "score": scores.tolist(),
        "keep": keep_flags.tolist(),
    }
    return keep_flags, report
