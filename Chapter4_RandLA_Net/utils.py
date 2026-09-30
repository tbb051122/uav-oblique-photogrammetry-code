# -*- coding: utf-8 -*-
"""
点云基础操作工具
=================
KNN 邻域搜索、邻居特征聚合、随机采样（论文 3.3.1 中 RandLA-Net 使用的
随机采样策略）、最近邻特征插值（解码器上采样）等。
"""

import torch


def batched_gather(features, idx):
    """
    沿批量维度索引聚集特征。

    参数:
        features: [B, N, C]
        idx:      [B, S] 或 [B, S, K]
    返回:
        [B, S, C] 或 [B, S, K, C]
    """
    batch = torch.arange(features.shape[0], device=features.device)
    if idx.dim() == 2:
        batch = batch.view(-1, 1).expand_as(idx)
        return features[batch, idx]
    if idx.dim() == 3:
        batch = batch.view(-1, 1, 1).expand_as(idx)
        return features[batch, idx]
    raise ValueError("idx 维度必须为 2 或 3")


def knn_search(xyz, k, chunk_size=512):
    """
    批量 K 近邻搜索（基于欧氏距离，包含自身点）。

    为避免一次生成整张 [N, N] 距离矩阵导致显存溢出，按 chunk 分块计算。

    参数:
        xyz:        [B, N, 3] 点云坐标
        k:          近邻数量
        chunk_size: 分块查询点数
    返回:
        [B, N, K] 每个点的 K 个近邻索引（按距离升序）
    """
    b, n, _ = xyz.shape
    k = min(k, n)
    device = xyz.device
    out_idx = torch.empty(b, n, k, dtype=torch.long, device=device)

    for bi in range(b):
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            # [chunk, n] 分块距离矩阵
            dist = torch.cdist(xyz[bi, start:end].unsqueeze(0),
                               xyz[bi].unsqueeze(0)).squeeze(0)
            _, idx = torch.topk(dist, k, dim=-1, largest=False, sorted=True)
            out_idx[bi, start:end] = idx
    return out_idx


def random_sampling(xyz, features, num_samples):
    """
    随机采样（论文 3.3.1：RandLA-Net 采用随机采样降低计算规模）。

    每帧点云独立执行无放回随机采样；点不足时执行有放回采样。
    返回采样后的坐标、特征以及用于还原的索引。
    """
    b, n, _ = xyz.shape
    device = xyz.device
    idx_list = []
    for bi in range(b):
        if num_samples <= n:
            perm = torch.randperm(n, device=device)
            idx_list.append(perm[:num_samples])
        else:
            idx_list.append(torch.randint(0, n, (num_samples,), device=device))
    idx = torch.stack(idx_list, dim=0)
    sampled_xyz = batched_gather(xyz, idx)
    sampled_features = batched_gather(features, idx) if features is not None else None
    return sampled_xyz, sampled_features, idx


def gather_neighbour_features(features, idx):
    """根据邻居索引聚合特征：[B, N, C] + [B, N, K] -> [B, N, K, C]"""
    return batched_gather(features, idx)


def nearest_neighbor_interpolation(xyz_source, features_source,
                                   xyz_query, chunk_size=512):
    """
    最近邻特征插值（RandLA-Net 解码器上采样）。

    对 xyz_query 中的每个点，在 xyz_source（上一层较疏点云）中寻找最近点，
    并将 features_source 中对应特征赋值给查询点。

    参数:
        xyz_source:     [B, Ns, 3] 待插值源点（疏）
        features_source: [B, Ns, C]
        xyz_query:      [B, Nq, 3] 目标点（密）
    返回:
        [B, Nq, C] 插值后的特征
    """
    b, nq, _ = xyz_query.shape
    _, ns, c = features_source.shape
    device = xyz_query.device
    feats_t = features_source.permute(0, 2, 1)  # [B, C, Ns]
    out = torch.empty(b, nq, c, dtype=features_source.dtype, device=device)

    for bi in range(b):
        for start in range(0, nq, chunk_size):
            end = min(start + chunk_size, nq)
            dist = torch.cdist(xyz_query[bi, start:end].unsqueeze(0),
                               xyz_source[bi].unsqueeze(0)).squeeze(0)
            near_idx = dist.argmin(dim=-1)  # [chunk]
            chunk_out = feats_t[bi][:, near_idx]  # [C, chunk]
            out[bi, start:end] = chunk_out.transpose(0, 1)
    return out


def square_distance(src, dst):
    """两组点间的成对平方距离 [B, N, M]。"""
    b, n, _ = src.shape
    _, m, _ = dst.shape
    dist = -2.0 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(b, n, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(b, 1, m)
    return dist


def index_points(points, idx):
    """兼容旧版：根据索引从点云中选取点。"""
    return batched_gather(points, idx)
