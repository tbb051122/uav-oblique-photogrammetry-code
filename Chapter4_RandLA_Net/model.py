# -*- coding: utf-8 -*-
"""
改进 RandLA-Net 网络模型（对应论文 4.2）

网络包含：
* 输入特征优化（4.2.2）：融合 XYZ 与 RGB，形成 XYZRGB 六维联合特征；
* 随机采样（4.2.1）：编码器每层将点数降低到 1/4，控制大规模点云计算量；
* 局部特征聚合 LFA：LocSE + Attentive Pooling（扩张残差块）；
* 多尺度局部特征融合模块（4.2.3）：编码器第一层使用 K=8/16/32 三个
  邻域尺度提取局部特征、拼接并压缩；论文指出该模块仅在第一层使用，
  后续层级采用单尺度特征以平衡效率与表达能力；
* 编码器-解码器：解码器以最近邻特征插值恢复空间分辨率并与跳跃连接融合；
* 分类头：输出每个点的类别 logits。

输入:
    xyz:      [B, N, 3] 坐标（已归一化）
    features: [B, N, 3]（仅 XYZ）或 [B, N, 6]（XYZRGB）
输出:
    [B, num_classes, N] 每个点的类别分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (batched_gather, gather_neighbour_features, knn_search,
                   nearest_neighbor_interpolation, random_sampling)


def _conv_bn_relu1d(in_c, out_c):
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=1, bias=False),
        nn.BatchNorm1d(out_c),
        nn.ReLU(inplace=True),
    )


def _conv_bn_relu2d(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class LocalSpatialEncoding(nn.Module):
    """
    局部空间编码（LocSE，论文 4.2.1 LFA 的组成部分）。

    对每个点的邻域，编码中心点坐标、邻居坐标、相对位移与距离，得到
    几何关系特征；同时将邻居特征经过共享 MLP 变换后与几何编码相加。
    """
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.geo_mlp = _conv_bn_relu2d(10, out_channels)
        self.neighbor_mlp = _conv_bn_relu2d(in_channels, out_channels)

    def forward(self, xyz, features):
        """
        xyz:      [B, N, 3]
        features: [B, N, C_in]
        return:   [B, N, K, C_out]
        """
        b, n, c = features.shape
        idx = knn_search(xyz, self.k)
        neighbor_xyz = batched_gather(xyz, idx)              # [B,N,K,3]
        center_xyz = xyz[:, :, None, :].expand_as(neighbor_xyz)
        relative = neighbor_xyz - center_xyz
        dist = torch.norm(relative, dim=-1, keepdim=True)
        local_info = torch.cat([center_xyz, neighbor_xyz,
                                relative, dist], dim=-1)     # [B,N,K,10]

        local_encoding = self.geo_mlp(local_info.permute(0, 3, 1, 2))
        neighbor_features = batched_gather(features, idx)    # [B,N,K,C_in]
        feature_encoding = self.neighbor_mlp(
            neighbor_features.permute(0, 3, 1, 2))
        encoding = local_encoding + feature_encoding         # [B,C,N,K]
        return encoding.permute(0, 2, 3, 1)


class AttentivePooling(nn.Module):
    """
    注意力池化（LFA 第二部分）：用注意力权重聚合 K 个邻居特征。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
        )
        self.mlp = _conv_bn_relu1d(in_channels, out_channels)

    def forward(self, x):
        """
        x: [B, N, K, C]
        return: [B, N, C_out]
        """
        x_perm = x.permute(0, 3, 1, 2)      # [B,C,N,K]
        scores = self.score_mlp(x_perm)     # [B,1,N,K]
        scores = F.softmax(scores, dim=-1).squeeze(1)  # [B,N,K]
        weighted = x * scores.unsqueeze(-1)
        pooled = weighted.sum(dim=2)        # [B,N,C]
        out = self.mlp(pooled.permute(0, 2, 1))
        return out.permute(0, 2, 1)


class DilatedResidualBlock(nn.Module):
    """
    扩张残差块（DRB）：LocSE -> Attentive Pooling + 残差连接。
    """
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.locse = LocalSpatialEncoding(in_channels, out_channels, k)
        self.att_pool = AttentivePooling(out_channels, out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz, features):
        residual = self.shortcut(features.permute(0, 2, 1)).permute(0, 2, 1)
        encoded = self.locse(xyz, features)
        pooled = self.att_pool(encoded)
        return F.relu(pooled + residual, inplace=True)


class MultiScaleLocalFeatureAggregation(nn.Module):
    """
    多尺度局部特征融合模块（论文 4.2.3）。

    对每个点分别以 K=8/16/32 构建邻域，各尺度内构造局部关系特征并经
    共享 MLP 映射，随后在各尺度内最大池化；最后将多尺度特征拼接并经过
    1x1 卷积压缩到目标通道数。

    论文 4.2.3 小结指出该模块仅在编码器第一层使用。
    """
    def __init__(self, in_channels, out_channels, scales=(8, 16, 32)):
        super().__init__()
        self.scales = list(scales)
        self.geo_mlp = _conv_bn_relu2d(10, out_channels)
        self.neighbor_mlp = _conv_bn_relu2d(in_channels, out_channels)
        self.compress = _conv_bn_relu1d(
            out_channels * len(self.scales), out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz, features):
        b, n, c = features.shape
        residual = self.shortcut(features.permute(0, 2, 1))  # [B,C,N]
        scale_pools = []

        for k in self.scales:
            idx = knn_search(xyz, min(k, n))
            neighbor_xyz = batched_gather(xyz, idx)
            center_xyz = xyz[:, :, None, :].expand_as(neighbor_xyz)
            relative = neighbor_xyz - center_xyz
            dist = torch.norm(relative, dim=-1, keepdim=True)
            local_info = torch.cat(
                [center_xyz, neighbor_xyz, relative, dist], dim=-1)
            neighbor_features = batched_gather(features, idx)

            local_encoding = self.geo_mlp(local_info.permute(0, 3, 1, 2))
            feature_encoding = self.neighbor_mlp(
                neighbor_features.permute(0, 3, 1, 2))
            encoding = local_encoding + feature_encoding     # [B,C,N,K]
            pooled = encoding.max(dim=-1).values             # 论文采用最大池化
            scale_pools.append(pooled)

        fused = torch.cat(scale_pools, dim=1)                # [B,C*S,N]
        out = self.compress(fused) + residual
        return F.relu(out, inplace=True).permute(0, 2, 1)


class RandLANet(nn.Module):
    """
    改进 RandLA-Net。

    结构（编码器每层两个残差块，层间点数降为 1/4）：
        第一层:  MS-LFA(8->16) + DRB(16->16)   （多尺度融合仅此处）
        第二层:  DRB(16->32) + DRB(32->32)
        第三层:  DRB(32->64) + DRB(64->64)
        第四层:  DRB(64->128) + DRB(128->128)
        第五层:  DRB(128->256) + DRB(256->256)
    解码器通过最近邻插值逐步恢复分辨率，并与编码器跳跃连接拼接。
    """
    def __init__(self, num_classes=6,
                 in_channels=6, k=16, num_points=40960,
                 scales=(8, 16, 32), use_multi_scale=True):
        super().__init__()
        # 避免循环导入：num_classes 由调用方显式给出
        self.num_classes = num_classes
        self.k = k
        d = 8
        self.fc_start = _conv_bn_relu1d(in_channels, d)

        self.encoder = nn.ModuleList()
        self.blocks_per_level = [2, 2, 2, 2, 2]
        if use_multi_scale:
            self.encoder.append(MultiScaleLocalFeatureAggregation(
                d, 2 * d, scales=scales))
        else:
            # 消融设置：编码器第一层改用固定邻域的单尺度局部特征聚合
            self.encoder.append(DilatedResidualBlock(d, 2 * d, k=k))
        self.encoder.append(DilatedResidualBlock(2 * d, 2 * d, k=k))
        self.encoder.append(DilatedResidualBlock(2 * d, 4 * d, k=k))
        self.encoder.append(DilatedResidualBlock(4 * d, 4 * d, k=k))
        self.encoder.append(DilatedResidualBlock(4 * d, 8 * d, k=k))
        self.encoder.append(DilatedResidualBlock(8 * d, 8 * d, k=k))
        self.encoder.append(DilatedResidualBlock(8 * d, 16 * d, k=k))
        self.encoder.append(DilatedResidualBlock(16 * d, 16 * d, k=k))
        self.encoder.append(DilatedResidualBlock(16 * d, 32 * d, k=k))
        self.encoder.append(DilatedResidualBlock(32 * d, 32 * d, k=k))

        # 解码器压缩：拼接当前特征与跳跃连接后降维
        self.decoder = nn.ModuleList([
            _conv_bn_relu1d(32 * d + 16 * d, 16 * d),   # 256+128 -> 128
            _conv_bn_relu1d(16 * d + 8 * d, 8 * d),     # 128+64  -> 64
            _conv_bn_relu1d(8 * d + 4 * d, 4 * d),      # 64+32   -> 32
            _conv_bn_relu1d(4 * d + 2 * d, 2 * d),      # 32+16   -> 16
        ])
        self.fc_end = nn.Conv1d(2 * d, num_classes, kernel_size=1)

    def forward(self, xyz, features):
        """
        xyz:      [B, N, 3] 点云坐标（归一化）
        features: [B, N, 3] 或 [B, N, 6]（XYZRGB）
        return:   [B, num_classes, N]
        """
        b, n, _ = xyz.shape
        device = xyz.device
        f = self.fc_start(features.permute(0, 2, 1)).permute(0, 2, 1)

        # ---------- 编码器 ----------
        skips = []            # 每层降采样前的 (xyz, features)
        block_i = 0
        for li, n_blocks in enumerate(self.blocks_per_level):
            for _ in range(n_blocks):
                f = self.encoder[block_i](xyz, f)
                block_i += 1
            skips.append((xyz, f))
            if li < len(self.blocks_per_level) - 1:
                n_next = max(n // (4 ** (li + 1)), 8)
                if n_next < f.shape[1]:
                    xyz, f, _ = random_sampling(xyz, f, n_next)

        # ---------- 解码器 ----------
        cur_xyz, cur_feat = skips[-1]
        for up_i, skip in enumerate(reversed(skips[:-1])):
            skip_xyz, skip_feat = skip
            interp = nearest_neighbor_interpolation(
                cur_xyz, cur_feat, skip_xyz)
            cat = torch.cat([interp, skip_feat], dim=-1)
            cur_feat = self.decoder[up_i](cat.permute(0, 2, 1)).permute(0, 2, 1)
            cur_xyz = skip_xyz

        out = self.fc_end(cur_feat.permute(0, 2, 1))
        return out


if __name__ == "__main__":
    # 网络自检
    net = RandLANet(num_classes=6, in_channels=6, num_points=2048)
    x = torch.randn(2, 2048, 3)
    feat = torch.randn(2, 2048, 6)
    with torch.no_grad():
        out = net(x, feat)
    print("input:", tuple(x.shape), "output:", tuple(out.shape))
    print("参数数量: {:.2f} M".format(
        sum(p.numel() for p in net.parameters()) / 1e6))
