# -*- coding: utf-8 -*-
"""
把整块 LAS 按指定体素下采样成训练样本，保留下原始 RGB、坐标和 CRS。

用法：
python make_tile_sample.py --input Tile_27.las --output Tile_27_sample_005m.las --voxel 0.05
"""

import argparse
import copy
import os

import numpy as np
import laspy


def main():
    parser = argparse.ArgumentParser(description="LAS 体素下采样")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voxel", type=float, default=0.05)
    args = parser.parse_args()

    xs, ys, zs, rs, gs, bs = [], [], [], [], [], []
    with laspy.open(args.input) as fh:
        header = fh.header
        for pts in fh.chunk_iterator(4_000_000):
            xs.append(np.asarray(pts.x, dtype=np.float64))
            ys.append(np.asarray(pts.y, dtype=np.float64))
            zs.append(np.asarray(pts.z, dtype=np.float64))
            rs.append(np.asarray(pts.red, dtype=np.uint16))
            gs.append(np.asarray(pts.green, dtype=np.uint16))
            bs.append(np.asarray(pts.blue, dtype=np.uint16))

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    r = np.concatenate(rs)
    g = np.concatenate(gs)
    b = np.concatenate(bs)
    print("input points", len(x))

    origin = np.array([x.min(), y.min(), z.min()], dtype=np.float64)
    kx = np.floor((x - origin[0]) / args.voxel).astype(np.int64)
    ky = np.floor((y - origin[1]) / args.voxel).astype(np.int64)
    kz = np.floor((z - origin[2]) / args.voxel).astype(np.int64)
    key = (kx * 10_000_000 + ky) * 1_000_000 + kz
    _, idx = np.unique(key, return_index=True)
    print("kept points", len(idx))

    out_header = laspy.LasHeader(point_format=header.point_format.id,
                                 version=header.version)
    out_header.scales = header.scales
    out_header.offsets = header.offsets
    out_header.vlrs = copy.deepcopy(list(header.vlrs))

    out = laspy.LasData(out_header)
    out.x = x[idx]
    out.y = y[idx]
    out.z = z[idx]
    out.red = r[idx]
    out.green = g[idx]
    out.blue = b[idx]
    out.classification = np.zeros(len(idx), dtype=np.uint8)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.write(args.output)
    print("saved", args.output)


if __name__ == "__main__":
    main()
