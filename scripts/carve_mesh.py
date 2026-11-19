#!/usr/bin/env python3
"""Remove mesh parts that are not the object, using the object masks (visual hull).

A vertex that projects OUTSIDE the object mask in a large fraction of the views
that see it cannot be on the object: it comes from shadows, the mirror
reflection on the tablet, or depth noise at the silhouette. Those vertices' faces
are removed, the largest connected piece is kept and holes are closed again.

Usage: python scripts/carve_mesh.py in.ply out.ply --mvs work/mvs [--min-out-frac 0.3]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pymeshlab as ml
import trimesh


def read_model(mvs):
    cams = {}
    for l in open(Path(mvs) / "sparse" / "cameras.txt"):
        if l.startswith("#") or not l.strip():
            continue
        p = l.split()
        cams[int(p[0])] = (int(p[2]), int(p[3]), *map(float, p[4:8]))
    views = []
    lines = [l for l in open(Path(mvs) / "sparse" / "images.txt") if not l.startswith("#")]
    for l in lines[0::2]:
        p = l.split()
        if len(p) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, p[1:8])
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        views.append((R, np.array([tx, ty, tz]), cams[int(p[8])], p[9]))
    return views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--mvs", default="work/mvs")
    ap.add_argument("--min-out-frac", type=float, default=0.3,
                    help="remove a vertex if it is outside the mask in at least this "
                         "fraction of the views that see it (masks drop dark contact "
                         "shadows, so a real surface point can fall outside a few masks)")
    ap.add_argument("--erode", type=int, default=0, help="extra mask erosion (px)")
    args = ap.parse_args()

    m = trimesh.load(args.inp, process=False, force="mesh")
    V = np.asarray(m.vertices)
    out_count = np.zeros(len(V), np.int32)
    seen = np.zeros(len(V), np.int32)
    for R, t, (w, h, fx, fy, cx, cy), name in read_model(args.mvs):
        mask = cv2.imread(str(Path(args.mvs) / "masks" / (Path(name).stem + ".mask.png")),
                          cv2.IMREAD_GRAYSCALE) > 127
        if args.erode:
            k = 2 * args.erode + 1
            mask = cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        Xc = V @ R.T + t
        z = Xc[:, 2]
        u = fx * Xc[:, 0] / z + cx - 0.5   # COLMAP -> pixel index
        v = fy * Xc[:, 1] / z + cy - 0.5
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inside = (z > 0) & (ui >= 0) & (vi >= 0) & (ui < w) & (vi < h)
        # crop border: the crop was made around the mask, so the border itself is background
        seen += inside
        idx = np.nonzero(inside)[0]
        out_count[idx] += ~mask[vi[idx], ui[idx]]
    frac = out_count / np.maximum(seen, 1)
    for f in (0.05, 0.1, 0.2, 0.3, 0.5):
        print(f"  outside in >= {f:.0%} of views: {(frac >= f).sum()}")
    bad = frac >= args.min_out_frac
    print(f"{bad.sum()} / {len(V)} vertices outside the visual hull "
          f"(median views per vertex {np.median(seen):.0f})")

    keep_face = ~bad[m.faces].any(1)
    m2 = trimesh.Trimesh(V, m.faces[keep_face], process=False)
    parts = m2.split(only_watertight=False)
    m2 = max(parts, key=lambda p: len(p.faces))
    tmp = Path(args.out).with_suffix(".tmp.ply")
    m2.export(tmp)

    ms = ml.MeshSet()
    ms.load_new_mesh(str(tmp))
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=200000, refinehole=True, selfintersection=False)
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_remove_unreferenced_vertices()
    ms.save_current_mesh(args.out, save_vertex_color=False, save_vertex_normal=False)
    tmp.unlink()
    r = trimesh.load(args.out, process=False)
    print(f"out: {len(r.vertices)} v / {len(r.faces)} f, watertight={r.is_watertight}, "
          f"bodies={r.body_count}")


if __name__ == "__main__":
    main()
