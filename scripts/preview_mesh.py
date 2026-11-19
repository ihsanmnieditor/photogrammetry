#!/usr/bin/env python3
"""Quick CPU preview of a mesh (shaded z-buffer splat of vertices, 4 views).
Usage: python scripts/preview_mesh.py mesh.ply out.png [--up-model work/mvs/sparse]"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).parent))


def render(V, N, R, size=600):
    P = V @ R.T
    Nn = N @ R.T
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    s = (size - 40) / (hi - lo).max()
    xy = ((P[:, :2] - lo) * s + 20).astype(int)
    xy[:, 1] = size - 1 - xy[:, 1]
    order = np.argsort(P[:, 2])  # far to near (camera looks down -z)
    img = np.full((size, size), 255, np.uint8)
    light = np.array([0.4, 0.6, 0.7]); light /= np.linalg.norm(light)
    shade = np.clip(Nn @ light, 0, 1) * 200 + 40
    for i in order:
        x, y = xy[i]
        if 0 <= x < size and 0 <= y < size:
            img[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = shade[i]
    return img


def rot(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--up-model", default="")
    a = ap.parse_args()
    m = trimesh.load(a.mesh, process=False, force="mesh")
    V = np.asarray(m.vertices) - m.vertices.mean(0)
    N = np.asarray(m.vertex_normals)
    if a.up_model:
        from export_web import rotation_to, world_up_from_cameras
        up = world_up_from_cameras(Path(a.up_model) / "images.txt", r"video__f_(\d+)", 56000)
        R0 = rotation_to(up, np.array([0, 1.0, 0]))
        V, N = V @ R0.T, N @ R0.T
    views = [rot("x", 20) @ rot("y", ang) for ang in (0, 90, 180, 270)] + [rot("x", -60)]
    cv2.imwrite(a.out, np.hstack([render(V, N, R, 420) for R in views]))
    print(f"{len(V)} vertices -> {a.out}")


if __name__ == "__main__":
    main()
