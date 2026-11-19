#!/usr/bin/env python3
"""Equalise exposure / white balance across the MVS images before texturing.

The underside was filmed later under dimmer, warmer light, so its texture came out
much darker than the top. The clay is one uniform colour, so each image is scaled
(per channel, in linear light) so that the median colour of its object pixels
matches a common reference. Shading and shadows inside each image are preserved.

Originals are moved to <mvs>/images_orig (once); normalised copies replace <mvs>/images.
Usage: python scripts/normalize_colors.py --mvs work/mvs [--ref-prefix video__f_0]
       python scripts/normalize_colors.py --calibrate-photos work/mvs work/mvs/scene_refine.ply
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def to_lin(x):
    return np.power(x / 255.0, 2.2)


def to_srgb(x):
    return np.clip(np.power(np.clip(x, 0, 1), 1 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvs", default="work/mvs")
    ap.add_argument("--ref-prefix", default="video__f_0",
                    help="images whose clay colour defines the reference (top orbit, 0-56 s)")
    ap.add_argument("--max-gain", type=float, default=3.0)
    args = ap.parse_args()

    mvs = Path(args.mvs)
    src = mvs / "images_orig"
    if not src.exists():
        shutil.move(str(mvs / "images"), str(src))
        (mvs / "images").mkdir()
    names = sorted(p.name for p in src.glob("*.jpg"))

    med = {}
    for n in names:
        img = cv2.imread(str(src / n)).astype(np.float32)
        m = cv2.imread(str(mvs / "masks" / (Path(n).stem + ".mask.png")), cv2.IMREAD_GRAYSCALE)
        m = cv2.erode(m, np.ones((15, 15), np.uint8)) > 127
        lin = to_lin(img[m])
        # use the brighter half of the clay (lit surface) so self-shadows don't skew it
        lum = lin.mean(1)
        lit = lin[lum >= np.median(lum)]
        med[n] = np.median(lit, axis=0)
    ref_names = [n for n in names if n.startswith(args.ref_prefix)] or names
    ref = np.median(np.stack([med[n] for n in ref_names]), axis=0)
    print("reference clay colour (linear BGR):", np.round(ref, 3))

    for n in names:
        gain = np.clip(ref / np.maximum(med[n], 1e-4), 1 / args.max_gain, args.max_gain)
        img = cv2.imread(str(src / n)).astype(np.float32)
        out = to_srgb(to_lin(img) * gain)
        cv2.imwrite(str(mvs / "images" / n), out, [cv2.IMWRITE_JPEG_QUALITY, 97])
    gains = np.stack([np.clip(ref / med[n], 1 / args.max_gain, args.max_gain) for n in names])
    print(f"{len(names)} images normalised; gain min {gains.min():.2f} / median "
          f"{np.median(gains):.2f} / max {gains.max():.2f}")
    worst = sorted(names, key=lambda n: -np.mean(ref / med[n]))[:5]
    print("brightened most:", ", ".join(f"{n} x{np.mean(ref / med[n]):.2f}" for n in worst))




def calibrate_photos(mvs, mesh_path, photo_prefix="photos__", min_cos=0.5):
    """Second pass: match each still photo to the video on surface points both see.
    The stills (4K) win texture selection, so their exposure must agree with the video."""
    import sys
    import trimesh
    sys.path.insert(0, str(Path(__file__).parent))
    from carve_mesh import read_model
    mvs = Path(mvs)
    m = trimesh.load(mesh_path, process=False, force="mesh")
    fc, fn = m.triangles_center, m.face_normals
    samples = {}
    for R, t, (w, h, fx, fy, cx, cy), name in read_model(mvs):
        C = -R.T @ t
        d = C - fc
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        Xc = fc @ R.T + t
        z = Xc[:, 2]
        u = fx * Xc[:, 0] / z + cx - 0.5
        v = fy * Xc[:, 1] / z + cy - 0.5
        ok = (z > 0) & ((fn * d).sum(1) > min_cos) & (u >= 0) & (v >= 0) & (u < w - 1) & (v < h - 1)
        img = cv2.imread(str(mvs / "images" / name)).astype(np.float32)
        msk = cv2.imread(str(mvs / "masks" / (Path(name).stem + ".mask.png")), 0)
        msk = cv2.erode(msk, np.ones((9, 9), np.uint8)) > 127
        idx = np.nonzero(ok)[0]
        ui, vi = u[idx].astype(int), v[idx].astype(int)
        good = msk[vi, ui]
        samples[name] = (idx[good], to_lin(img[vi[good], ui[good]]))
    # video consensus per face
    nf = len(fc)
    acc, cnt = np.zeros((nf, 3)), np.zeros(nf)
    for n, (idx, col) in samples.items():
        if n.startswith(photo_prefix):
            continue
        np.add.at(acc, idx, col)
        np.add.at(cnt, idx, 1)
    vid = acc / np.maximum(cnt, 1)[:, None]
    for n, (idx, col) in samples.items():
        if not n.startswith(photo_prefix):
            continue
        sel = cnt[idx] >= 3
        if sel.sum() < 200:
            print(f"  {n}: too few shared points, unchanged")
            continue
        gain = np.median(vid[idx[sel]], 0) / np.maximum(np.median(col[sel], 0), 1e-4)
        gain = np.clip(gain, 0.5, 2.5)
        img = cv2.imread(str(mvs / "images" / n)).astype(np.float32)
        cv2.imwrite(str(mvs / "images" / n), to_srgb(to_lin(img) * gain),
                    [cv2.IMWRITE_JPEG_QUALITY, 97])
        print(f"  {n}: gain {np.round(gain, 2)} from {sel.sum()} shared points")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--calibrate-photos":
        # python scripts/normalize_colors.py --calibrate-photos work/mvs work/mvs/scene_refine.ply
        calibrate_photos(sys.argv[2], sys.argv[3])
    else:
        main()
