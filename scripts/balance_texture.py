#!/usr/bin/env python3
"""Low-frequency colour balance of the OpenMVS texture atlas.

OpenMVS's global seam leveling makes neighbouring patches agree, but it also
spreads the dark contact-shadow rim around the base over the whole underside.
Here every face gets a *target* colour = median, over all cameras that see it
face-on and inside the object mask, of the (exposure-normalised) image colour.
The ratio target / current texture colour is smoothed over the surface (so only
large-scale tone changes; fingerprints and local shadows stay as they are),
clamped, and multiplied into the atlas per face (in linear light).

Usage: python scripts/balance_texture.py work/mvs/scene_texture.ply out.png --mvs work/mvs
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))
from carve_mesh import read_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--mvs", default="work/mvs")
    ap.add_argument("--min-cos", type=float, default=0.5)
    ap.add_argument("--smooth-iters", type=int, default=40)
    ap.add_argument("--max-gain", type=float, default=2.5)
    args = ap.parse_args()

    m = trimesh.load(args.mesh, process=False)
    tex_img = m.visual.material.image.convert("RGB")
    tex = np.asarray(tex_img, np.float32) / 255.0
    H, W, _ = tex.shape
    F = m.faces
    fc, fn = m.triangles_center, m.face_normals
    uvt = m.visual.uv[F]  # (nf, 3, 2)

    # current texture colour per face: mean of centroid + 3 shrunken corners
    samples = [uvt.mean(1)] + [uvt.mean(1) * 0.5 + uvt[:, k] * 0.5 for k in range(3)]
    cur = np.zeros((len(F), 3))
    for s in samples:
        px = np.clip(s[:, 0] * (W - 1), 0, W - 1).astype(int)
        py = np.clip((1 - s[:, 1]) * (H - 1), 0, H - 1).astype(int)
        cur += tex[py, px]
    cur = np.power(cur / len(samples), 2.2)

    # multi-view target colour (median over good views); streamed, keep up to K samples
    K = 24
    buf = np.full((len(F), K, 3), np.nan, np.float32)
    cnt = np.zeros(len(F), np.int32)
    for R, t, (w, h, fx, fy, cx, cy), name in read_model(args.mvs):
        C = -R.T @ t
        d = C - fc
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        cos = (fn * d).sum(1)
        Xc = fc @ R.T + t
        z = Xc[:, 2]
        u = fx * Xc[:, 0] / z + cx - 0.5
        v = fy * Xc[:, 1] / z + cy - 0.5
        ok = (z > 0) & (cos > args.min_cos) & (u >= 0) & (v >= 0) & (u < w - 1) & (v < h - 1)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            continue
        img = cv2.imread(str(Path(args.mvs) / "images" / name))[..., ::-1].astype(np.float32) / 255
        msk = cv2.imread(str(Path(args.mvs) / "masks" / (Path(name).stem + ".mask.png")), 0)
        msk = cv2.erode(msk, np.ones((7, 7), np.uint8)) > 127
        ui, vi = u[idx].astype(int), v[idx].astype(int)
        good = msk[vi, ui]
        idx, ui, vi = idx[good], ui[good], vi[good]
        col = np.power(img[vi, ui], 2.2)
        slot = cnt[idx]
        room = slot < K
        buf[idx[room], slot[room]] = col[room]
        cnt[idx[room]] += 1
    have = cnt >= 3
    target = np.nanmedian(buf, axis=1)
    print(f"faces with >=3 good views: {have.mean():.1%}")

    # log-gain per face, smoothed over face adjacency (only where measured, then diffused)
    g = np.zeros((len(F), 3))
    g[have] = np.log(np.maximum(target[have], 1e-4) / np.maximum(cur[have], 1e-4))
    wgt = have.astype(float)
    adj = m.face_adjacency
    A = sparse.coo_matrix((np.ones(len(adj) * 2), (np.r_[adj[:, 0], adj[:, 1]],
                          np.r_[adj[:, 1], adj[:, 0]])), shape=(len(F), len(F))).tocsr()
    A = A + sparse.identity(len(F))
    num, den = g * wgt[:, None], wgt.copy()
    for _ in range(args.smooth_iters):
        num, den = A @ num, A @ den
        s = den.max()
        num, den = num / s, den / s
    gain = np.exp(num / np.maximum(den, 1e-12)[:, None])
    gain = np.clip(gain, 1 / args.max_gain, args.max_gain)
    print("gain percentiles (lum):", np.round(np.percentile(gain.mean(1), [1, 25, 50, 75, 99]), 2))

    # rasterise per-face gain into atlas space
    gmap = np.zeros((H, W, 3), np.float32)
    painted = np.zeros((H, W), np.uint8)
    pts = np.stack([uvt[..., 0] * W, (1 - uvt[..., 1]) * H], -1)
    pts = np.round(pts * 4).astype(np.int32)  # 2 fractional bits for fillPoly shift
    for i in range(len(F)):
        cv2.fillPoly(gmap, [pts[i]], gain[i].tolist(), lineType=cv2.LINE_8, shift=2)
        cv2.fillPoly(painted, [pts[i]], 1, lineType=cv2.LINE_8, shift=2)
    # gutters / unpainted texels take the gain of the NEAREST painted texel
    _, labels = cv2.distanceTransformWithLabels((painted == 0).astype(np.uint8), cv2.DIST_L2, 3,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    # (painted texels are the zero "sources"; each pixel gets the label of its nearest one)
    ys, xs = np.nonzero(painted)
    lut = np.zeros((labels.max() + 1, 3), np.float32)
    lut[labels[ys, xs]] = gmap[ys, xs]
    gmap = lut[labels]
    lin = np.power(tex, 2.2) * gmap
    out = np.clip(np.power(np.clip(lin, 0, 1), 1 / 2.2) * 255 + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(args.out, out[..., ::-1])
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
