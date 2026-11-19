#!/usr/bin/env python3
"""Sparse reconstruction with learned features (DISK + LightGlue, via kornia),
fed into COLMAP (pycolmap) for geometric verification and incremental mapping.

The object is masked (scripts/make_masks.py) because it was moved / flipped
between shots, so only object pixels are rigid. Features are extracted on a crop
around the mask, so the object is seen at full resolution.

Image layout (one COLMAP camera per sub-folder):
    <images>/video/*.jpg     frames from the video (time-ordered names)
    <images>/photos/*.jpg    4K stills
    <masks>/<subfolder>/<stem>.png

Pairs: every image vs its +-W temporal neighbours in each folder, plus every
K-th image vs every K-th image (loop closure / other segments), plus all photos
vs every K-th frame.

Usage:
    python scripts/sfm_learned.py --images work/sfm/images --masks work/sfm/masks --out work/sfm
"""
import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
import kornia.feature as KF
import kornia.geometry as KG
from kornia.io import load_image, ImageLoadType


def load_crop(path, mask_path, max_size, margin=0.08):
    """Image cropped to the mask bbox, resized to <= max_size.
    Returns tensor, crop offset, scale (orig px per tensor px), eroded mask."""
    img = load_image(str(path), ImageLoadType.RGB32)[None]
    H, W = img.shape[-2:]
    mask = None
    x0, y0, x1, y1 = 0, 0, W, H
    if mask_path is not None and Path(mask_path).exists():
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        ys, xs = np.nonzero(m)
        pad = int(margin * max(xs.max() - xs.min(), ys.max() - ys.min()))
        x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x1, y1 = min(W, xs.max() + pad), min(H, ys.max() + pad)
        k = max(3, int(0.004 * max(H, W)) | 1)
        mask = cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    img = img[..., y0:y1, x0:x1]
    h, w = img.shape[-2:]
    s = max_size / max(h, w)
    if s < 1:
        img = KG.resize(img, (int(round(h * s)), int(round(w * s))), antialias=True)
    return img, (x0, y0), (w / img.shape[-1], h / img.shape[-2]), mask


def make_pairs(names, window, every):
    groups = defaultdict(list)
    for n in names:
        groups[n.split("/")[0]].append(n)
    pairs = set()
    for g in groups.values():
        g.sort()
        for i in range(len(g)):
            for j in range(i + 1, min(len(g), i + window + 1)):
                pairs.add((g[i], g[j]))
    video = sorted(groups.get("video", []))
    key = video[::every]
    for a, b in itertools.combinations(key, 2):
        pairs.add((a, b))
    for p in groups.get("photos", []):
        for k in key[::2]:
            pairs.add((p, k))
        for q in groups["photos"]:
            if p < q:
                pairs.add((p, q))
    return sorted(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-size", type=int, default=1280)
    ap.add_argument("--num-features", type=int, default=4096)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--focal-photos", type=float, default=3900.0)  # 36mm-eq @ 4096 px
    ap.add_argument("--focal-video", type=float, default=1500.0)   # rough guess, refined in BA
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    root = Path(args.images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "database.db"
    if db_path.exists():
        db_path.unlink()

    # */*.jpg (not rglob) so symlinked sub-folders are followed; only masked images
    names = sorted(str(p.relative_to(root)) for p in root.glob("*/*.jpg")
                   if (Path(args.masks) / p.relative_to(root).with_suffix(".png")).exists())
    print(f"{len(names)} images")

    pycolmap.Database.open(db_path).close()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    pycolmap.import_images(db_path, root, pycolmap.CameraMode.PER_FOLDER,
                           image_names=names, options=reader)

    db = pycolmap.Database.open(db_path)
    id_of = {im.name: im.image_id for im in db.read_all_images()}
    cam_of = {im.name: im.camera_id for im in db.read_all_images()}
    done = set()
    for n, cid in cam_of.items():
        if cid in done:
            continue
        done.add(cid)
        cam = db.read_camera(cid)
        f = args.focal_photos if n.startswith("photos/") else args.focal_video
        cam.params = [f, f, cam.width / 2, cam.height / 2, 0, 0, 0, 0]
        cam.has_prior_focal_length = n.startswith("photos/")
        db.update_camera(cam)
        print(f"camera {cid}: {cam.width}x{cam.height} f0={f} ({n.split('/')[0]})")

    # --- features -------------------------------------------------------------
    disk = KF.DISK.from_pretrained("depth").eval()
    feats = {}
    for n in names:
        mpath = Path(args.masks) / (str(Path(n).with_suffix("")) + ".png")
        img, (ox, oy), (sx, sy), mask = load_crop(root / n, mpath, args.max_size)
        f = disk(img, args.num_features, pad_if_not_divisible=True)[0]
        kp, desc = f.keypoints.clone(), f.descriptors
        kp_full = (kp + 0.5) * torch.tensor([sx, sy]) + torch.tensor([ox, oy])
        if mask is not None:
            xi = kp_full[:, 0].long().clamp(0, mask.shape[1] - 1).numpy()
            yi = kp_full[:, 1].long().clamp(0, mask.shape[0] - 1).numpy()
            keep = torch.from_numpy(mask[yi, xi])
            kp, desc, kp_full = kp[keep], desc[keep], kp_full[keep]
        feats[n] = dict(kp=kp, desc=desc, size=torch.tensor(img.shape[-2:][::-1]))
        db.write_keypoints(id_of[n], kp_full.numpy().astype(np.float32))
    print("features done", flush=True)

    # --- matching ---------------------------------------------------------------
    lg = KF.LightGlue("disk", depth_confidence=0.95, width_confidence=0.99).eval()
    pairs = make_pairs(names, args.window, args.every)
    print(f"{len(pairs)} pairs to match", flush=True)
    pairs_txt = out / "pairs.txt"
    n_written = 0
    with open(pairs_txt, "w") as fp:
        for k, (a, b) in enumerate(pairs):
            fa, fb = feats[a], feats[b]
            res = lg({
                "image0": {"keypoints": fa["kp"][None], "descriptors": fa["desc"][None],
                           "image_size": fa["size"][None]},
                "image1": {"keypoints": fb["kp"][None], "descriptors": fb["desc"][None],
                           "image_size": fb["size"][None]},
            })
            m = res["matches"][0].numpy().astype(np.uint32)
            if len(m) >= 20:
                db.write_matches(id_of[a], id_of[b], m)
                fp.write(f"{a} {b}\n")
                n_written += 1
            if k % 200 == 0:
                print(f"  {k}/{len(pairs)} matched, {n_written} kept", flush=True)
    db.close()

    pycolmap.verify_matches(db_path, pairs_txt)

    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_num_matches = 20
    opts.mapper.abs_pose_min_num_inliers = 20
    opts.ba_refine_principal_point = False
    opts.multiple_models = True
    sparse = out / "sparse"
    sparse.mkdir(exist_ok=True)
    recs = pycolmap.incremental_mapping(db_path, root, sparse, options=opts)
    for k, r in recs.items():
        regs = sorted(im.name for im in r.images.values())
        print(f"model {k}: {r.num_reg_images()} images "
              f"({sum(x.startswith('photos/') for x in regs)} photos), "
              f"{r.num_points3D()} points, err {r.compute_mean_reprojection_error():.3f}px")
        for c in r.cameras.values():
            print("   camera", c.camera_id, c.params[:2])


if __name__ == "__main__":
    main()
