#!/usr/bin/env python3
"""Turn the COLMAP sparse model into an OpenMVS-ready, object-cropped workspace.

For every registered image:
  * undistort (OPENCV model -> PINHOLE with the same K) image and mask,
  * crop both to the object mask bbox (+margin) and shift the principal point,
  * write a classic COLMAP *text* model (one PINHOLE camera per image) that
    OpenMVS's InterfaceCOLMAP reads.
Cropping keeps full resolution on the clay while cutting dense-stereo cost ~5-10x.

Output:
  <out>/images/<name>.jpg
  <out>/masks/<stem>.mask.png      (OpenMVS --mask-path convention, 255 = object)
  <out>/sparse/{cameras,images,points3D}.txt

Usage:
  python scripts/prepare_mvs.py --model work/sfm/sparse --images work/sfm/images \
      --masks work/sfm/masks --out work/mvs [--dilate 6]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pycolmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=0.06)
    ap.add_argument("--dilate", type=int, default=5, help="mask dilation in px")
    ap.add_argument("--only", default="", help="comma-separated name prefixes to keep")
    args = ap.parse_args()

    rec = pycolmap.Reconstruction(args.model)
    out = Path(args.out)
    for d in ("images", "masks", "sparse"):
        (out / d).mkdir(parents=True, exist_ok=True)
    keep_prefix = [p for p in args.only.split(",") if p]

    cams_txt, imgs_txt = [], []
    used_pts = {}
    cam_id = 0
    for img_id in sorted(rec.reg_image_ids()):
        im = rec.image(img_id)
        if keep_prefix and not any(im.name.startswith(p) for p in keep_prefix):
            continue
        cam = rec.camera(im.camera_id)
        prm = list(cam.params)
        if cam.model_name == "OPENCV":
            fx, fy, cx, cy, k1, k2, p1, p2 = prm
        elif cam.model_name == "SIMPLE_RADIAL":
            fx, cx, cy, k1 = prm
            fy, k2, p1, p2 = fx, 0.0, 0.0, 0.0
        elif cam.model_name == "RADIAL":
            fx, cx, cy, k1, k2 = prm
            fy, p1, p2 = fx, 0.0, 0.0
        else:
            raise ValueError(f"unsupported camera model {cam.model_name}")
        K = np.array([[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1]])  # OpenCV convention
        dist = np.array([k1, k2, p1, p2])
        img = cv2.imread(str(Path(args.images) / im.name))
        mpath = Path(args.masks) / (str(Path(im.name).with_suffix("")) + ".png")
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        mx, my = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
        img_u = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        mask_u = cv2.remap(mask, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        if args.dilate > 0:
            k = 2 * args.dilate + 1
            mask_u = cv2.dilate(mask_u, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        ys, xs = np.nonzero(mask_u > 127)
        if len(xs) == 0:
            continue
        pad = int(args.margin * max(np.ptp(xs), np.ptp(ys)))
        x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x1, y1 = min(w, xs.max() + pad + 1), min(h, ys.max() + pad + 1)
        crop, mcrop = img_u[y0:y1, x0:x1], mask_u[y0:y1, x0:x1]

        flat = im.name.replace("/", "__")
        cv2.imwrite(str(out / "images" / flat), crop, [cv2.IMWRITE_JPEG_QUALITY, 97])
        cv2.imwrite(str(out / "masks" / (Path(flat).stem + ".mask.png")),
                    np.where(mcrop > 127, 255, 0).astype(np.uint8))

        cam_id += 1
        cams_txt.append(f"{cam_id} PINHOLE {x1 - x0} {y1 - y0} {fx} {fy} "
                        f"{cx - x0} {cy - y0}")

        # undistort observed keypoints (COLMAP coords -> OpenCV -> undistorted -> crop)
        pts2d = [(p.xy, p.point3D_id) for p in im.points2D if p.has_point3D()]
        line2 = []
        if pts2d:
            xy = np.array([p[0] for p in pts2d], np.float64) - 0.5
            und = cv2.undistortPoints(xy[:, None, :], K, dist, P=K)[:, 0, :] + 0.5
            und -= [x0, y0]
            for (u, v), (_, pid) in zip(und, pts2d):
                if 0 <= u < x1 - x0 and 0 <= v < y1 - y0:
                    idx = len(line2)
                    line2.append(f"{u:.3f} {v:.3f} {pid}")
                    used_pts.setdefault(pid, []).append((img_id, idx))
        q = im.cam_from_world().rotation.quat  # x, y, z, w
        t = im.cam_from_world().translation
        imgs_txt.append(f"{img_id} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} "
                        f"{cam_id} {flat}\n" + " ".join(line2))
        # image ids must match the camera ids order we wrote -> keep mapping
    # rewrite image lines so CAMERA_ID matches (already does), write files
    with open(out / "sparse" / "cameras.txt", "w") as f:
        f.write("# Camera list\n" + "\n".join(cams_txt) + "\n")
    with open(out / "sparse" / "images.txt", "w") as f:
        f.write("# Image list\n" + "\n".join(imgs_txt) + "\n")
    with open(out / "sparse" / "points3D.txt", "w") as f:
        f.write("# 3D point list\n")
        for pid, track in used_pts.items():
            if len(track) < 2:
                continue
            p = rec.point3D(pid)
            c = p.color
            f.write(f"{pid} {p.xyz[0]} {p.xyz[1]} {p.xyz[2]} {c[0]} {c[1]} {c[2]} {p.error} "
                    + " ".join(f"{i} {j}" for i, j in track) + "\n")
    print(f"{cam_id} images, {sum(len(t) >= 2 for t in used_pts.values())} points -> {out}")


if __name__ == "__main__":
    main()
