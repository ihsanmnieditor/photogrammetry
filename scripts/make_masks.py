#!/usr/bin/env python3
"""Segment the clay object from each photo (white object on black tablet).

The photos look like the object/tablet was rotated between shots (turntable style),
so the background (counter, room-light bokeh, mirror reflections) is NOT rigid
with the object. We mask everything except the object so that SfM and MVS only
use object pixels.

Method: IS-Net salient-object segmentation (rembg), then remove dark pixels
(the black mirror reflection under the object) and keep the largest blob.

Outputs:
  masks/<name>.png          binary mask (255 = object), full resolution
  work/mask_preview.jpg     contact sheet for visual checking
"""
import sys
from pathlib import Path

import cv2
import numpy as np


_SESSION = None


def segment(img):
    """rembg (IS-Net) saliency mask -> largest blob -> drop mirror reflection."""
    global _SESSION
    from rembg import new_session, remove
    if _SESSION is None:
        _SESSION = new_session("isnet-general-use")
    h, w = img.shape[:2]
    s = 1024 / max(h, w)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    alpha = remove(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), session=_SESSION, only_mask=True)
    alpha = np.asarray(alpha)
    m = (alpha > 127).astype(np.uint8) * 255
    # the mirror reflection on the black tablet is dark: drop very dark pixels
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    m[gray < 60] = 0
    # warm bokeh discs from room lights are saturated; the clay is not
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    m[(hsv[..., 1] > 100) & (hsv[..., 2] > 140)] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        raise RuntimeError("no object found")
    m = np.where(lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = cv2.drawContours(np.zeros_like(m), cnts, -1, 255, cv2.FILLED)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 127


def main(img_dir="images", out_dir="masks", preview="work/mask_preview.jpg", max_tiles="24"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    tiles = []
    for p in sorted(Path(img_dir).glob("*.jpg")):
        img = cv2.imread(str(p))
        if (out / f"{p.stem}.png").exists():
            m = cv2.imread(str(out / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
        else:
            try:
                m = segment(img)
            except RuntimeError as e:
                print(p.name, "SKIPPED:", e, flush=True)
                continue
        mm = (m * 255).astype(np.uint8)
        cv2.imwrite(str(out / f"{p.stem}.png"), mm)
        vis = img.copy()
        vis[~m] = (vis[~m] * 0.25).astype(np.uint8)
        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 6)
        cv2.putText(vis, p.name, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
        tiles.append(cv2.resize(vis, (640, 360)))
        print(p.name, f"{m.mean() * 100:.1f}% object", flush=True)
    if len(tiles) > int(max_tiles):
        step = len(tiles) / int(max_tiles)
        tiles = [tiles[int(i * step)] for i in range(int(max_tiles))]
    cols = 3 if len(tiles) <= 9 else 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    Path(preview).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(preview, np.vstack(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
