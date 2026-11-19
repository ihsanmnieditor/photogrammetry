#!/usr/bin/env python3
"""Pick the sharpest frame in every window of a video (default 3 per second).

Usage:
    python scripts/extract_frames.py video.mp4 frames/ --fps 3 --skip 66.5-68.8
    (--skip takes comma-separated start-end second ranges to drop, e.g. hand in view)
"""
import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--skip", default="", help="ranges in seconds, e.g. '66.5-68.8,80-81'")
    args = ap.parse_args()

    skips = [tuple(map(float, r.split("-"))) for r in args.skip.split(",") if r]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    win = max(1, round(vfps / args.fps))
    best, best_s, idx, kept = None, -1.0, 0, 0

    def flush():
        nonlocal best, best_s, kept
        if best is not None:
            t, frame = best
            cv2.imwrite(str(out / f"f_{t * 1000:06.0f}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            kept += 1
        best, best_s = None, -1.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / vfps
        if idx % win == 0:
            flush()
        idx += 1
        if any(a <= t <= b for a, b in skips):
            continue
        g = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
        s = cv2.Laplacian(g, cv2.CV_64F).var()
        if s > best_s:
            best, best_s = (t, frame), s
    flush()
    print(f"{idx} frames read, {kept} kept -> {out}")


if __name__ == "__main__":
    main()
