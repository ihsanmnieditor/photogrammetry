#!/usr/bin/env python3
"""Upscale the texture atlas so close-ups stay crisp.

Default method "lanczos": 2x Lanczos + light unsharp mask. Keeps the real captured
fingerprint ridges and makes them crisper. (Tested "esrgan", Real-ESRGAN x4plus:
it smooths fingerprint ridges away as if they were noise, so it is NOT the default.)

Weights: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
Usage: python scripts/enhance_texture.py in.png out.png --size 4096 [--method esrgan --weights RealESRGAN_x4plus.pth]
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--method", choices=["lanczos", "esrgan"], default="lanczos")
    ap.add_argument("--weights", help="RealESRGAN_x4plus.pth (only for --method esrgan)")
    ap.add_argument("--unsharp", type=int, default=60, help="unsharp percent (lanczos)")
    ap.add_argument("--size", type=int, default=4096, help="output texture size")
    ap.add_argument("--tile", type=int, default=192)
    ap.add_argument("--pad", type=int, default=12)
    args = ap.parse_args()

    if args.method == "lanczos":
        from PIL import ImageFilter
        im = Image.open(args.inp).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
        im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=args.unsharp, threshold=2))
        im.save(args.out)
        print(f"saved {args.out} {im.size}")
        return

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    net = RRDBNet()
    sd = torch.load(args.weights, map_location="cpu", weights_only=True)
    net.load_state_dict(sd.get("params_ema", sd.get("params", sd)), strict=True)
    net.eval()

    img = np.asarray(Image.open(args.inp).convert("RGB"), np.float32) / 255.0
    H, W, _ = img.shape
    out = np.zeros((H * 4, W * 4, 3), np.float32)
    empty = img[-1, -1]  # atlas background (empty-color)
    t, p = args.tile, args.pad
    tiles = [(y, x) for y in range(0, H, t) for x in range(0, W, t)]
    for k, (y, x) in enumerate(tiles):
        y0, x0 = max(0, y - p), max(0, x - p)
        y1, x1 = min(H, y + t + p), min(W, x + t + p)
        patch = img[y0:y1, x0:x1]
        if np.abs(patch - empty).max() < 2 / 255:  # unused atlas area
            out[y * 4:(y + t) * 4, x * 4:(x + t) * 4] = empty
            continue
        inp = torch.from_numpy(patch).permute(2, 0, 1)[None]
        sr = net(inp)[0].permute(1, 2, 0).clamp(0, 1).numpy()
        oy, ox = (y - y0) * 4, (x - x0) * 4
        hh, ww = min(t, H - y) * 4, min(t, W - x) * 4
        out[y * 4:y * 4 + hh, x * 4:x * 4 + ww] = sr[oy:oy + hh, ox:ox + ww]
        if k % 20 == 0:
            print(f"tile {k}/{len(tiles)}", flush=True)
    res = Image.fromarray((out * 255 + 0.5).astype(np.uint8))
    if res.size[0] != args.size:
        res = res.resize((args.size, args.size), Image.LANCZOS)
    res.save(args.out)
    print(f"saved {args.out} {res.size}")


if __name__ == "__main__":
    main()
