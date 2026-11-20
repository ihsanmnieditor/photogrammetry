#!/usr/bin/env python3
"""Build the GitHub Pages site in docs/ from viewer/index.html.

docs/index.html   full HTML document: model-viewer, 360 / zoom / pan, AR
                  (Android: Scene Viewer with karipap.glb, iPhone: Quick Look with karipap.usdz)
                  + Open Graph tags so WhatsApp shows a title, text and thumbnail
docs/karipap.glb, docs/karipap.usdz, docs/preview.jpg (made by scripts/pages_preview.js)

Usage: python scripts/build_pages.py [--base-url https://USER.github.io/REPO/]
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="https://ihsanmnieditor.github.io/photogrammetry/")
a = ap.parse_args()
base = a.base_url.rstrip("/") + "/"

docs = ROOT / "docs"
docs.mkdir(exist_ok=True)
for f in ("karipap.glb", "karipap.usdz"):
    shutil.copy2(ROOT / "viewer" / f, docs / f)
(docs / ".nojekyll").write_text("")

page = (ROOT / "viewer" / "index.html").read_text()
title = "Karipap Arca"
desc = ("Arca tanah liat berbentuk karipap dalam 3D: putar 360°, zoom sampai nampak "
        "cap jari, dan letak atas meja dengan AR.")
og = f'''<meta property="og:type" content="website">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:image" content="{base}preview.jpg">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:url" content="{base}">
<meta name="twitter:card" content="summary_large_image">
'''
page = page.replace("<title>Karipap Arca</title>", "<title>Karipap Arca</title>\n" + og, 1)
page = ('<!doctype html>\n<html lang="ms">\n<head>\n'
        + page.replace("<style>", "<style>\nbody{margin:0}", 1)
              .replace('<main class="stage">', '</head>\n<body>\n<main class="stage">', 1)
        + "\n</body>\n</html>\n")
(docs / "index.html").write_text(page)
print("docs/ built for", base)
