#!/usr/bin/env python3
"""Build the two extra viewer pages from viewer/index.html:

* viewer/artifact.html + viewer/karipap.json: for hosts that only serve web types
  (.json/.jpg, not .glb/.usdz). glTF is JSON, so the geometry buffer is embedded as
  a data URI and the texture stays a separate .jpg. iPhone AR: model-viewer builds
  the USDZ on the fly when no ios-src is given.
* viewer/karipap-arca-offline.html: one self-contained file with the GLB embedded
  (needs internet only for the model-viewer script).
"""
import base64
import json
from pathlib import Path

V = Path(__file__).resolve().parent.parent / "viewer"
page = (V / "index.html").read_text()

d = json.loads((V / "karipap.gltf").read_text())
d["buffers"][0]["uri"] = "data:application/octet-stream;base64," + \
    base64.b64encode((V / "karipap.bin").read_bytes()).decode()
(V / "karipap.json").write_text(json.dumps(d))
(V / "artifact.html").write_text(
    page.replace('src="karipap.glb"', 'src="karipap.json"').replace('    ios-src="karipap.usdz"\n', ""))

glb = base64.b64encode((V / "karipap.glb").read_bytes()).decode()
off = page.replace('src="karipap.glb"', f'src="data:model/gltf-binary;base64,{glb}"') \
          .replace('    ios-src="karipap.usdz"\n', "")
off = ('<!doctype html>\n<html lang="ms">\n<head>\n'
       + off.replace("<style>", "<style>\nbody{margin:0}", 1)
            .replace('<main class="stage">', '</head>\n<body>\n<main class="stage">', 1)
       + "\n</body>\n</html>\n")
(V / "karipap-arca-offline.html").write_text(off)
print("built artifact.html, karipap.json, karipap-arca-offline.html")
