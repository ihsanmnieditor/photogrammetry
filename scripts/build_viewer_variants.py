#!/usr/bin/env python3
"""Build the two extra viewer pages from viewer/index.html:

* viewer/karipap_mesh.json for viewer/artifact.html (hand-written three.js page):
  see scripts/build_artifact_mesh.py. (A glTF with a data: URI buffer does NOT load
  on hosts whose CSP has connect-src 'self', which is why artifact.html is not a
  model-viewer page.)
* viewer/karipap-arca-offline.html: one self-contained file with the GLB embedded
  (needs internet only for the model-viewer script).
"""
import base64
from pathlib import Path

V = Path(__file__).resolve().parent.parent / "viewer"
page = (V / "index.html").read_text()

import subprocess, sys  # noqa: E401
subprocess.run([sys.executable, str(Path(__file__).parent / "build_artifact_mesh.py")], check=True)

glb = base64.b64encode((V / "karipap.glb").read_bytes()).decode()
off = page.replace('src="karipap.glb"', f'src="data:model/gltf-binary;base64,{glb}"') \
          .replace('    ios-src="karipap.usdz"\n', "")
off = ('<!doctype html>\n<html lang="ms">\n<head>\n'
       + off.replace("<style>", "<style>\nbody{margin:0}", 1)
            .replace('<main class="stage">', '</head>\n<body>\n<main class="stage">', 1)
       + "\n</body>\n</html>\n")
(V / "karipap-arca-offline.html").write_text(off)
print("built karipap_mesh.json, karipap-arca-offline.html")
