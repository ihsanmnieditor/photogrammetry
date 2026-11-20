#!/usr/bin/env python3
"""Write viewer/karipap_mesh.json for the three.js viewer (viewer/artifact.html).

Hosts with a strict Content-Security-Policy (connect-src 'self') refuse fetches of
data:/blob: URLs, so a glTF with an embedded buffer never loads there. This file is
plain JSON fetched by relative URL; the arrays are base64 strings decoded in the page:
  positions uint16 (quantised to the bounding box), uvs uint16, indices uint32.
Normals are not needed: the material is unlit (texture = captured light & shadows).
"""
import base64
import json
from pathlib import Path

import numpy as np
import trimesh

V = Path(__file__).resolve().parent.parent / "viewer"
scene = trimesh.load(V / "karipap.glb")
m = list(scene.geometry.values())[0]
pos = np.asarray(m.vertices, np.float64)
uv = np.asarray(m.visual.uv, np.float64)  # v-up (OBJ convention) = three.js with flipY
lo, hi = pos.min(0), pos.max(0)
q = np.round((pos - lo) / (hi - lo) * 65535).astype(np.uint16)
quv = np.round(np.clip(uv, 0, 1) * 65535).astype(np.uint16)
idx = np.asarray(m.faces, np.uint32)
err = np.abs(q / 65535 * (hi - lo) + lo - pos).max()
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
doc = {"format": "karipap-mesh-1", "vertexCount": len(pos), "faceCount": len(idx),
       "bboxMin": lo.tolist(), "bboxMax": hi.tolist(),
       "positions": b64(q), "uvs": b64(quv), "indices": b64(idx),
       "texture": "karipap_texture.jpg"}
(V / "karipap_mesh.json").write_text(json.dumps(doc))
print(f"{len(pos)} v / {len(idx)} f, max quantisation error {err * 1000:.4f} mm, "
      f"{(V / 'karipap_mesh.json').stat().st_size / 1e6:.1f} MB")
