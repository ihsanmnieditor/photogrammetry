#!/usr/bin/env python3
"""Export the textured OpenMVS mesh for the web viewer and AR.

* levels the model: world "up" = average camera up-vector of the frames where the
  phone orbited the object resting on the table (default: video frames < 56 s)
* recentres it so it sits on the floor (y = 0) at the origin
* scales the longest side to --length metres (real size estimate, AR shows 1:1)
* writes  <out>/karipap.glb   (glTF binary, JPEG texture, for browsers / Android AR)
          <out>/karipap.usdz  (for iPhone / iPad Quick Look AR)

Usage:
  python scripts/export_web.py work/mvs/scene_texture.obj --model work/mvs/sparse \
      --out viewer [--length 0.075] [--max-texture 8192]
"""
import argparse
import io
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def quat_to_R(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def world_up_from_cameras(images_txt, name_re, max_ms):
    ups = []
    lines = [l for l in open(images_txt) if not l.startswith("#")]
    for l in lines[0::2]:
        p = l.split()
        if len(p) < 10:
            continue
        m = re.search(name_re, p[9])
        if not m or int(m.group(1)) > max_ms:
            continue
        R = quat_to_R(*map(float, p[1:5]))  # cam_from_world
        ups.append(-R[1, :])  # camera -Y axis expressed in world coords
    up = np.mean(ups, axis=0)
    print(f"world up from {len(ups)} cameras: {up / np.linalg.norm(up)} "
          f"(consistency {np.linalg.norm(up):.3f})")
    return up / np.linalg.norm(up)


def rotation_to(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), np.dot(a, b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def write_usdz(path, V, F, N, UV, tex_jpg_bytes):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt
    tmp = Path(tempfile.mkdtemp())
    (tmp / "texture.jpg").write_bytes(tex_jpg_bytes)
    usda = tmp / "karipap.usdc"
    stage = Usd.Stage.CreateNew(str(usda))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Karipap")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Karipap/Mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.reshape(-1).astype(np.int32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(N.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    lo, hi = V.min(0), V.max(0)
    mesh.CreateExtentAttr([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                 UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(UV.astype(np.float32)))

    mat = UsdShade.Material.Define(stage, "/Karipap/Clay")
    pbr = UsdShade.Shader.Define(stage, "/Karipap/Clay/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, "/Karipap/Clay/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, "/Karipap/Clay/diffuse")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("texture.jpg")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    UsdUtils.CreateNewARKitUsdzPackage(Sdf.AssetPath(str(usda)), str(path))
    shutil.rmtree(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--model", required=True, help="dir with images.txt (prepare_mvs output)")
    ap.add_argument("--out", default="viewer")
    ap.add_argument("--length", type=float, default=0.075, help="longest side in metres")
    ap.add_argument("--max-texture", type=int, default=8192)
    ap.add_argument("--up-name-re", default=r"video__f_(\d+)")
    ap.add_argument("--up-max-ms", type=int, default=56000)
    ap.add_argument("--flip-up", action="store_true")
    ap.add_argument("--texture", help="use this image instead of the mesh's own texture "
                                      "(e.g. the upscaled atlas from enhance_texture.py)")
    ap.add_argument("--lit", action="store_true", help="PBR-lit material instead of unlit")
    args = ap.parse_args()
    global UNLIT
    UNLIT = not args.lit

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(args.mesh, process=False, force="mesh")
    print(f"loaded {len(mesh.vertices)} v / {len(mesh.faces)} f")

    up = world_up_from_cameras(Path(args.model) / "images.txt", args.up_name_re, args.up_max_ms)
    if args.flip_up:
        up = -up
    R = rotation_to(up, np.array([0.0, 1.0, 0.0]))
    V = (R @ np.asarray(mesh.vertices).T).T
    # yaw: put the longest horizontal extent along X
    xz = V[:, [0, 2]] - V[:, [0, 2]].mean(0)
    w, vec = np.linalg.eigh(np.cov(xz.T))
    a = vec[:, np.argmax(w)]
    yaw = np.arctan2(a[1], a[0])
    c, s = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    V = (Ry @ V.T).T
    lo, hi = V.min(0), V.max(0)
    scale = args.length / (hi - lo).max()
    V = (V - [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]) * scale
    mesh.vertices = V
    dims = V.max(0) - V.min(0)
    print(f"size (m): L={dims[0]:.4f} H={dims[1]:.4f} W={dims[2]:.4f}")

    mat = mesh.visual.material
    img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
    if args.texture:
        img = Image.open(args.texture)
    img = img.convert("RGB")
    V = np.asarray(mesh.vertices, np.float32)
    F = np.asarray(mesh.faces, np.uint32)
    N = np.asarray(mesh.vertex_normals, np.float32)
    UV = np.asarray(mesh.visual.uv, np.float32)

    def jpeg(size, q):
        im = img.copy()
        if max(im.size) > size:
            im.thumbnail((size, size), Image.LANCZOS)
        b = io.BytesIO()
        im.save(b, "JPEG", quality=q, optimize=True, progressive=True)
        return b.getvalue(), im.size

    # 1) web viewer: .gltf + .bin + full-res texture as separate files (each < 15 MB)
    hi_jpg, hi_size = jpeg(args.max_texture, 88)
    (out / "karipap_texture.jpg").write_bytes(hi_jpg)
    write_gltf(out / "karipap.gltf", out / "karipap.bin", V, F, N, UV, "karipap_texture.jpg")
    print(f"karipap.gltf/.bin {(out / 'karipap.bin').stat().st_size / 1e6:.1f} MB, "
          f"texture {hi_size} {len(hi_jpg) / 1e6:.1f} MB")

    # 2) single-file GLB (AR on Android, offline HTML) with a 4K texture
    lo_jpg, lo_size = jpeg(4096, 88)
    write_glb(out / "karipap.glb", V, F, N, UV, lo_jpg)
    print(f"karipap.glb {(out / 'karipap.glb').stat().st_size / 1e6:.1f} MB (texture {lo_size})")

    # 3) USDZ for iPhone / iPad Quick Look AR, same 4K texture
    write_usdz(out / "karipap.usdz", V, F, N, UV, lo_jpg)
    print(f"karipap.usdz {(out / 'karipap.usdz').stat().st_size / 1e6:.1f} MB")


UNLIT = True


def _gltf_json(V, F, N, UV, image):
    import json
    nv = len(V)
    views, accessors = [], []
    offset = 0
    for arr, target, typ, comp in ((V, 34962, "VEC3", 5126), (N, 34962, "VEC3", 5126),
                                   (UV, 34962, "VEC2", 5126), (F, 34963, "SCALAR", 5125)):
        nbytes = arr.nbytes
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": nbytes, "target": target})
        acc = {"bufferView": len(views) - 1, "componentType": comp,
               "count": nv if typ != "SCALAR" else int(F.size), "type": typ}
        if arr is V:
            acc["min"], acc["max"] = V.min(0).tolist(), V.max(0).tolist()
        accessors.append(acc)
        offset += nbytes
    doc = {
        "asset": {"version": "2.0", "generator": "photogrammetry/export_web.py"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Karipap"}],
        "meshes": [{"name": "Karipap", "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3, "material": 0}]}],
        "materials": [{"name": "Clay", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.85},
            # photogrammetry texture already contains the real light & shadows:
            # show it as captured (viewer falls back to lit PBR if unsupported)
            **({"extensions": {"KHR_materials_unlit": {}}} if UNLIT else {})}],
        **({"extensionsUsed": ["KHR_materials_unlit"]} if UNLIT else {}),
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [image],
        "accessors": accessors, "bufferViews": views,
        "buffers": [{"byteLength": offset}],
    }
    return doc, offset, json


def _gltf_bin(V, F, N, UV):
    uv = UV.copy()
    uv[:, 1] = 1.0 - uv[:, 1]  # OBJ (v up) -> glTF (v down)
    return b"".join(a.astype(a.dtype).tobytes() for a in (V, N, uv.astype(np.float32), F))


def write_gltf(gltf_path, bin_path, V, F, N, UV, image_uri):
    doc, n, json = _gltf_json(V, F, N, UV, {"uri": image_uri, "mimeType": "image/jpeg"})
    doc["buffers"][0]["uri"] = Path(bin_path).name
    Path(bin_path).write_bytes(_gltf_bin(V, F, N, UV))
    Path(gltf_path).write_text(json.dumps(doc))


def write_glb(path, V, F, N, UV, jpg):
    import struct
    binary = _gltf_bin(V, F, N, UV)
    pad = (-len(binary)) % 4
    img_off = len(binary) + pad
    doc, n, json = _gltf_json(V, F, N, UV, {"bufferView": 4, "mimeType": "image/jpeg"})
    doc["bufferViews"].append({"buffer": 0, "byteOffset": img_off, "byteLength": len(jpg)})
    blob = binary + b"\0" * pad + jpg
    blob += b"\0" * ((-len(blob)) % 4)
    doc["buffers"][0]["byteLength"] = len(blob)
    js = json.dumps(doc).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + blob)


if __name__ == "__main__":
    main()
