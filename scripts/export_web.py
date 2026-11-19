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
    args = ap.parse_args()

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

    # texture: limit size, re-encode JPEG
    mat = mesh.visual.material
    img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
    img = img.convert("RGB")
    if max(img.size) > args.max_texture:
        img.thumbnail((args.max_texture, args.max_texture), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90, optimize=True)
    jpg = buf.getvalue()
    print(f"texture {img.size}, {len(jpg) / 1e6:.1f} MB")

    pbr = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.open(io.BytesIO(jpg)), metallicFactor=0.0, roughnessFactor=0.85,
        doubleSided=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=pbr)
    glb = mesh.export(file_type="glb")
    (out / "karipap.glb").write_bytes(glb)
    print(f"karipap.glb {len(glb) / 1e6:.1f} MB")

    write_usdz(out / "karipap.usdz", np.asarray(mesh.vertices), np.asarray(mesh.faces),
               np.asarray(mesh.vertex_normals), np.asarray(mesh.visual.uv), jpg)
    print(f"karipap.usdz {(out / 'karipap.usdz').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
