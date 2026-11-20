# Karipap Arca: photogrammetry → 3D / AR viewer

A clay *karipap* sculpture reconstructed in 3D from a 90 s phone video and 9 still
photos, using **COLMAP + OpenMVS**. The result is shown in a web page where people can
rotate it 360°, zoom, pan and place it in their room with **AR**.

```
viewer/index.html                 web viewer (model-viewer): rotate / zoom / pan / AR / fingerprint close-up
viewer/karipap.glb                textured 3D model, 4K texture (browsers, Android AR, Blender…)
viewer/karipap.usdz               same model for iPhone / iPad AR (Quick Look)
viewer/karipap-arca-offline.html  single-file version with the model embedded (desktop / offline)
viewer/artifact.html + karipap_mesh.json + karipap_texture.jpg   three.js variant for strict-CSP hosts (the shared link)
```

**Result:** 110k vertices / 221k faces, one solid piece (unseen areas closed), 4096² texture; ~7.5 × 3.1 × 2.0 cm
(real size is an estimate: nothing of known size was in the shots).

## How it was made

| Step | Tool | Notes |
|---|---|---|
| Frames | `scripts/extract_frames.py` | sharpest frame per 1/3 s → 262 frames; skips 66–69 s where a hand flips the object |
| Masks | `scripts/make_masks.py` (rembg IS-Net) | the object was **moved and flipped** during capture, so only object pixels are rigid; the mirror-like tablet reflection and bokeh are cut out |
| SfM | `scripts/sfm_learned.py` (DISK + LightGlue → **COLMAP**/pycolmap) | features on mask-cropped images; video mapped first, then the 4K photos registered into it. 270/271 images registered |
| MVS prep | `scripts/prepare_mvs.py` | undistort, crop every image to the object, shift principal points → ~5–10× less dense-stereo work |
| Dense, mesh, refine, texture | **OpenMVS v2.3** | `DensifyPointCloud --mask-path`, `ReconstructMesh`, `RefineMesh`, `TextureMesh` (2048² atlas at the cameras' native resolution, masks exclude background pixels) |
| Clean + carve | `scripts/clean_mesh.py`, `scripts/carve_mesh.py` | drop floaters; remove vertices outside the object masks in ≥30 % of views (shadow / reflection skirts); close holes so the model is solid |
| Colour | `scripts/normalize_colors.py` | per-image exposure/white-balance to a common clay colour; 4K stills cross-calibrated to the video |
| Texture tone | `scripts/balance_texture.py` | smooth multi-view colour correction (global seam leveling had darkened the underside) |
| Texture detail | `scripts/enhance_texture.py` | 2× Lanczos + unsharp. Real-ESRGAN was tested and rejected: it erased fingerprint ridges |
| Export | `scripts/export_web.py`, `scripts/build_viewer_variants.py` | levels the model (camera "up"), sits it on the floor, scales to size; GLB + USDZ + glTF; material is *unlit* so the texture shows the real captured light and shadows |

Everything runs on CPU. `run_pipeline.sh` has the full sequence.

### Why not plain SIFT on the 9 photos?

With only 9 photos ~40° apart, a low-texture clay surface, a mirror-like tablet and
the object being turned between shots, plain COLMAP SIFT registered only 2 images.
The video (many overlapping views) plus object masks plus learned features fixed that.

### Setup notes

* COLMAP 3.9 from apt (CLI), `pycolmap` 4.x from PyPI, PyTorch CPU + kornia, rembg,
  pymeshlab, trimesh, usd-core.
* SfM detail: the video was shot zoomed in (f ≈ 4100 px at 1920 px wide), and a
  wrong focal prior plus OPENCV's separate fx/fy broke mapping; SIMPLE_RADIAL with a
  video-only first stage fixed it.
* OpenMVS **v2.3.0** built from source with `-DOpenMVS_USE_CUDA=OFF`; apply
  `patches/openmvs-2.3-formatTime-overflow.patch` (an `snprintf` size bug that aborts
  `DensifyPointCloud` on Ubuntu's fortified glibc).
* The source video is not committed (228 MB, over GitHub's file limit).

## Sharing on WhatsApp

Send a **link** to the viewer page, not the `.html` file. WhatsApp opens `.html`
attachments as plain files on most phones (no 3D). A link opens in the phone's
browser, where the model can be rotated, zoomed and viewed in AR.
