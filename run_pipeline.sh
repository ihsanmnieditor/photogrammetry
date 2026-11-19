#!/usr/bin/env bash
# Full photogrammetry pipeline: video + photos -> textured 3D model -> web/AR viewer.
# CPU-only (no CUDA needed). See README.md for the tools to install.
set -euo pipefail
cd "$(dirname "$0")"

OMVS=${OMVS:-/opt/src/openMVS-2.3/build_cpu/bin}   # OpenMVS v2.3.0 binaries
VIDEO=${VIDEO:-work/video/karipap.mp4}
W=work

# 1. frames: sharpest frame per 1/3 s; skip the seconds where a hand flips the object
python3 scripts/extract_frames.py "$VIDEO" $W/frames --fps 3 --skip 66.3-68.9

# 2. object masks (object was moved/flipped -> only object pixels are rigid)
python3 scripts/make_masks.py images masks
python3 scripts/make_masks.py $W/frames $W/frame_masks $W/frame_mask_preview.jpg 30

# 3. SfM: DISK + LightGlue features on masked crops -> COLMAP incremental mapping
mkdir -p $W/sfm/images $W/sfm/masks
ln -sfn ../../frames $W/sfm/images/video;  ln -sfn ../../../images $W/sfm/images/photos
ln -sfn ../../frame_masks $W/sfm/masks/video; ln -sfn ../../../masks $W/sfm/masks/photos
python3 scripts/sfm_learned.py --images $W/sfm/images --masks $W/sfm/masks --out $W/sfm

# 4. undistort + crop to object, write COLMAP text model for OpenMVS
python3 scripts/prepare_mvs.py --model $W/sfm/sparse --images $W/sfm/images \
    --masks $W/sfm/masks --out $W/mvs

# 5. OpenMVS: dense -> mesh -> clean -> carve (visual hull) -> refine
#    (OpenMVS v2.3.0 + patches/openmvs-2.3-formatTime-overflow.patch)
cd $W/mvs
$OMVS/InterfaceCOLMAP -i . -o scene.mvs --image-folder images
$OMVS/DensifyPointCloud scene.mvs --mask-path masks --resolution-level 0 \
    --max-resolution 4096 --min-resolution 640 --number-views 8 --number-views-fuse 3
$OMVS/ReconstructMesh scene_dense.mvs -p scene_dense.ply -o scene_mesh.ply
cd ../..
python3 scripts/clean_mesh.py $W/mvs/scene_mesh.ply $W/mvs/scene_mesh_clean.ply
python3 scripts/carve_mesh.py $W/mvs/scene_mesh_clean.ply $W/mvs/scene_mesh_carved.ply \
    --mvs $W/mvs --min-out-frac 0.3
cd $W/mvs
$OMVS/RefineMesh scene_dense.mvs -m scene_mesh_carved.ply -o scene_refine.ply \
    --resolution-level 0 --min-resolution 640 --max-views 8 --max-face-area 16 --scales 2 \
    --close-holes 0
cd ../..

# 6. colour consistency across the capture (top vs flipped segment, video vs stills)
python3 scripts/normalize_colors.py --mvs $W/mvs
python3 scripts/normalize_colors.py --calibrate-photos $W/mvs $W/mvs/scene_refine.ply
cp $W/mvs/masks/*.mask.png $W/mvs/images/     # TextureMesh ignores label 0 (background)

# 7. texture (OpenMVS), balance large-scale tone, upscale 2x keeping fine detail
cd $W/mvs
$OMVS/TextureMesh scene_dense.mvs -m scene_refine.ply -o scene_texture.ply \
    --resolution-level 0 --min-resolution 640 --max-texture-size 8192 \
    --empty-color 13222580 --close-holes 50 --ignore-mask-label 0
cd ../..
python3 scripts/balance_texture.py $W/mvs/scene_texture.ply $W/mvs/scene_texture0_bal.png --mvs $W/mvs
python3 scripts/enhance_texture.py $W/mvs/scene_texture0_bal.png $W/mvs/tex_final.png --size 4096

# 8. web + AR exports: GLB (browsers/Android AR), USDZ (iPhone AR), glTF+JSON for hosting
python3 scripts/export_web.py $W/mvs/scene_texture.ply --model $W/mvs/sparse --out viewer \
    --length 0.075 --texture $W/mvs/tex_final.png --max-texture 4096
python3 scripts/build_viewer_variants.py
