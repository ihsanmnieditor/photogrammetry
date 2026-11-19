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

# 5. OpenMVS: dense -> mesh -> clean/close holes -> refine -> texture
cd $W/mvs
$OMVS/InterfaceCOLMAP -i . -o scene.mvs --image-folder images
$OMVS/DensifyPointCloud scene.mvs --mask-path masks --resolution-level 0 \
    --max-resolution 4096 --number-views 8 --number-views-fuse 3 --remove-dmaps 0
$OMVS/ReconstructMesh scene_dense.mvs -p scene_dense.ply -o scene_mesh.ply
python3 ../../scripts/clean_mesh.py scene_mesh.ply scene_mesh_clean.ply
$OMVS/RefineMesh scene_dense.mvs -m scene_mesh_clean.ply -o scene_refine.ply \
    --resolution-level 0 --max-face-area 8 --scales 2
$OMVS/TextureMesh scene_dense.mvs -m scene_refine.ply -o scene_texture.obj \
    --export-type obj --resolution-level 0 --max-texture-size 8192 \
    --empty-color 12698551 --cost-smoothness-ratio 0.1
cd ../..

# 6. web + AR exports (GLB for browsers/Android, USDZ for iPhone)
python3 scripts/export_web.py $W/mvs/scene_texture.obj --model $W/mvs/sparse --out viewer
