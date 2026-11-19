#!/usr/bin/env python3
"""Clean the OpenMVS mesh before refinement / texturing.

* keep only the largest connected piece (drops floaters / reflection blobs)
* remove non-manifold bits, close holes (areas no camera saw, e.g. contact patch)
* light smoothing on the filled patches only is left to RefineMesh
* optional decimation to a target face count

Usage: python scripts/clean_mesh.py in.ply out.ply [--faces 1500000] [--hole 100000]
"""
import argparse

import pymeshlab as ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--faces", type=int, default=0, help="decimate to this many faces (0 = keep)")
    ap.add_argument("--hole", type=int, default=100000, help="max hole size (edges) to close")
    args = ap.parse_args()

    ms = ml.MeshSet()
    ms.load_new_mesh(args.inp)
    print("in :", ms.current_mesh().vertex_number(), "v", ms.current_mesh().face_number(), "f")
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
    # largest component only
    ms.meshing_remove_connected_component_by_diameter(
        mincomponentdiag=ml.PercentageValue(25), removeunref=True)
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_remove_unreferenced_vertices()
    if args.faces and ms.current_mesh().face_number() > args.faces:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=args.faces, preservenormal=True, preserveboundary=True,
            planarquadric=True, autoclean=True)
    try:
        ms.meshing_close_holes(maxholesize=args.hole, newfaceselected=False,
                               selfintersection=False, refinehole=True)
    except Exception as e:  # very large / complex holes
        print("close_holes failed:", e)
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_remove_unreferenced_vertices()
    m = ms.current_mesh()
    print("out:", m.vertex_number(), "v", m.face_number(), "f")
    ms.save_current_mesh(args.out, save_vertex_color=False, save_vertex_normal=False)


if __name__ == "__main__":
    main()
