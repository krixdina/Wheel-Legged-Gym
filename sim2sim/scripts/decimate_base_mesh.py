"""Decimate base_link.STL so the real chassis shape fits MuJoCo's mesh limit.

base_link.STL has ~1.06M triangles, which exceeds MuJoCo's hard cap of 200000
faces per mesh. A box stand-in loses the real form needed for demo videos, so
instead we run a quadric edge-collapse decimation (fast-simplification backend
via trimesh) down to a face count well under the cap. At ~80k faces the chassis
is visually indistinguishable from the original in rendered video.

This is a one-off asset-preparation step. It writes a watertight OBJ next to the
other sim2sim assets; the runtime (isaac_gym env) only needs to load that OBJ
and does not need trimesh / fast-simplification.

Run with the isaac_lab env, which has trimesh + fast-simplification:
    conda run -n isaac_lab python sim2sim/scripts/decimate_base_mesh.py
"""
import os

import trimesh

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO_ROOT, "resources/robots/wheel_legged_fyt/meshes/base_link.STL")
OUT_DIR = os.path.join(REPO_ROOT, "sim2sim/model/meshes")
OUT = os.path.join(OUT_DIR, "base_link_decimated.obj")

# Target face count: comfortably under MuJoCo's 200k cap while preserving the
# chassis silhouette for video. Lower if MuJoCo compile time matters more.
TARGET_FACES = 80000


def main():
    mesh = trimesh.load(SRC)
    print(f"source: {len(mesh.faces)} faces, {len(mesh.vertices)} verts")

    simplified = mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
    print(f"decimated: {len(simplified.faces)} faces, {len(simplified.vertices)} verts")

    # Sanity: bounding box must stay the same so the chassis keeps its real size.
    src_ext = mesh.bounding_box.extents
    dec_ext = simplified.bounding_box.extents
    print(f"bbox extents  src={src_ext.round(4)}  decimated={dec_ext.round(4)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    simplified.export(OUT)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
