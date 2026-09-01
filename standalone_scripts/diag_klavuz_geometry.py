"""Diagnostic: compare geometry of the disputed Klavuz face band (274-361) against
the pseudo-ground-truth Thread band (113-270) and the Stock remainder.

Answers one question: is the region where models B/C/F predict Thread and Model A
predicts Stock geometrically indistinguishable from the accepted Thread faces?
"""

import json
import sys
from collections import Counter

import numpy as np

JSON_PATH = r"\\Gr-sw66464\d\Demo\grab_cad_brepmfr_testing\jsons\Klavuz_101.json"


def load_faces(path):
    with open(path, "r") as fh:
        data = json.load(fh)
    faces = data["faces"]
    faces = sorted(faces, key=lambda f: int(f["id"]))
    return data, faces


def face_stats(face):
    uv = np.asarray(face["uv"], dtype=np.float64).reshape(5, 5, 7)
    mask = uv[:, :, 6] > 0.5
    pts = uv[:, :, 0:3][mask] if mask.any() else uv[:, :, 0:3].reshape(-1, 3)
    nrm = uv[:, :, 3:6][mask] if mask.any() else uv[:, :, 3:6].reshape(-1, 3)
    return {
        "type": int(face["z"]),
        "area": float(face["y"]),
        "loop": int(face["l"]),
        "ndeg": int(face.get("a", 0) or 0),
        "centroid": pts.mean(axis=0),
        "pts": pts,
        "nrm": nrm,
    }


def summarize(name, idxs, stats, axis, origin):
    if not idxs:
        print(f"{name}: empty")
        return
    areas = np.array([stats[i]["area"] for i in idxs])
    types = Counter(stats[i]["type"] for i in idxs)
    loops = Counter(stats[i]["loop"] for i in idxs)
    degs = np.array([stats[i]["ndeg"] for i in idxs])

    allpts = np.vstack([stats[i]["pts"] for i in idxs])
    rel = allpts - origin
    along = rel @ axis
    radial = np.linalg.norm(rel - np.outer(along, axis), axis=1)

    print(f"\n--- {name}  (n={len(idxs)}) ---")
    print(f"  face_type histogram      : {dict(sorted(types.items()))}")
    print(f"  loop-count histogram     : {dict(sorted(loops.items()))}")
    print(f"  area   median/mean/min/max: {np.median(areas):.4f} / {areas.mean():.4f} / {areas.min():.4f} / {areas.max():.4f}")
    print(f"  degree median/min/max    : {np.median(degs):.1f} / {degs.min()} / {degs.max()}")
    print(f"  axial  extent (min..max) : {along.min():.3f} .. {along.max():.3f}")
    print(f"  radial dist  med/min/max : {np.median(radial):.3f} / {radial.min():.3f} / {radial.max():.3f}")


def main():
    data, faces = load_faces(JSON_PATH)
    n = len(faces)
    print(f"Loaded {n} faces from {JSON_PATH}")
    print(f"top-level JSON keys: {list(data.keys())}")

    stats = [face_stats(f) for f in faces]

    # Whole-part scale / placement, from all UV points.
    allpts = np.vstack([s["pts"] for s in stats])
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    print("\n=== PART SCALE / PLACEMENT (raw JSON units, no normalization in pipeline) ===")
    print(f"  bbox min      : {np.round(lo, 3)}")
    print(f"  bbox max      : {np.round(hi, 3)}")
    print(f"  bbox diagonal : {np.linalg.norm(hi - lo):.3f}")
    print(f"  bbox size     : {np.round(hi - lo, 3)}")
    print(f"  centroid      : {np.round(allpts.mean(axis=0), 3)}")
    print(f"  dist of centroid from origin: {np.linalg.norm(allpts.mean(axis=0)):.3f}")

    # Estimate part axis as principal direction of the point cloud.
    c = allpts.mean(axis=0)
    u, s, vt = np.linalg.svd(allpts - c, full_matrices=False)
    axis = vt[0]
    print(f"  principal axis: {np.round(axis, 3)}  (singular values {np.round(s[:3], 1)})")

    gt_thread = list(range(113, 192)) + list(range(193, 271)) + [273] + list(range(362, 365))
    disputed = list(range(274, 362))
    other_stock = [
        i for i in range(n)
        if i not in set(gt_thread) and i not in set(disputed)
    ]

    summarize("PSEUDO-GT THREAD  113-191, 193-270, 273, 362-364", gt_thread, stats, axis, c)
    summarize("DISPUTED BAND     274-361 (A=Stock, B/C/F=Thread)", disputed, stats, axis, c)
    summarize("REMAINING FACES   (stock + text)", other_stock, stats, axis, c)

    # Segment-level view: consecutive runs of the thread-like index space.
    print("\n=== per-segment detail over face index 100..380 ===")
    for lo_i, hi_i, tag in [
        (113, 191, "GT thread seg 1"),
        (193, 270, "GT thread seg 2"),
        (274, 361, "DISPUTED seg 3"),
        (362, 364, "GT thread seg 4"),
    ]:
        idxs = list(range(lo_i, hi_i + 1))
        areas = np.array([stats[i]["area"] for i in idxs])
        types = Counter(stats[i]["type"] for i in idxs)
        allp = np.vstack([stats[i]["pts"] for i in idxs])
        rel = allp - c
        along = rel @ axis
        radial = np.linalg.norm(rel - np.outer(along, axis), axis=1)
        print(
            f"  {tag:16s} idx {lo_i:3d}-{hi_i:3d} n={len(idxs):3d} "
            f"types={dict(sorted(types.items()))} "
            f"area_med={np.median(areas):.4f} "
            f"axial={along.min():7.2f}..{along.max():7.2f} "
            f"radial_med={np.median(radial):.2f}"
        )


if __name__ == "__main__":
    sys.exit(main())
