"""Compare absolute geometric scale of training graphs vs the Klavuz test part.

The pipeline applies no per-part centering or scaling, so UV point coordinates,
face_area and edge_len reach the network in raw model units. This measures how
far the Klavuz part sits from the training distribution on those raw scales.
"""

import glob
import json
import os
import random

import numpy as np
import torch

KLAVUZ_JSON = r"\\Gr-sw66464\d\Demo\grab_cad_brepmfr_testing\jsons\Klavuz_101.json"
PT_DIRS = [
    r"Z:\thread_and_text\abc_for_modelA_finetuning",
    r"Z:\thread_and_text\no_a2",
]
SAMPLE = 150


def stats_from_arrays(node_data, face_area, edge_len, thread_mask=None):
    uv = np.asarray(node_data, dtype=np.float64).reshape(-1, 5, 5, 7)
    m = uv[:, :, :, 6] > 0.5
    pts = uv[:, :, :, 0:3]
    flat = pts.reshape(-1, 3)
    fm = m.reshape(-1)
    use = flat[fm] if fm.any() else flat
    lo, hi = use.min(axis=0), use.max(axis=0)
    out = {
        "n_faces": uv.shape[0],
        "bbox_diag": float(np.linalg.norm(hi - lo)),
        "bbox_max_side": float((hi - lo).max()),
        "centroid_offset": float(np.linalg.norm(use.mean(axis=0))),
        "coord_absmax": float(np.abs(use).max()),
        "area_median": float(np.median(np.asarray(face_area, dtype=np.float64))),
        "area_max": float(np.asarray(face_area, dtype=np.float64).max()),
        "edgelen_median": float(np.median(np.asarray(edge_len, dtype=np.float64))),
    }
    if thread_mask is not None and np.any(thread_mask):
        fa = np.asarray(face_area, dtype=np.float64)
        out["thread_area_median"] = float(np.median(fa[thread_mask]))
    return out


def scan_pt_dir(d):
    files = []
    for root, _dirs, fns in os.walk(d):
        for fn in fns:
            if fn.endswith(".pt"):
                files.append(os.path.join(root, fn))
        if len(files) > 4000:
            break
    return files


def main():
    print("=== Klavuz_101 (real GrabCAD test part) ===")
    with open(KLAVUZ_JSON) as fh:
        kd = json.load(fh)
    kfaces = sorted(kd["faces"], key=lambda f: int(f["id"]))
    knode = np.asarray([f["uv"] for f in kfaces], dtype=np.float32).reshape(-1, 5, 5, 7)
    karea = np.asarray([f["y"] for f in kfaces], dtype=np.float32)
    kelen = np.asarray([e["l"] for e in kd["edges"]], dtype=np.float32)
    thread_idx = np.zeros(len(kfaces), dtype=bool)
    for i in list(range(113, 192)) + list(range(193, 271)) + list(range(274, 362)):
        thread_idx[i] = True
    ks = stats_from_arrays(knode, karea, kelen, thread_idx)
    for k, v in ks.items():
        print(f"  {k:18s}: {v}")

    for d in PT_DIRS:
        if not os.path.isdir(d):
            print(f"\n(skip missing {d})")
            continue
        files = scan_pt_dir(d)
        if not files:
            print(f"\n(no .pt under {d})")
            continue
        random.seed(0)
        pick = random.sample(files, min(SAMPLE, len(files)))
        rows = []
        thread_areas = []
        for p in pick:
            try:
                g = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                continue
            try:
                nd = g.node_data.float().numpy()
                fa = g.face_area.float().numpy()
                el = g.edge_len.float().numpy()
                lab = g.label_feature.numpy().reshape(-1) if hasattr(g, "label_feature") else None
            except Exception:
                continue
            tm = (lab == 1) if lab is not None else None
            try:
                rows.append(stats_from_arrays(nd, fa, el, tm))
            except Exception:
                continue
            if tm is not None and tm.any():
                thread_areas.append(float(np.median(fa[tm])))

        if not rows:
            print(f"\n(no readable graphs under {d})")
            continue

        print(f"\n=== TRAINING SAMPLE: {d}  (n={len(rows)} graphs of {len(files)} found) ===")
        for key in ["n_faces", "bbox_diag", "bbox_max_side", "centroid_offset",
                    "coord_absmax", "area_median", "area_max", "edgelen_median"]:
            v = np.array([r[key] for r in rows if key in r], dtype=np.float64)
            if v.size == 0:
                continue
            print(f"  {key:18s}: p5={np.percentile(v,5):11.4f}  median={np.median(v):11.4f}  "
                  f"p95={np.percentile(v,95):11.4f}   klavuz={ks.get(key, float('nan')):.4f}")
        if thread_areas:
            ta = np.array(thread_areas)
            print(f"  {'thread_area_median':18s}: p5={np.percentile(ta,5):11.6f}  "
                  f"median={np.median(ta):11.6f}  p95={np.percentile(ta,95):11.6f}   "
                  f"klavuz={ks.get('thread_area_median', float('nan')):.6f}")
            print(f"  --> ratio training_median / klavuz = "
                  f"{np.median(ta)/max(ks.get('thread_area_median',1e-12),1e-12):.1f}x")

        bd = np.array([r["bbox_diag"] for r in rows])
        print(f"  --> bbox_diag ratio training_median / klavuz = {np.median(bd)/ks['bbox_diag']:.2f}x")


if __name__ == "__main__":
    main()
