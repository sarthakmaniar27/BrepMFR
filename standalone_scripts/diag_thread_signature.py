"""Compare the feature signature of TRAINING Thread faces (synthetic, inserted by
SolidWorks) against the REAL thread faces of the Klavuz part.

If the two differ systematically, a fine-tuned model can reach ~99% thread recall
in-distribution while scoring 0% on a real threaded part, because it latched onto
the synthetic insertion signature rather than thread geometry.
"""

import json
import os
import random
from collections import Counter

import numpy as np
import torch

KLAVUZ_JSON = r"\\Gr-sw66464\d\Demo\grab_cad_brepmfr_testing\jsons\Klavuz_101.json"
PT_DIRS = [
    (r"Z:\thread_and_text\abc_for_modelA_finetuning", "replay+uniqueABC (D/E/F train set)"),
    (r"Z:\thread_and_text\no_a2", "Model A train set"),
]
SAMPLE_GRAPHS = 400


def pct(counter):
    tot = sum(counter.values()) or 1
    return {k: round(100.0 * v / tot, 2) for k, v in sorted(counter.items())}


def klavuz_profile():
    with open(KLAVUZ_JSON) as fh:
        d = json.load(fh)
    faces = sorted(d["faces"], key=lambda f: int(f["id"]))
    thread = list(range(113, 192)) + list(range(193, 271)) + [273] + list(range(274, 362)) + list(range(362, 365))
    tset = sorted(set(thread))
    types = Counter(int(faces[i]["z"]) for i in tset)
    loops = Counter(int(faces[i]["l"]) for i in tset)
    degs = Counter(int(faces[i].get("a", 0)) for i in tset)
    areas = np.array([float(faces[i]["y"]) for i in tset])

    # Edge-level signature restricted to edges between two thread faces.
    tset_s = set(tset)
    e_type, e_conv, e_len, e_ang = Counter(), Counter(), [], []
    for e in d["edges"]:
        nf = e.get("nf", [])
        if len(nf) == 2 and int(nf[0]) in tset_s and int(nf[1]) in tset_s:
            e_type[int(e["t"])] += 1
            e_conv[int(e["c"])] += 1
            e_len.append(float(e["l"]))
            e_ang.append(float(e["a"]))
    return {
        "n": len(tset),
        "face_type_pct": pct(types),
        "loop_pct": pct(loops),
        "degree_pct": pct(degs),
        "area_median": float(np.median(areas)),
        "edge_type_pct": pct(e_type),
        "edge_conv_pct": pct(e_conv),
        "edge_len_median": float(np.median(e_len)) if e_len else None,
        "edge_ang_median": float(np.median(e_ang)) if e_ang else None,
        "n_thread_edges": len(e_len),
    }


def scan(d, tag):
    files = []
    for root, _dd, fns in os.walk(d):
        for fn in fns:
            if fn.endswith(".pt"):
                files.append(os.path.join(root, fn))
        if len(files) > 20000:
            break
    if not files:
        print(f"\n(no .pt under {d})")
        return
    random.seed(1)
    pick = random.sample(files, min(SAMPLE_GRAPHS, len(files)))

    types, loops, degs, e_type, e_conv = Counter(), Counter(), Counter(), Counter(), Counter()
    areas, elens, eangs = [], [], []
    graphs_with_thread = 0
    thread_face_counts = []

    for p in pick:
        try:
            g = torch.load(p, map_location="cpu", weights_only=False)
            lab = g.label_feature.numpy().reshape(-1)
        except Exception:
            continue
        tm = lab == 1
        if not tm.any():
            continue
        graphs_with_thread += 1
        thread_face_counts.append(int(tm.sum()))
        types.update(g.face_type.numpy().reshape(-1)[tm].tolist())
        loops.update(g.face_loop.numpy().reshape(-1)[tm].tolist())
        areas.extend(g.face_area.float().numpy().reshape(-1)[tm].tolist())
        ei = g.edge_index.numpy()
        deg = np.bincount(ei[0], minlength=len(lab))
        degs.update(deg[tm].tolist())
        both = tm[ei[0]] & tm[ei[1]]
        if both.any():
            e_type.update(g.edge_type.numpy().reshape(-1)[both].tolist())
            e_conv.update(g.edge_conv.numpy().reshape(-1)[both].tolist())
            elens.extend(g.edge_len.float().numpy().reshape(-1)[both].tolist())
            eangs.extend(g.edge_ang.float().numpy().reshape(-1)[both].tolist())

    print(f"\n=== TRAINING THREAD FACES: {tag} ===")
    print(f"  graphs sampled with thread: {graphs_with_thread} / {len(pick)}")
    print(f"  thread faces total        : {sum(types.values())}")
    print(f"  thread faces per part med : {np.median(thread_face_counts):.0f}")
    print(f"  face_type %               : {pct(types)}")
    print(f"  face_loop %               : {pct(loops)}")
    print(f"  degree %                  : {pct(degs)}")
    print(f"  face_area median          : {np.median(areas):.6f}")
    print(f"  edge_type % (thread-thread): {pct(e_type)}")
    print(f"  edge_conv % (thread-thread): {pct(e_conv)}")
    print(f"  edge_len median           : {np.median(elens):.6f}" if elens else "  edge_len: n/a")
    print(f"  edge_ang median           : {np.median(eangs):.6f}" if eangs else "  edge_ang: n/a")


if __name__ == "__main__":
    kp = klavuz_profile()
    print("=== KLAVUZ REAL THREAD FACES (all 3 flute lands, 249 faces) ===")
    for k, v in kp.items():
        print(f"  {k:22s}: {v}")
    for d, tag in PT_DIRS:
        if os.path.isdir(d):
            scan(d, tag)
        else:
            print(f"\n(skip missing {d})")
