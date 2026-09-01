"""Search the training set for faces labelled Stock that carry the Klavuz real-thread
signature.

The ABC "Stock-only" approval filter kept a part only when a CADSynth model
predicted no Thread/Text face above ~0.80 confidence. Native threads that the
model was merely UNSURE about therefore survive the filter and get labelled Stock.
That would place a systematic population of true-thread-geometry / Stock-label
faces into the fine-tuning data.

Klavuz real-thread signature, measured on the actual part:
  face_type == 6, face_loop == 1, face degree ~4, small face area,
  internal edges tangent-continuous (|edge_ang| > 3.0) and edge_conv == 0.
"""

import os
import random
from collections import deque

import numpy as np
import torch

PT_DIR = r"Z:\thread_and_text\abc_for_modelA_finetuning"
APPROVED_LIST = r"Z:\thread_and_text\no_confident_thread_or_text.txt"
MIN_CLUSTER = 15
SAMPLE = 1200


def approved_stems():
    with open(APPROVED_LIST) as fh:
        return {
            os.path.splitext(os.path.basename(l.strip()))[0]
            for l in fh if l.strip()
        }


def pt_files(d):
    out = []
    for root, _dd, fns in os.walk(d):
        out += [os.path.join(root, f) for f in fns if f.endswith(".pt")]
    return out


def find_thread_like_clusters(g):
    """Connected clusters of Stock-labelled faces that look like real thread."""
    lab = g.label_feature.numpy().reshape(-1)
    ftype = g.face_type.numpy().reshape(-1)
    floop = g.face_loop.numpy().reshape(-1)
    area = g.face_area.float().numpy().reshape(-1)
    ei = g.edge_index.numpy()
    eang = g.edge_ang.float().numpy().reshape(-1)
    econv = g.edge_conv.numpy().reshape(-1)
    n = len(lab)
    if n < MIN_CLUSTER:
        return []

    cand = (lab == 0) & (ftype == 6) & (floop == 1)
    if cand.sum() < MIN_CLUSTER:
        return []

    # Keep only tangent-continuous, smooth edges between two candidate faces.
    keep = cand[ei[0]] & cand[ei[1]] & (np.abs(eang) > 3.0) & (econv == 0)
    if keep.sum() < MIN_CLUSTER:
        return []

    adj = [[] for _ in range(n)]
    for s, t in zip(ei[0][keep], ei[1][keep]):
        adj[s].append(t)
        adj[t].append(s)

    seen = np.zeros(n, dtype=bool)
    clusters = []
    for i in range(n):
        if not cand[i] or seen[i] or not adj[i]:
            continue
        comp, q = [], deque([i])
        seen[i] = True
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        if len(comp) >= MIN_CLUSTER:
            a = area[comp]
            # Real thread faces are small and highly uniform in area.
            if np.median(a) < 0.02 and (a.std() / max(np.median(a), 1e-9)) < 1.5:
                clusters.append((len(comp), float(np.median(a))))
    return clusters


def main():
    ap = approved_stems()
    print(f"approved Stock-only parts listed : {len(ap)}")
    files = pt_files(PT_DIR)
    print(f".pt files found                  : {len(files)}")

    matched = [f for f in files
               if os.path.splitext(os.path.basename(f))[0] in ap]
    print(f".pt files from the approved list  : {len(matched)}")

    random.seed(7)
    for tag, pool in [("APPROVED Stock-only ABC parts", matched),
                      ("all training parts (baseline)", files)]:
        if not pool:
            print(f"\n({tag}: nothing to scan)")
            continue
        pick = random.sample(pool, min(SAMPLE, len(pool)))
        hits, sizes, scanned = 0, [], 0
        examples = []
        for p in pick:
            try:
                g = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                continue
            scanned += 1
            cl = find_thread_like_clusters(g)
            if cl:
                hits += 1
                biggest = max(cl, key=lambda x: x[0])
                sizes.append(biggest[0])
                if len(examples) < 12:
                    examples.append((os.path.basename(p), biggest))
        print(f"\n=== {tag} ===")
        print(f"  scanned                        : {scanned}")
        print(f"  parts with a Stock-labelled     ")
        print(f"  thread-signature cluster >= {MIN_CLUSTER}: {hits}  "
              f"({100.0*hits/max(scanned,1):.1f}%)")
        if sizes:
            s = np.array(sizes)
            print(f"  cluster size median/max        : {np.median(s):.0f} / {s.max()}")
            print(f"  total mislabelled faces (est.) : {int(s.sum())} in {scanned} parts")
            print("  examples (part, cluster_faces, median_area):")
            for nm, (cs, ma) in examples:
                print(f"    {nm[:64]:64s} {cs:4d}  {ma:.6f}")


if __name__ == "__main__":
    main()
