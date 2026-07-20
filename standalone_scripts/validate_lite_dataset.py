# -*- coding: utf-8 -*-
"""Fast layout + sample-graph check for Z:\\thread_and_text\\lite (no full rglob)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

LITE_ROOT = Path(r"Z:\thread_and_text\lite")


def _count_lines(path: Path) -> int:
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


def _resolve_pt(pyg_root: Path, stem: str) -> Path | None:
    # Common layouts: pyg/<stem>.pt or nested
    direct = pyg_root / f"{stem}.pt"
    if direct.is_file():
        return direct
    matches = list(pyg_root.rglob(f"{stem}.pt"))
    return matches[0] if matches else None


def main() -> int:
    print("=" * 60)
    print("LITE DATASET LAYOUT CHECK (fast)")
    print("=" * 60)
    print(f"root: {LITE_ROOT} exists={LITE_ROOT.is_dir()}")

    required_dirs = ["pyg", "label"]
    required_files = ["train.txt", "val.txt", "test.txt"]
    ok = True
    for d in required_dirs:
        p = LITE_ROOT / d
        exists = p.is_dir()
        print(f"  [{'OK' if exists else 'MISSING'}] dir {d}/")
        ok = ok and exists
    split_counts = {}
    for f in required_files:
        p = LITE_ROOT / f
        exists = p.is_file()
        n = _count_lines(p) if exists else 0
        split_counts[f] = n
        print(f"  [{'OK' if exists else 'MISSING'}] {f}: lines={n}")
        ok = ok and exists

    if not ok:
        return 1

    print(f"  [INFO] split total stems={sum(split_counts.values())} (expect ~#graphs)")

    # Sample stems from each split
    samples = []
    for split in required_files:
        stems = [ln.strip() for ln in (LITE_ROOT / split).read_text(encoding="utf-8").splitlines() if ln.strip()]
        # take first, middle, last
        picks = {stems[0], stems[len(stems) // 2], stems[-1]}
        for s in picks:
            samples.append((split, s))

    profiles = Counter()
    has_flags = Counter()
    label_vals = Counter()
    bad = []
    resolved = 0

    print("\n" + "=" * 60)
    print(f"SAMPLE GRAPH INSPECTION (n={len(samples)})")
    print("=" * 60)

    pyg = LITE_ROOT / "pyg"
    for split, stem in samples:
        path = _resolve_pt(pyg, stem)
        if path is None:
            print(f"  [FAIL] {split}: stem={stem} .pt not found under pyg/")
            bad.append(stem)
            continue
        resolved += 1
        g = torch.load(path, map_location="cpu", weights_only=False)
        prof = getattr(g, "inference_profile", None)
        profiles[str(prof)] += 1
        a1 = bool(getattr(g, "has_a1", getattr(g, "spatial_pos", None) is not None))
        a2 = bool(getattr(g, "has_a2", False))
        a3 = bool(getattr(g, "has_a3", getattr(g, "edge_path", None) is not None))
        has_flags[(a1, a2, a3)] += 1
        sp = getattr(g, "spatial_pos", None) is not None
        d2 = getattr(g, "d2_distance", None) is not None
        ep = getattr(g, "edge_path", None) is not None
        ab = getattr(g, "attn_bias", None) is not None
        n = int(g.node_data.shape[0])
        e = int(g.edge_data.shape[0])
        uniq = sorted(int(v) for v in g.label_feature.unique().tolist())
        for v in uniq:
            label_vals[v] += 1
        ok_lite = (
            prof == "lite"
            and (not a1)
            and (not a2)
            and (not a3)
            and (not sp)
            and (not d2)
            and (not ep)
        )
        # UV shapes expected for lite export
        node_ok = tuple(g.node_data.shape[1:]) == (5, 5, 7)
        edge_ok = tuple(g.edge_data.shape[1:]) == (5, 7)
        status = "OK" if ok_lite and node_ok and edge_ok else "WARN"
        if status != "OK":
            bad.append(stem)
        print(
            f"  [{status}] {split} {path.name} N={n} E={e} "
            f"node={tuple(g.node_data.shape)} edge={tuple(g.edge_data.shape)} "
            f"profile={prof} a1/a2/a3={a1}/{a2}/{a3} "
            f"tensors_sp/d2/ep/ab={sp}/{d2}/{ep}/{ab} labels={uniq}"
        )

        # label json sibling check (optional)
        label_json = LITE_ROOT / "label" / f"{stem}.json"
        if not label_json.is_file():
            # try nested
            hits = list((LITE_ROOT / "label").rglob(f"{stem}.json"))
            if not hits:
                print(f"    [WARN] no label json for {stem}")

    print("\nsummary:")
    print(f"  resolved={resolved}/{len(samples)}")
    print(f"  profiles={dict(profiles)}")
    print(f"  has_a1_a2_a3_counts={ {str(k): v for k,v in has_flags.items()} }")
    print(f"  label_ids_seen={sorted(label_vals)}")
    if bad:
        print(f"  [FAIL] problems: {bad}")
        return 2
    print("  [PASS] layout + sampled graphs are lite-compatible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
