# -*- coding: utf-8 -*-
"""One-off helper: rebuild alpha=1.0 weights from counts in source_train_alpha05.json."""
import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
SRC = _REPO / "artifacts/class_weights/stage1/source_train_alpha05.json"
OUT = _REPO / "artifacts/class_weights/ablation/source_train_alpha100.json"


def main():
    with open(SRC, encoding="utf-8") as f:
        d = json.load(f)
    counts = np.array(d["counts"], dtype=np.float64)
    total = counts.sum()
    freq = np.maximum(counts / total, 1e-8)
    w = (1.0 / freq) ** 1.0
    w = w / w.mean()
    wmin = float(d["weight_min"])
    wmax = float(d["weight_max"])
    w = np.clip(w, wmin, wmax)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "inv_freq_pow",
        "alpha": 1.0,
        "num_classes": int(d["num_classes"]),
        "num_samples": int(d["num_samples"]),
        "total_faces": int(total),
        "weight_min": wmin,
        "weight_max": wmax,
        "counts": counts.astype(int).tolist(),
        "weights": w.astype(float).tolist(),
        "computed_from": d.get("computed_from", ""),
        "split_file": d.get("split_file", "train.txt"),
        "derived_from_json": SRC.as_posix(),
        "note": (
            "Counts copied from source_train_alpha05.json; weights recomputed with alpha=1.0. "
            "Equivalent to compute_class_weights.py on the same split without torch.load scan."
        ),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUT.resolve()}")
    print(f"mean={w.mean():.4f} min={w.min():.4f} max={w.max():.4f}")


if __name__ == "__main__":
    main()
