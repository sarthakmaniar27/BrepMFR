import os
import csv
import argparse
from pathlib import Path
from collections import Counter

import torch
from dgl.data.utils import load_graphs
from tqdm import tqdm


def walk_bin_files(root_dir: Path):
    files = []
    for root, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".bin"):
                files.append(Path(root) / fn)
    return sorted(files)


def scan_label_distribution(bin_dir: Path, num_classes: int):
    files = walk_bin_files(bin_dir)

    label_counter = Counter()
    total_faces = 0
    files_scanned = 0
    files_failed = 0
    missing_label_key = 0
    invalid_label_count = 0

    for fp in tqdm(files, desc=f"Scanning {bin_dir.name}", unit="file"):
        try:
            graphs, _ = load_graphs(str(fp))
            g = graphs[0]

            if "f" not in g.ndata:
                missing_label_key += 1
                continue

            labels = g.ndata["f"]
            if not torch.is_tensor(labels) or labels.numel() == 0:
                continue

            labels = labels.detach().cpu().view(-1).tolist()

            for lbl in labels:
                lbl = int(lbl)
                label_counter[lbl] += 1
                total_faces += 1
                if lbl < 0 or lbl >= num_classes:
                    invalid_label_count += 1

            files_scanned += 1

        except Exception:
            files_failed += 1

    rows = []
    for cls in range(num_classes):
        count = label_counter.get(cls, 0)
        pct = (count / total_faces * 100.0) if total_faces > 0 else 0.0
        rows.append({
            "class_id": cls,
            "count": count,
            "percent": pct,
        })

    summary = {
        "dataset_dir": str(bin_dir),
        "files_found": len(files),
        "files_scanned": files_scanned,
        "files_failed": files_failed,
        "missing_label_key_files": missing_label_key,
        "total_faces": total_faces,
        "invalid_label_count": invalid_label_count,
    }

    return rows, summary, label_counter


def print_comparison(cadsynth_rows, mfcad_rows, cadsynth_summary, mfcad_summary):
    print("\n" + "=" * 100)
    print("DATASET SUMMARY")
    print("=" * 100)
    print(f"CADSynth  -> files_scanned={cadsynth_summary['files_scanned']}, total_faces={cadsynth_summary['total_faces']}, "
          f"files_failed={cadsynth_summary['files_failed']}, missing_label_key={cadsynth_summary['missing_label_key_files']}, "
          f"invalid_labels={cadsynth_summary['invalid_label_count']}")
    print(f"MFCAD++   -> files_scanned={mfcad_summary['files_scanned']}, total_faces={mfcad_summary['total_faces']}, "
          f"files_failed={mfcad_summary['files_failed']}, missing_label_key={mfcad_summary['missing_label_key_files']}, "
          f"invalid_labels={mfcad_summary['invalid_label_count']}")

    print("\n" + "=" * 100)
    print(f"{'Class':<10}{'CADSynth Count':>18}{'CADSynth %':>14}{'MFCAD++ Count':>18}{'MFCAD++ %':>14}")
    print("=" * 100)

    for r1, r2 in zip(cadsynth_rows, mfcad_rows):
        print(
            f"{r1['class_id']:<10}"
            f"{r1['count']:>18}"
            f"{r1['percent']:>14.4f}"
            f"{r2['count']:>18}"
            f"{r2['percent']:>14.4f}"
        )


def save_csv(csv_path: Path, cadsynth_rows, mfcad_rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id",
            "cadsynth_count",
            "cadsynth_percent",
            "mfcadpp_count",
            "mfcadpp_percent",
        ])
        for r1, r2 in zip(cadsynth_rows, mfcad_rows):
            writer.writerow([
                r1["class_id"],
                r1["count"],
                f"{r1['percent']:.6f}",
                r2["count"],
                f"{r2['percent']:.6f}",
            ])


def main():
    parser = argparse.ArgumentParser(description="Compare class label distribution in CADSynth and MFCAD++ bin datasets.")
    parser.add_argument("--cadsynth_bin_dir", type=str, required=True, help="Path to CADSynth bin folder")
    parser.add_argument("--mfcad_bin_dir", type=str, required=True, help="Path to MFCAD++ bin folder")
    parser.add_argument("--num_classes", type=int, default=25, help="Number of classes")
    parser.add_argument("--csv_out", type=str, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    cadsynth_dir = Path(args.cadsynth_bin_dir)
    mfcad_dir = Path(args.mfcad_bin_dir)

    cadsynth_rows, cadsynth_summary, _ = scan_label_distribution(cadsynth_dir, args.num_classes)
    mfcad_rows, mfcad_summary, _ = scan_label_distribution(mfcad_dir, args.num_classes)

    print_comparison(cadsynth_rows, mfcad_rows, cadsynth_summary, mfcad_summary)

    if args.csv_out:
        save_csv(Path(args.csv_out), cadsynth_rows, mfcad_rows)
        print(f"\n[INFO] CSV saved to: {args.csv_out}")


if __name__ == "__main__":
    main()


### Example run


# python compare_class_distribution.py ^
#   --cadsynth_bin_dir "C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment4\source_dataset\output\bin" ^
#   --mfcad_bin_dir "C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment4\target_dataset\output\bin" ^
#   --num_classes 25 ^
#   --csv_out "C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\class_distribution_comparison.csv"


# ### What it gives you

# * total files scanned
# * total face count
# * per-class count
# * per-class percentage
# * side-by-side CADSynth vs MFCAD++

# If you want, I can also give you a second version that outputs **class names** instead of only class IDs, using your remap/class mapping.
