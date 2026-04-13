import os
import argparse
from pathlib import Path
import torch
from dgl.data.utils import load_graphs
from tqdm import tqdm


def walk_files(root: Path, suffix: str):
    out = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(suffix):
                out.append(Path(r) / fn)
    return sorted(out)


def stem_to_label_path(label_dir: Path, bin_path: Path) -> Path:
    # 00004552_101.bin -> 00004552_101.json
    # mfcad_13921_101.bin -> mfcad_13921_101.json
    return label_dir / (bin_path.stem + ".json")


def delete_pair(bin_path: Path, label_path: Path, dry_run: bool):
    if dry_run:
        print(f"[DRY] delete bin : {bin_path}")
        print(f"[DRY] delete json: {label_path}")
        return

    if bin_path.exists():
        bin_path.unlink()

    if label_path.exists():
        label_path.unlink()


def scan_invalid_face_area_and_delete(
    bin_dir: Path,
    label_dir: Path,
    delete: bool,
    dry_run: bool,
    eps: float,
    max_examples: int = 20,
):
    """
    Flags a file if ANY face area satisfies: face_area <= eps
      - eps=0.0  => removes negative + exact zero
      - eps=1e-12 => removes negative + near-zero
    """
    bins = walk_files(bin_dir, ".bin")
    print(f"[INFO] Found {len(bins)} bin files in: {bin_dir}")
    print(f"[INFO] Label dir: {label_dir}")
    print(f"[INFO] Invalid rule: ndata:y <= {eps}\n")

    bad = []
    failed = 0
    missing_y = 0

    for fp in tqdm(bins, desc="Scanning ndata:y", unit="file"):
        try:
            graphs, _ = load_graphs(str(fp))
            g = graphs[0]

            if "y" not in g.ndata:
                missing_y += 1
                continue

            y = g.ndata["y"]
            if not torch.is_tensor(y) or y.numel() == 0:
                continue

            y = y.detach().cpu()

            invalid_mask = y <= eps
            if invalid_mask.any():
                minv = float(y.min().item())
                invalidc = int(invalid_mask.sum().item())
                bad.append((fp, minv, invalidc))

        except Exception as e:
            failed += 1
            # keep it short
            print(f"[ERROR] {fp.name}: {e}")

    print("\n---------------- RESULTS ----------------")
    print(f"Files scanned                 : {len(bins)}")
    print(f"Files missing ndata:y         : {missing_y}")
    print(f"Files failed to load          : {failed}")
    print(f"Files with invalid face area  : {len(bad)}")

    if bad:
        print(f"\nExamples (first {min(max_examples, len(bad))}):")
        for fp, minv, invalidc in bad[:max_examples]:
            print(f"  {fp.name} | min_y={minv:.6g} | count(y<=eps)={invalidc}")

    if delete and bad:
        print("\n[WARNING] Deleting corrupted BIN + corresponding JSON...")
        for fp, _, _ in bad:
            label_fp = stem_to_label_path(label_dir, fp)
            delete_pair(fp, label_fp, dry_run=dry_run)

        if dry_run:
            print("\n[DRY RUN] No files were actually deleted.")
        else:
            print("\nDeletion completed.")

    return bad


def delete_orphan_labels(bin_dir: Path, label_dir: Path, delete: bool, dry_run: bool):
    bins = walk_files(bin_dir, ".bin")
    labels = walk_files(label_dir, ".json")

    bin_stems = set(p.stem for p in bins)
    orphans = [j for j in labels if j.stem not in bin_stems]

    print(f"[INFO] Bin files    : {len(bins)}")
    print(f"[INFO] Label files  : {len(labels)}")
    print(f"[INFO] Orphan labels (no matching .bin): {len(orphans)}\n")

    if orphans:
        print("Examples (first 30):")
        for j in orphans[:30]:
            print(f"  {j.name}")

    if delete and orphans:
        print("\n[WARNING] Deleting orphan label JSONs...")
        for j in orphans:
            if dry_run:
                print(f"[DRY] delete json: {j}")
            else:
                j.unlink()
        if dry_run:
            print("\n[DRY RUN] No files were actually deleted.")
        else:
            print("\nDeletion completed.")

    return orphans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin_dir", type=str, required=True, help="Folder containing .bin files")
    ap.add_argument("--label_dir", type=str, required=True, help="Folder containing .json label files")
    ap.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["scan_invalid_area", "delete_orphan_labels"],
        help=(
            "scan_invalid_area: scan bins for invalid ndata:y and optionally delete bin+json. "
            "delete_orphan_labels: delete jsons that have no corresponding bin."
        ),
    )
    ap.add_argument("--delete", action="store_true", help="Actually delete files (otherwise just report)")
    ap.add_argument("--dry_run", action="store_true", help="Print what would be deleted (safe preview)")
    ap.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Invalid threshold for face area. Files with any ndata:y <= eps are flagged. "
             "Use 0.0 for negative+zero. Use 1e-12 for negative+near-zero.",
    )
    ap.add_argument("--max_examples", type=int, default=20, help="How many bad-file examples to print")

    args = ap.parse_args()

    bin_dir = Path(args.bin_dir)
    label_dir = Path(args.label_dir)

    if args.mode == "scan_invalid_area":
        scan_invalid_face_area_and_delete(
            bin_dir,
            label_dir,
            delete=args.delete,
            dry_run=args.dry_run,
            eps=args.eps,
            max_examples=args.max_examples,
        )
    elif args.mode == "delete_orphan_labels":
        delete_orphan_labels(bin_dir, label_dir, delete=args.delete, dry_run=args.dry_run)


if __name__ == "__main__":
    main()