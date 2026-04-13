import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------
# CONFIG
# -----------------------
ROOT_DIR = Path(
    r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment6\target_dataset\input\json_old_labels_og_mfcad_label_indices"
)
OUT_CSV = Path(
    r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment6\target_dataset\input\negative_labels_info.csv"
)

TARGET_LABEL = -10


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(str(x).strip())
    except Exception:
        return None


def analyze_json(path: Path) -> Tuple[int, int, int, str]:
    """
    Returns:
        total_faces, total_edges, neg_label_faces, error_msg
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        faces = data.get("faces")
        if not isinstance(faces, list):
            return 0, 0, 0, "Missing/invalid 'faces'"

        edges = data.get("edges")
        if not isinstance(edges, list):
            edges = data.get("topology", {}).get("edges", [])
            if not isinstance(edges, list):
                edges = []

        neg_count = 0
        for face in faces:
            if isinstance(face, dict):
                lbl = _safe_int(face.get("label"))
                if lbl == TARGET_LABEL:
                    neg_count += 1

        return len(faces), len(edges), neg_count, ""

    except Exception as e:
        return 0, 0, 0, f"{type(e).__name__}: {e}"


def ask_yes_no(prompt: str, default: str = "n") -> bool:
    val = input(prompt).strip().lower()
    if not val:
        val = default.lower()
    return val in {"y", "yes"}


def main():
    rows: List[Dict[str, Any]] = []
    files_to_delete: List[Path] = []

    total_files_scanned = 0
    files_with_neg = 0
    total_faces_all = 0
    total_edges_all = 0
    total_neg_faces_all = 0
    error_files = 0

    if not ROOT_DIR.exists():
        print(f"[ERROR] Folder does not exist: {ROOT_DIR}")
        return

    json_files = sorted(ROOT_DIR.rglob("*.json"))

    if not json_files:
        print(f"[INFO] No JSON files found in: {ROOT_DIR}")
        return

    for fp in json_files:
        total_files_scanned += 1

        total_faces, total_edges, neg_faces, err = analyze_json(fp)

        if err:
            error_files += 1

        if neg_faces > 0:
            files_with_neg += 1
            total_faces_all += total_faces
            total_edges_all += total_edges
            total_neg_faces_all += neg_faces
            files_to_delete.append(fp)

            rows.append(
                {
                    "file": fp.name,
                    "path": str(fp),
                    "total_faces": total_faces,
                    "total_edges": total_edges,
                    "neg_label_faces": neg_faces,
                    "error": err,
                }
            )

    print("========== ANALYSIS COMPLETE ==========")
    print(f"Folder scanned: {ROOT_DIR}")
    print(f"Total JSON files scanned: {total_files_scanned}")
    print(f"Files containing label {TARGET_LABEL}: {files_with_neg}")
    print(f"Total faces with label {TARGET_LABEL}: {total_neg_faces_all}")
    print(f"Files with read/parse issues: {error_files}")

    # Optional CSV output
    save_csv = ask_yes_no("\nDo you want to save a CSV report? (y/n): ", default="y")
    if save_csv:
        rows.append(
            {
                "file": "__SUMMARY__",
                "path": str(ROOT_DIR),
                "total_faces": total_faces_all,
                "total_edges": total_edges_all,
                "neg_label_faces": total_neg_faces_all,
                "error": f"total_files={total_files_scanned}; files_with_neg={files_with_neg}; error_files={error_files}",
            }
        )

        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["file", "path", "total_faces", "total_edges", "neg_label_faces", "error"]

        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"CSV saved to: {OUT_CSV}")
    else:
        print("CSV output skipped.")

    # Optional deletion
    if files_with_neg == 0:
        print("\nNo files contain the negative label. Nothing to delete.")
        return

    print(f"\n{files_with_neg} files contain label {TARGET_LABEL}.")

    do_delete = ask_yes_no(
        f"Do you want to delete these {files_with_neg} files? (y/n): ",
        default="n"
    )
    if not do_delete:
        print("Deletion skipped.")
        return

    confirm = input(
        f"Type 'delete' to permanently remove {files_with_neg} files: "
    ).strip().lower()

    if confirm != "delete":
        print("Deletion cancelled.")
        return

    deleted = 0
    failed = 0

    for fp in files_to_delete:
        try:
            fp.unlink()
            deleted += 1
        except Exception as e:
            failed += 1
            print(f"[WARN] Failed to delete {fp}: {e}")

    print("\n========== DELETION COMPLETE ==========")
    print(f"Files deleted: {deleted}")
    print(f"Files failed to delete: {failed}")


if __name__ == "__main__":
    main()