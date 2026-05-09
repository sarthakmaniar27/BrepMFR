import os
import argparse
import torch
from dgl.data.utils import load_graphs
from tqdm import tqdm


def find_bin_files(root_dir, suffix=".bin"):
    bin_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(suffix):
                bin_files.append(os.path.join(root, f))
    return sorted(bin_files)


def scan_invalid_face_area(root_dir, delete=False, eps=1e-12):
    files = find_bin_files(root_dir)
    print(f"[INFO] Found {len(files)} .bin files\n")

    bad_files = []

    for fp in tqdm(files, desc="Scanning bins", unit="file"):
        try:
            graphs, _ = load_graphs(fp)
            g = graphs[0]

            if "y" not in g.ndata:
                continue

            face_area = g.ndata["y"]

            if not torch.is_tensor(face_area) or face_area.numel() == 0:
                continue

            face_area = face_area.detach().cpu()

            # INVALID CONDITION:
            invalid_mask = face_area <= eps   # catches negative AND zero

            if invalid_mask.any():
                min_val = float(face_area.min())
                invalid_count = int(invalid_mask.sum())

                print("\n[INVALID FACE AREA FOUND]")
                print(f"File            : {fp}")
                print(f"Min value       : {min_val}")
                print(f"Invalid count   : {invalid_count}")
                print("-" * 60)

                bad_files.append(fp)

        except Exception as e:
            print(f"[ERROR] Failed to load {fp}: {e}")

    print("\n==============================")
    print(f"Total files with invalid face area (<= {eps}): {len(bad_files)}")
    print("==============================\n")

    if delete and bad_files:
        print("[WARNING] Deleting corrupted files...")
        for fp in bad_files:
            try:
                os.remove(fp)
                print(f"Deleted: {fp}")
            except Exception as e:
                print(f"[ERROR] Could not delete {fp}: {e}")

        print("\nDeletion completed.")

    return bad_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to folder containing .bin files"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete files that contain negative face areas"
    )

    args = parser.parse_args()

    scan_invalid_face_area(
    root_dir=args.root_dir,
    delete=args.delete,
    eps=1e-12
    )



# import os
# import argparse
# import torch
# from dgl.data.utils import load_graphs
# from tqdm import tqdm


# def find_bin_files(root_dir, suffix=".bin"):
#     bin_files = []
#     for root, _, files in os.walk(root_dir):
#         for f in files:
#             if f.lower().endswith(suffix):
#                 bin_files.append(os.path.join(root, f))
#     return sorted(bin_files)


# def scan_negative_face_area(root_dir, delete=False):
#     files = find_bin_files(root_dir)
#     print(f"[INFO] Found {len(files)} .bin files\n")

#     bad_files = []

#     for fp in tqdm(files, desc="Scanning bins", unit="file"):
#         try:
#             graphs, _ = load_graphs(fp)
#             g = graphs[0]

#             if "y" not in g.ndata:
#                 continue

#             face_area = g.ndata["y"]

#             if not torch.is_tensor(face_area) or face_area.numel() == 0:
#                 continue

#             face_area = face_area.detach().cpu()

#             neg_mask = face_area < 0
#             if neg_mask.any():
#                 min_val = float(face_area.min())
#                 neg_count = int(neg_mask.sum())

#                 print("\n[NEGATIVE FACE AREA FOUND]")
#                 print(f"File         : {fp}")
#                 print(f"Min value    : {min_val}")
#                 print(f"Neg count    : {neg_count}")
#                 print("-" * 60)

#                 bad_files.append(fp)

#         except Exception as e:
#             print(f"[ERROR] Failed to load {fp}: {e}")

#     print("\n==============================")
#     print(f"Total files with negative face area: {len(bad_files)}")
#     print("==============================\n")

#     if delete and bad_files:
#         print("[WARNING] Deleting corrupted files...")
#         for fp in bad_files:
#             try:
#                 os.remove(fp)
#                 print(f"Deleted: {fp}")
#             except Exception as e:
#                 print(f"[ERROR] Could not delete {fp}: {e}")

#         print("\nDeletion completed.")

#     return bad_files


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--root_dir",
#         type=str,
#         required=True,
#         help="Path to folder containing .bin files"
#     )
#     parser.add_argument(
#         "--delete",
#         action="store_true",
#         help="Delete files that contain negative face areas"
#     )

#     args = parser.parse_args()

#     scan_negative_face_area(
#         root_dir=args.root_dir,
#         delete=args.delete
#     )