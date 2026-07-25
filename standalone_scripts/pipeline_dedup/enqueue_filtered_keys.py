#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enqueue newly filtered (no Thread/Text) STEP keys into the pending queue.

Dedup: never enqueue a key that is already in
  - pending_keys.txt
  - stage2_distributed_keys.txt
  - stage2_done_keys.txt
  - stage1_seen_keys.txt (optional; always marked after enqueue)

Typical Stage-1 hourly loop (on the inference machine):

  # 1) infer only new JSONs
  conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_json_batch_inference.py `
    --json-dir C:\\jsons --skip-existing

  # 2) rebuild allowlist from Stage-2 filter output
  python standalone_scripts/export_step_allowlist_from_inference.py

  # 3) enqueue only NEW clean keys (skip already done / distributed / pending)
  python standalone_scripts/pipeline_dedup/enqueue_filtered_keys.py `
    --allowlist C:\\jsons\\inference\\allowed_step_keys.txt `
    --state-dir D:\\thread_and_text\\pipeline_state
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as script from repo root or from this folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from key_utils import (
    DEFAULT_STATE_DIR,
    append_keys,
    load_keys,
    state_paths,
    write_keys,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allowlist",
        type=Path,
        required=True,
        help="Text file of clean STEP keys (from export_step_allowlist_from_inference.py).",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help="Central ledger folder (must be reachable from this agent).",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=0,
        help="Optional cap on how many NEW keys to enqueue this run (0 = no cap).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.allowlist.is_file():
        print(f"ERROR: Allowlist not found: {args.allowlist}", file=sys.stderr)
        return 1

    paths = state_paths(args.state_dir)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    for p in paths.values():
        if not p.exists():
            write_keys(p, set())

    candidates = load_keys(args.allowlist)
    pending = load_keys(paths["pending"])
    distributed = load_keys(paths["stage2_distributed"])
    done = load_keys(paths["stage2_done"])
    seen = load_keys(paths["stage1_seen"])

    blocked = pending | distributed | done
    new_keys = sorted(candidates - blocked)

    if args.max_new and args.max_new > 0:
        new_keys = new_keys[: args.max_new]

    print(f"[INFO] Allowlist candidates : {len(candidates)}")
    print(f"[INFO] Already pending      : {len(pending)}")
    print(f"[INFO] Already distributed  : {len(distributed)}")
    print(f"[INFO] Already stage2 done  : {len(done)}")
    print(f"[INFO] NEW to enqueue       : {len(new_keys)}")

    if args.dry_run:
        for k in new_keys[:20]:
            print(f"  would enqueue: {k}")
        if len(new_keys) > 20:
            print(f"  ... and {len(new_keys) - 20} more")
        return 0

    if new_keys:
        added_p, total_p = append_keys(paths["pending"], new_keys)
        added_s, total_s = append_keys(paths["stage1_seen"], candidates)
        print(f"[OK] Pending += {added_p} (total pending={total_p})")
        print(f"[OK] Stage1 seen += {added_s} (total seen={total_s})")
    else:
        # Still mark all allowlist keys as seen so Stage-1 does not thrash.
        added_s, total_s = append_keys(paths["stage1_seen"], candidates)
        print("[OK] Nothing new to enqueue.")
        print(f"[OK] Stage1 seen += {added_s} (total seen={total_s})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
