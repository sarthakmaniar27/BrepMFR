#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harvest Stage-2 JSON outputs and prune finished STEPs from abc_steps_filtered.

On each agent:
  1) Scan C:\\Threads\\jsons for STEP keys that already have synthetic output
  2) Delete matching .step/.stp from C:\\abc_steps_filtered
  3) Write harvested_keys.txt for the controller to merge into stage2_done

Also writes a small summary JSON for Jenkins.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from key_utils import extract_key, keys_from_json_dir, list_step_files, write_keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, default=Path(r"C:\Threads\jsons"))
    parser.add_argument("--filtered-dir", type=Path, default=Path(r"C:\abc_steps_filtered"))
    parser.add_argument(
        "--out-keys",
        type=Path,
        required=True,
        help="Where to write harvested STEP keys (one per line).",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=None,
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete STEPs in filtered-dir whose key appears in json-dir.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    host = os.environ.get("COMPUTERNAME", "local")
    done_keys = keys_from_json_dir(args.json_dir) if args.json_dir.is_dir() else set()
    write_keys(args.out_keys, done_keys)

    deleted = 0
    remaining = 0
    if args.filtered_dir.is_dir():
        steps = list_step_files(args.filtered_dir)
        remaining = len(steps)
        if args.prune:
            for step in steps:
                key = extract_key(step.name)
                if key and key in done_keys:
                    if args.dry_run:
                        print(f"  would delete: {step.name}")
                        deleted += 1
                    else:
                        try:
                            step.unlink(missing_ok=True)
                            deleted += 1
                        except OSError as exc:
                            print(f"  FAILED delete {step.name}: {exc}", file=sys.stderr)
            remaining = len(list_step_files(args.filtered_dir))

    summary = {
        "node": host,
        "json_dir": str(args.json_dir),
        "json_keys": len(done_keys),
        "deleted_steps": deleted,
        "remaining_filtered_steps": remaining,
        "prune": bool(args.prune),
        "dry_run": bool(args.dry_run),
    }
    print(
        f"[OK] {host}: json_keys={len(done_keys)} deleted={deleted} "
        f"remaining_filtered={remaining}"
    )
    if args.out_summary:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
