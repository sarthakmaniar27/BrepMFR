#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-queue keys that were marked distributed but may never have been copied.

Use after a bad distribute run that committed the full manifest even when
agents reported MISSING / empty C:\\abc_steps.

Default action:
  - Move ALL keys from stage2_distributed_keys.txt back into pending_keys.txt
  - Clear stage2_distributed_keys.txt
  - Leave stage2_done_keys.txt untouched (still skip already-finished work)

Safe to re-run. Next distribute wave will append-copy again; keys already
present in C:\\abc_steps_filtered count as success.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument(
        "--keep-distributed-intersection-with-done",
        action="store_true",
        help="If set, do not requeue keys that are also in stage2_done.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = state_paths(args.state_dir)
    if not paths["stage2_distributed"].is_file():
        print(f"ERROR: missing {paths['stage2_distributed']}", file=sys.stderr)
        return 1

    distributed = load_keys(paths["stage2_distributed"])
    pending = load_keys(paths["pending"])
    done = load_keys(paths["stage2_done"])

    to_requeue = set(distributed)
    if args.keep_distributed_intersection_with_done:
        to_requeue -= done

    print(f"[INFO] Distributed now : {len(distributed)}")
    print(f"[INFO] Pending now     : {len(pending)}")
    print(f"[INFO] Done (untouched): {len(done)}")
    print(f"[INFO] Will requeue    : {len(to_requeue)}")

    if args.dry_run:
        print("[DRY-RUN] No changes written.")
        return 0

    new_pending = pending | to_requeue
    # Keys that stay in distributed (only if we excluded done ones)
    stay_distributed = distributed - to_requeue

    write_keys(paths["pending"], new_pending)
    write_keys(paths["stage2_distributed"], stay_distributed)

    print(f"[OK] Pending now      : {len(new_pending)}")
    print(f"[OK] Distributed now  : {len(stay_distributed)}")
    print("[OK] Requeue complete. Re-run the distribute job.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
