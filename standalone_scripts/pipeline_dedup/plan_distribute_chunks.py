#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build per-node allowlist chunks from the pending queue (dedup-safe).

Reads ledgers under --state-dir, computes:

  to_ship = pending - stage2_done - stage2_distributed

Then round-robins keys into chunk_<NODE>.txt under --out-dir.

Does NOT move files yet — Jenkins agents copy from local C:\\abc_steps using
each chunk as an allowlist (append into C:\\abc_steps_filtered).

After a successful distribute Jenkins stage, call --commit to:
  - append shipped keys to stage2_distributed_keys.txt
  - remove them from pending_keys.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from key_utils import (
    DEFAULT_STATE_DIR,
    append_keys,
    load_keys,
    remove_keys,
    state_paths,
    write_keys,
)

DEFAULT_NODES = [
    "WALSWKQA19383",
    "WALSWKQA19381",
    "WALSWKQA19380",
    "WALSWKQA19374",
    "WALSWKQA19437",
    "WALSWKQA19438",
    "WALSWKQA19439",
    "WALSWKQA19440",
    "WALSWKQA19441",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Workspace folder to write chunk_<NODE>.txt files into.",
    )
    parser.add_argument(
        "--nodes",
        nargs="*",
        default=DEFAULT_NODES,
        help="Jenkins agent names (order used for round-robin).",
    )
    parser.add_argument(
        "--max-keys",
        type=int,
        default=0,
        help="Optional cap for this distribute wave (0 = all pending).",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="After chunks exist, mark those keys distributed and drop from pending.",
    )
    parser.add_argument(
        "--commit-from",
        type=Path,
        default=None,
        help="JSON manifest from a prior --plan run (keys that were actually chunked).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = state_paths(args.state_dir)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    for p in paths.values():
        if not p.exists():
            write_keys(p, set())

    if args.commit:
        manifest_path = args.commit_from or (args.out_dir / "distribute_manifest.json")
        if not manifest_path.is_file():
            print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
            return 1
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        shipped = set(data.get("keys", []))
        if args.dry_run:
            print(f"[DRY-RUN] Would commit {len(shipped)} keys as distributed")
            return 0
        added, total_d = append_keys(paths["stage2_distributed"], shipped)
        removed, total_p = remove_keys(paths["pending"], shipped)
        print(f"[OK] Distributed ledger += {added} (total={total_d})")
        print(f"[OK] Pending removed {removed} (total pending={total_p})")
        return 0

    pending = load_keys(paths["pending"])
    done = load_keys(paths["stage2_done"])
    distributed = load_keys(paths["stage2_distributed"])

    to_ship = sorted(pending - done - distributed)
    if args.max_keys and args.max_keys > 0:
        to_ship = to_ship[: args.max_keys]

    nodes = [n.strip() for n in args.nodes if n.strip()]
    if not nodes:
        print("ERROR: No nodes provided", file=sys.stderr)
        return 1

    print(f"[INFO] Pending           : {len(pending)}")
    print(f"[INFO] Done (skip)       : {len(done)}")
    print(f"[INFO] Distributed (skip): {len(distributed)}")
    print(f"[INFO] To ship this wave : {len(to_ship)}")
    print(f"[INFO] Nodes             : {len(nodes)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks: dict[str, list[str]] = {n: [] for n in nodes}
    for i, key in enumerate(to_ship):
        chunks[nodes[i % len(nodes)]].append(key)

    if args.dry_run:
        for n, keys in chunks.items():
            print(f"  {n}: {len(keys)} keys")
        return 0

    for n, keys in chunks.items():
        out = args.out_dir / f"chunk_{n}.txt"
        write_keys(out, keys)
        print(f"[OK] Wrote {out.name}: {len(keys)} keys")

    manifest = {
        "keys": to_ship,
        "per_node": {n: len(k) for n, k in chunks.items()},
        "count": len(to_ship),
    }
    manifest_path = args.out_dir / "distribute_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Manifest: {manifest_path} ({len(to_ship)} keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
