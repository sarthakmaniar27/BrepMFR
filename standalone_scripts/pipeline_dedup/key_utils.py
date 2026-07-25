#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared STEP-key helpers for the continuous Stage1/Stage2 pipeline."""

from __future__ import annotations

import os
import re
from pathlib import Path

KEY_PATTERN = re.compile(r"^(?P<key>.+?_step_\d+)", re.IGNORECASE)

# Default ledger root on the state machine (GR-SW66464).
DEFAULT_STATE_DIR = Path(r"D:\thread_and_text\pipeline_state")
DEFAULT_DONE_JSON_DIR = Path(r"D:\thread_and_text\abc_json")

STAGE1_SEEN = "stage1_seen_keys.txt"
STAGE2_DONE = "stage2_done_keys.txt"
STAGE2_DISTRIBUTED = "stage2_distributed_keys.txt"
PENDING = "pending_keys.txt"


def extract_key(name_or_path: str) -> str | None:
    stem = Path(str(name_or_path).strip().strip('"')).stem
    match = KEY_PATTERN.match(stem)
    return match.group("key").lower() if match else None


def load_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = extract_key(line) if "_step_" in line.lower() else line.lower()
        if key:
            keys.add(key)
    return keys


def write_keys(path: Path, keys: set[str] | list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_keys = sorted(set(keys))
    path.write_text("\n".join(sorted_keys) + ("\n" if sorted_keys else ""), encoding="utf-8")
    return len(sorted_keys)


def append_keys(path: Path, keys: set[str] | list[str]) -> tuple[int, int]:
    """Append missing keys. Returns (added, total_after)."""
    existing = load_keys(path)
    before = len(existing)
    existing |= {k.lower() for k in keys if k}
    write_keys(path, existing)
    return len(existing) - before, len(existing)


def remove_keys(path: Path, keys: set[str] | list[str]) -> tuple[int, int]:
    """Remove keys from a ledger file. Returns (removed, total_after)."""
    existing = load_keys(path)
    before = len(existing)
    drop = {k.lower() for k in keys if k}
    existing -= drop
    write_keys(path, existing)
    return before - len(existing), len(existing)


def keys_from_json_dir(folder: Path) -> set[str]:
    """Unique STEP keys from a folder of *.json (many JSONs may share one key)."""
    keys: set[str] = set()
    if not folder.is_dir():
        return keys
    for name in os.listdir(folder):
        if not name.lower().endswith(".json"):
            continue
        key = extract_key(name)
        if key:
            keys.add(key)
    return keys


def list_step_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    out: list[Path] = []
    for name in os.listdir(folder):
        lower = name.lower()
        if lower.endswith(".step") or lower.endswith(".stp"):
            out.append(folder / name)
    return out


def state_paths(state_dir: Path) -> dict[str, Path]:
    return {
        "stage1_seen": state_dir / STAGE1_SEEN,
        "stage2_done": state_dir / STAGE2_DONE,
        "stage2_distributed": state_dir / STAGE2_DISTRIBUTED,
        "pending": state_dir / PENDING,
    }
