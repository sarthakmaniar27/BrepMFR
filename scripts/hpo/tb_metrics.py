# -*- coding: utf-8 -*-
"""Read the latest scalar from a Lightning TensorBoard log directory."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_tensorboard_scalar_dir(run_ckpt_dir: Path) -> Optional[Path]:
    """Resolve TB event dir for a Lightning run.

    Preferred layout (current): checkpoint dir ``results/stageN/<run>/`` and logs at
    ``results/logs/stageN/<run>/tensorboard/version_*``.

    Legacy: ``results/stageN/<run>/tensorboard/`` next to checkpoints.
    """
    run_ckpt_dir = Path(run_ckpt_dir)
    stage_name = run_ckpt_dir.parent.name
    results_root = run_ckpt_dir.parent.parent
    new_tb = results_root / "logs" / stage_name / run_ckpt_dir.name / "tensorboard"
    legacy_tb = run_ckpt_dir / "tensorboard"
    tb = new_tb if new_tb.is_dir() else legacy_tb
    if not tb.is_dir():
        return None
    versions = sorted(tb.glob("version_*"))
    if not versions:
        return tb
    return versions[-1]


def latest_scalar_value(logdir: Path, tag_candidates: List[str]) -> Optional[float]:
    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    for cand in tag_candidates:
        if cand in tags:
            series = ea.Scalars(cand)
            if series:
                return float(series[-1].value)
    return None
