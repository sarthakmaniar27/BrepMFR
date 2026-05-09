# -*- coding: utf-8 -*-
"""Bootstrap repo root onto ``sys.path`` for runnable scripts relocated under ``scripts/`` or ``tools/``."""

from __future__ import annotations

import sys
from pathlib import Path


def setup(script_file: str) -> Path:
    """Return repo root and ensure it is the first entry on ``sys.path``."""
    p = Path(script_file).resolve()
    for d in p.parents:
        if (d / "segmentation.py").is_file():
            sd = str(d)
            if sd not in sys.path:
                sys.path.insert(0, sd)
            return d
    raise RuntimeError(
        "Could not find BrepMFR_PyG repo root (expected segmentation.py). "
        f"Started from {script_file!r}"
    )


def load_via_script(script_file: str) -> Path:
    """Locate this repo via ``bootstrap_path.py`` on disk, then ``setup()``."""
    import importlib.util

    here = Path(script_file).resolve()
    for ancestor in here.parents:
        boot = ancestor / "bootstrap_path.py"
        if boot.is_file():
            spec = importlib.util.spec_from_file_location("_brep_bootstrap", boot)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load bootstrap from {boot}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup(str(here))
    raise RuntimeError(
        "bootstrap_path.py not found when walking upward from "
        f"{script_file!r}; run scripts from inside the repo clone."
    )
