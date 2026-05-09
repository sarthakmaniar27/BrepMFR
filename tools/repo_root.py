# -*- coding: utf-8 -*-
"""Repository root for ``tools/`` utilities.

After prepending the workspace root to ``sys.path`` using ``bootstrap_path.setup``,
``REPO_ROOT`` resolves to the folder that contains ``segmentation.py``.
"""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent


def ensure_repo_on_syspath(start_file: str) -> Path:
    """Walk upward from ``start_file`` until ``segmentation.py`` exists; prepend to ``sys.path``."""
    import sys

    p = Path(start_file).resolve()
    for d in p.parents:
        if (d / "segmentation.py").is_file():
            sd = str(d)
            if sd not in sys.path:
                sys.path.insert(0, sd)
            return d
    raise RuntimeError(
        "Could not find BrepMFR_PyG repo root (expected segmentation.py); "
        f"started from {start_file!r}"
    )
