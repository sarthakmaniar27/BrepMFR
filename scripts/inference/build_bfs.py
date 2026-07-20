#!/usr/bin/env python3
"""
Build the Cython BFS kernel (``_bfs_kernel``) for fast all-pairs shortest paths.

Usage (from repo root)::

    conda activate brep_mfr_pyg
    python scripts/inference/build_bfs.py

The compiled extension (``_bfs_kernel*.pyd`` on Windows, ``.so`` on Linux) is
placed next to ``_bfs_kernel.pyx`` so that ``json_to_brepmfr_pyg_optimized.py``
can import it directly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    pyx = here / "_bfs_kernel.pyx"
    if not pyx.is_file():
        print(f"ERROR: {pyx} not found.", file=sys.stderr)
        return 1

    # ---- imports ---------------------------------------------------------
    try:
        import numpy as np
        from Cython.Build import cythonize
        from setuptools import Distribution, Extension
        from setuptools.command.build_ext import build_ext
    except ImportError as exc:
        print(
            f"ERROR: missing build dependency: {exc}\n"
            "  Install with:  conda install cython numpy setuptools",
            file=sys.stderr,
        )
        return 1

    # ---- build -----------------------------------------------------------
    ext = Extension(
        "_bfs_kernel",
        sources=[str(pyx)],
        include_dirs=[np.get_include()],
        language="c",
    )

    extensions = cythonize([ext], language_level="3", annotate=False)

    dist = Distribution({"ext_modules": extensions})
    dist.parse_config_files()

    cmd = build_ext(dist)
    cmd.inplace = True
    cmd.ensure_finalized()

    # Build from the source directory so the .pyd lands beside the .pyx.
    old_cwd = os.getcwd()
    os.chdir(str(here))
    try:
        cmd.run()
    finally:
        os.chdir(old_cwd)

    # ---- verify ----------------------------------------------------------
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    try:
        from _bfs_kernel import all_pairs_bfs

        print(f"\n[OK] _bfs_kernel built and importable.")
        print(f"     Function: {all_pairs_bfs}")

        # Quick smoke test: 2-node graph  0 ⇄ 1
        import numpy as _np

        offsets = _np.array([0, 1, 2], dtype=_np.int32)
        targets = _np.array([1, 0], dtype=_np.int32)
        edge_ids = _np.array([0, 1], dtype=_np.int32)
        sp, ep = all_pairs_bfs(offsets, targets, edge_ids, 2, 4)
        assert sp[0, 0] == 0 and sp[1, 1] == 0, f"Self-distance failed: {sp}"
        assert sp[0, 1] == 1 and sp[1, 0] == 1, f"Neighbour distance failed: {sp}"
        assert ep[0, 1, 0] == 0, f"Edge path 0->1 failed: {ep[0, 1]}"
        assert ep[1, 0, 0] == 1, f"Edge path 1->0 failed: {ep[1, 0]}"
        print("     Smoke test PASSED.")
        return 0
    except ImportError as exc:
        print(f"Build appeared to succeed but import failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
