# -*- coding: utf-8 -*-
"""
occwl expects ``list_of_shapes_to_compound`` on ``OCC.Extend.DataExchange``.
pythonocc-core 7.7+ moved it to ``OCC.Extend.TopologyUtils``. Patch once before any occwl import
that pulls in ``occwl.compound``.
"""
from __future__ import annotations

_PATCHED = False


def apply_pythonocc_occwl_compatibility() -> None:
    global _PATCHED
    if _PATCHED:
        return
    import OCC.Extend.DataExchange as _dex

    if getattr(_dex, "list_of_shapes_to_compound", None) is not None:
        _PATCHED = True
        return
    try:
        from OCC.Extend.TopologyUtils import list_of_shapes_to_compound as _fn
    except ImportError as exc:
        raise ImportError(
            "Could not resolve list_of_shapes_to_compound for occwl. "
            "Install a matching pythonocc-core + occwl pair, or upgrade occwl."
        ) from exc
    _dex.list_of_shapes_to_compound = _fn
    _PATCHED = True
