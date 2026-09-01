"""Semi-supervised ABC adaptation for the Stage-1 B-rep segmenter.

The package is intentionally isolated from :mod:`segmentation` so the existing
supervised training and deployment workflows remain backward compatible.
"""

from .constants import IGNORE_LABEL, PACKAGE_ROOT, REPO_ROOT

__all__ = ["IGNORE_LABEL", "PACKAGE_ROOT", "REPO_ROOT"]

