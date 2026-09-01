from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

IGNORE_LABEL = -100
NUM_CLASSES = 3
CLASS_NAMES = ("stock", "thread", "text")
SPLITS = ("train", "val", "test")

# The converter and Graphormer encoder use these values throughout the main
# repository. Keeping them centralized prevents profile drift in this package.
MULTI_HOP_MAX_DIST = 16
SPATIAL_POS_MAX = 32
NO_A2_PROFILE = "no_a2"

