"""Kaggle GPU training entrypoint for AVCS Pure-RGB VLA Policy.

Re-exports and delegates to avcs.train.train_vla.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Bootstrap sys.path for Kaggle execution environment
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_WORKING_DIR = Path.cwd()
if str(_WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKING_DIR))
_KAGGLE_WORKING = Path("/kaggle/working")
if _KAGGLE_WORKING.exists() and str(_KAGGLE_WORKING) not in sys.path:
    sys.path.insert(0, str(_KAGGLE_WORKING))

from avcs.train.train_vla import *  # noqa: F401, F403
from avcs.train.train_vla import (
    VLAPolicyTrainingWrapper,
    main,
    run_phase4_scaled_bc_training,
)

if __name__ == "__main__":
    main()
