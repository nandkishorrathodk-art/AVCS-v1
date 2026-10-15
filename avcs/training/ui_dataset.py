"""Compatibility bridge forwarding UI dataset imports to GUI dataset."""
from avcs.training.gui_dataset import (
    DesktopGUIGenerator,
    DesktopGUIGenerator as DesktopUIGenerator,
    GUIScreenSample,
    GUIScreenSample as UIScreenSample,
    FovealGUIElementSample,
    FovealGUIElementSample as FovealElementSample,
    AVCSUnifiedGUIDataset,
    AVCSUnifiedGUIDataset as AVCSUnifiedTrainingDataset,
)

__all__ = [
    "DesktopGUIGenerator",
    "DesktopUIGenerator",
    "GUIScreenSample",
    "UIScreenSample",
    "FovealGUIElementSample",
    "FovealElementSample",
    "AVCSUnifiedGUIDataset",
    "AVCSUnifiedTrainingDataset",
]
