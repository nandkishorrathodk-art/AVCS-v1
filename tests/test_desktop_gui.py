"""Unit Tests for Native Desktop Computer GUI Application."""
from __future__ import annotations

import tkinter as tk
import pytest

from avcs.gui import AVCSDesktopApp
from avcs import SyntheticScreen, ScreenEye


def test_desktop_gui_app_initialization_and_widgets():
    """Verify AVCSDesktopApp builds native window, widgets, and bindings."""
    root = tk.Tk()
    root.withdraw()  # keep window off-screen during headless test

    app = AVCSDesktopApp(root=root, fps=10)

    assert root.title() == "AVCS // Artificial Visual Cortex - Desktop Station"
    assert app.canvas is not None
    assert app.query_entry is not None
    assert app.btn_look is not None
    assert app.btn_steer is not None
    assert app.lbl_category is not None
    assert app.lbl_stability is not None
    assert app.lbl_surprise is not None

    # Test query input change
    app.query_entry.delete(0, tk.END)
    app.query_entry.insert(0, "Close Window")
    assert app.query_entry.get() == "Close Window"

    # Test overlay toggle callbacks
    app.var_fovea.set(False)
    app._on_overlay_toggle()
    assert app.show_fovea is False

    app.var_top.set(True)
    app._on_top_toggle()
    assert app.always_on_top is True

    # Clean teardown
    app.on_close()
