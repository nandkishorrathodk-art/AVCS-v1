"""The AVCS GUI assistant: a live screen view and a Hinglish conversation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from avcs.gui.demo_source import CATEGORIES, DemoSceneSource
from avcs.gui.server import GuiApp, ScreenModel, build_app

_desktop_mod = None


def _load_desktop_gui() -> Any:
    global _desktop_mod
    if _desktop_mod is not None:
        return _desktop_mod
    if "avcs._gui" in sys.modules:
        _desktop_mod = sys.modules["avcs._gui"]
        return _desktop_mod
    gui_file = Path(__file__).resolve().parent.parent / "gui.py"
    if not gui_file.is_file():
        raise ImportError(f"Desktop GUI module not found at {gui_file}")
    spec = importlib.util.spec_from_file_location("avcs._gui", gui_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Desktop GUI spec from {gui_file}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["avcs._gui"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop("avcs._gui", None)
        raise
    _desktop_mod = mod
    return _desktop_mod


def launch_gui(*args: Any, mode: str = "desktop", **kwargs: Any) -> Any:
    """Launches the AVCS GUI application (desktop HUD by default, or web assistant)."""
    if args and isinstance(args[0], str) and args[0] in ("desktop", "assistant", "web", "server"):
        mode = args[0]
        args = args[1:]
    if mode in ("assistant", "web", "server"):
        from avcs.gui.run import main as run_main

        return run_main()
    mod = _load_desktop_gui()
    return mod.launch_gui(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "AVCSDesktopApp":
        mod = _load_desktop_gui()
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "AVCSDesktopApp",
    "CATEGORIES",
    "DemoSceneSource",
    "GuiApp",
    "ScreenModel",
    "build_app",
    "launch_gui",
]
