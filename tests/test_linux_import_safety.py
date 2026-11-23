"""Linux/Kaggle Import Safety Verification.

Verifies that avcs.protocol, avcs.models, avcs.train, avcs.data, and
avcs.data.lerobot_v3 NEVER eagerly import Windows-only modules (win32gui,
ctypes.windll, os_io) and import cleanly in headless Linux / Kaggle environments.
"""
from __future__ import annotations

import sys
import unittest
from unittest.mock import patch


class TestLinuxKaggleImportSafety(unittest.TestCase):
    """Verifies architectural import isolation for headless Linux/Kaggle environments."""

    def test_linux_safe_departments_never_import_os_io_or_win32(self):
        """Ensures protocol, models, train, and data.lerobot_v3 import without os_io or Win32."""
        # Ensure heavy libraries like torch and numpy are pre-initialized before mocking platform
        import torch  # noqa: F401
        import numpy  # noqa: F401

        # Save original sys.modules state
        saved_modules = dict(sys.modules)
        saved_platform = sys.platform

        # Modules to un-import for a clean test
        targets = [
            "avcs.protocol",
            "avcs.protocol.actions",
            "avcs.models",
            "avcs.models.vla_cortex",
            "avcs.models.local_model",
            "avcs.models.objectness_model",
            "avcs.train",
            "avcs.train.train_vla",
            "avcs.data",
            "avcs.data.lerobot_v3",
            "avcs.data.loader",
            "avcs.os_io",
            "avcs.os_io.computer",
            "avcs.os_io.hud",
        ]
        for mod in targets:
            sys.modules.pop(mod, None)

        # Forbidden modules on Linux/Kaggle
        forbidden_attempts = []

        class PoisonedImport:
            def __init__(self, name: str):
                self.name = name

            def __getattr__(self, item: str):
                msg = f"Forbidden Windows-only module access in Linux environment: {self.name}.{item}"
                forbidden_attempts.append(msg)
                raise ImportError(msg)

        blocked_names = [
            "win32gui",
            "win32api",
            "win32con",
            "win32process",
            "msvcrt",
            "avcs.os_io",
            "avcs.os_io.computer",
            "avcs.os_io.hud",
        ]
        for name in blocked_names:
            sys.modules[name] = PoisonedImport(name)  # type: ignore

        import ctypes
        had_windll = hasattr(ctypes, "windll")
        orig_windll = getattr(ctypes, "windll", None)

        class PoisonedWindll:
            def __getattr__(self, item: str):
                msg = f"Forbidden Windows-only module access in Linux environment: ctypes.windll.{item}"
                forbidden_attempts.append(msg)
                raise AttributeError(msg)

        if had_windll:
            ctypes.windll = PoisonedWindll()

        try:
            # Simulate Linux platform
            sys.platform = "linux"

            # 1. avcs.protocol
            import avcs.protocol
            import avcs.protocol.actions
            self.assertIsNotNone(avcs.protocol.Action)
            self.assertIsNotNone(avcs.protocol.ActionKind)

            # 2. avcs.models
            import avcs.models
            import avcs.models.vla_cortex
            self.assertIsNotNone(avcs.models.PureRGBVLAPolicy)
            self.assertIsNotNone(avcs.models.VLACortexModel)
            self.assertIsNotNone(avcs.models.ShowUITokenPruner)
            self.assertIsNotNone(avcs.models.LocalVAMClient)

            # 3. avcs.train
            import avcs.train
            import avcs.train.train_vla
            self.assertIsNotNone(avcs.train.run_phase4_scaled_bc_training)

            # 4. avcs.data.lerobot_v3 and avcs.data
            import avcs.data.lerobot_v3
            import avcs.data
            self.assertIsNotNone(avcs.data.lerobot_v3.LeRobotV3TrajectoryEpisode)
            self.assertIsNotNone(avcs.data.lerobot_v3.LeRobotV3Dataset)
            self.assertIsNotNone(avcs.data.lerobot_v3.QADoctor)
            self.assertIsNotNone(avcs.data.LeRobotV3Dataset)

            # Assert no forbidden attempts occurred
            self.assertEqual(
                forbidden_attempts,
                [],
                f"Windows-only imports were triggered during Linux import: {forbidden_attempts}",
            )

            # Assert avcs.os_io is strictly NOT imported
            loaded_os_io = [m for m in sys.modules if m.startswith("avcs.os_io") and not isinstance(sys.modules[m], PoisonedImport)]
            self.assertEqual(loaded_os_io, [], f"avcs.os_io was loaded: {loaded_os_io}")

        finally:
            # Restore original platform, ctypes, and modules
            sys.platform = saved_platform
            if had_windll:
                ctypes.windll = orig_windll
            for name in blocked_names:
                sys.modules.pop(name, None)
            sys.modules.update(saved_modules)


if __name__ == "__main__":
    unittest.main()
