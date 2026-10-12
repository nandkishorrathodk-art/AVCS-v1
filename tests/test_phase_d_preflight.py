"""Unit tests for AVCS Milestone 8 Phase D Preflight Engine."""
from __future__ import annotations

import hashlib
import unittest
from unittest.mock import patch

import numpy as np

import avcs_m7_validation.phase_d_preflight as preflight
from avcs.capture_adapter import Win32GDICaptureAdapter
from avcs_m7_validation.phase_d_preflight import (
    CANONICAL_MODEL_PARAMS,
    CANONICAL_SHAPE_SHA256,
    PhaseDPreflightEngine,
    PreflightStatus,
)
from forge.vision.cortex.motor.win32 import Win32CursorDriver


class PhaseDPreflightTests(unittest.TestCase):
    """Test suite validating Phase D preflight checks and fail-closed gates."""

    def setUp(self) -> None:
        self.engine = PhaseDPreflightEngine()

    def test_candidate_freeze_checks(self) -> None:
        """Verifies the frozen-candidate check passes when its architecture pin matches."""
        arch_path = self.engine.repo_root / "avcs" / "cortex_2m.py"
        arch_sha = hashlib.sha256(arch_path.read_bytes()).hexdigest()
        with patch.object(preflight, "CANONICAL_ARCH_SHA256", arch_sha):
            res = self.engine.check_candidate_freeze()

        self.assertEqual(res.status, PreflightStatus.PASSED)
        self.assertEqual(res.details["total_parameters"], CANONICAL_MODEL_PARAMS)
        self.assertEqual(res.details["trainable_parameters"], CANONICAL_MODEL_PARAMS)
        self.assertEqual(res.details["non_trainable_parameters"], 0)
        self.assertEqual(res.details["arch_sha256"], arch_sha)
        self.assertEqual(res.details["shape_sha256"], CANONICAL_SHAPE_SHA256)

    def test_candidate_freeze_rejects_architecture_mismatch(self) -> None:
        """Verifies an architecture hash mismatch blocks candidate certification."""
        with patch.object(preflight, "CANONICAL_ARCH_SHA256", "0" * 64):
            res = self.engine.check_candidate_freeze()

        self.assertEqual(res.status, PreflightStatus.FAILED)
        self.assertEqual(res.details["total_parameters"], CANONICAL_MODEL_PARAMS)
        self.assertEqual(res.details["shape_sha256"], CANONICAL_SHAPE_SHA256)
        self.assertNotEqual(res.details["arch_sha256"], "0" * 64)
        self.assertIn("hash mismatch", res.error_message)

    def test_live_provenance_classes(self) -> None:
        """Verifies the preflight rejects the configured synthetic fixture adapter."""
        with (
            patch.object(Win32GDICaptureAdapter, "__init__", return_value=None),
            patch.object(Win32GDICaptureAdapter, "open"),
            patch.object(
                Win32GDICaptureAdapter,
                "grab",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            patch.object(Win32GDICaptureAdapter, "close"),
            patch.object(Win32CursorDriver, "__init__", return_value=None),
        ):
            res = self.engine.check_live_provenance_classes()

        self.assertEqual(res.status, PreflightStatus.FAILED)
        self.assertTrue(res.details["physical_capture_operational"])
        self.assertTrue(res.details["physical_driver_operational"])
        self.assertFalse(res.details["configured_adapter_uses_live_io"])
        self.assertFalse(res.details["goal_conditioned_reasoner_present"])
        self.assertIn("not a live visual task worker", res.error_message)

    def test_negative_controls(self) -> None:
        """Verifies that synthetic provenance and duplicate trial IDs fail closed."""
        res = self.engine.check_negative_controls()
        self.assertEqual(res.status, PreflightStatus.PASSED)
        self.assertTrue(res.details["synthetic_provenance_blocked_under_live_enforcement"])
        self.assertTrue(res.details["duplicate_trial_id_blocked"])

    def test_12_scenario_readiness_audit(self) -> None:
        """Verifies that all 12 catalog scenarios have live application harnesses registered."""
        res = self.engine.audit_12_scenario_live_readiness()
        # All 12 scenarios are now mapped
        self.assertEqual(res.status, PreflightStatus.PASSED)
        self.assertEqual(res.details["unmapped_count"], 0)
        self.assertEqual(len(res.details["live_mapped_scenarios"]), 12)
        self.assertIn("native_file_save_as", res.details["live_mapped_scenarios"])
        self.assertIn("overlay_stale_target", res.details["live_mapped_scenarios"])

    def test_unmapped_scenario_fails_closed(self) -> None:
        """Verifies that missing or unmapped scenario harness immediately fails closed to BLOCKED."""
        from avcs_m7_validation.live_harness.registry import LIVE_HARNESS_REGISTRY
        # Temporarily remove a scenario to simulate missing harness
        saved = LIVE_HARNESS_REGISTRY.pop("native_file_save_as", None)
        try:
            res = self.engine.audit_12_scenario_live_readiness()
            self.assertEqual(res.status, PreflightStatus.BLOCKED)
            self.assertGreater(res.details["unmapped_count"], 0)
            self.assertIn("native_file_save_as", res.details["unmapped_scenarios"])
        finally:
            if saved is not None:
                LIVE_HARNESS_REGISTRY["native_file_save_as"] = saved


if __name__ == "__main__":
    unittest.main()
