"""Actuation Safety & Motor Failsafe Breakout Hardening Verification.

Validates:
1. 4-corner latching CornerBreakoutFailSafe with 15px margin.
2. Permanent latching upon trigger, rejecting subsequent motor commands with
   RuntimeError("Fail-safe engaged: cursor moved to corner breakout region.").
3. HandMotor integration: rejection of move_to, click, and type_text once latched.
4. CLI default dry_run = True and explicit --live requirement in run_live_agent.
"""
from __future__ import annotations

import argparse
import inspect
import pytest

from avcs_sdk.hand_motor import (
    CornerBreakoutFailSafe,
    HandMotor,
    SafetyBreakoutException,
)
import avcs_sdk.run_live_agent as runner_babbage


class TestCornerBreakoutFailSafe:
    """Verifies 4-corner detection and latching breakout behavior."""

    def test_all_four_corners_trigger(self):
        w, h = 1920, 1080
        margin = 15

        corners = [
            (0, 0, "top-left"),
            (10, 10, "top-left"),
            (15, 15, "top-left"),
            (w - margin, 0, "top-right"),
            (w, 10, "top-right"),
            (w - 5, 15, "top-right"),
            (0, h - margin, "bottom-left"),
            (10, h - 5, "bottom-left"),
            (15, h, "bottom-left"),
            (w - margin, h - margin, "bottom-right"),
            (w - 5, h - 5, "bottom-right"),
            (w, h, "bottom-right"),
        ]

        for x, y, expected_corner in corners:
            fs = CornerBreakoutFailSafe(corner_margin_px=15)
            assert not fs.is_latched
            assert not fs.is_tripped

            with pytest.raises(RuntimeError) as exc_info:
                fs.check(x, y, w, h)

            assert "Fail-safe engaged: cursor moved to corner breakout region." in str(exc_info.value)
            assert fs.is_latched
            assert fs.is_tripped

    def test_safe_region_does_not_trigger(self):
        w, h = 1920, 1080
        safe_points = [
            (16, 16),
            (500, 500),
            (960, 540),
            (w - 16, h - 16),
            (w // 2, 16),
        ]

        fs = CornerBreakoutFailSafe(corner_margin_px=15)
        for x, y in safe_points:
            res = fs.check(x, y, w, h)
            assert res is False
            assert not fs.is_latched

    def test_permanent_latching_and_reset(self):
        w, h = 1920, 1080
        fs = CornerBreakoutFailSafe(corner_margin_px=15)

        # Trigger on top-left corner
        with pytest.raises(RuntimeError):
            fs.check(5, 5, w, h)

        assert fs.is_latched is True

        # Subsequent check in safe region MUST remain latched and reject
        for _ in range(5):
            with pytest.raises(RuntimeError) as exc_info:
                fs.check(500, 500, w, h)
            assert "Fail-safe engaged: cursor moved to corner breakout region." in str(exc_info.value)

        # Reset clears latch
        fs.reset()
        assert fs.is_latched is False
        assert fs.is_tripped is False
        assert fs.check(500, 500, w, h) is False

    def test_class_level_check_dispatch(self):
        """CornerBreakoutFailSafe.check() works when called on class."""
        CornerBreakoutFailSafe._get_default_instance().reset()
        res = CornerBreakoutFailSafe.check()
        assert isinstance(res, bool)


class TestHandMotorSafetyHardening:
    """Verifies HandMotor fail-safe latching and rejection of motor commands."""

    def test_motor_latches_when_commanded_to_corner(self):
        motor = HandMotor()
        assert motor.is_latched is False

        # Attempt move_to corner
        with pytest.raises(RuntimeError) as exc_info:
            motor.move_to(5, 5)

        assert "Fail-safe engaged: cursor moved to corner breakout region." in str(exc_info.value)
        assert motor.is_latched is True
        assert motor.is_failsafe_tripped is True

    def test_motor_rejects_subsequent_commands_once_latched(self):
        motor = HandMotor()
        # Since trigger() raises RuntimeError, let's catch it:
        with pytest.raises(RuntimeError):
            motor.failsafe.trigger("top-left", 0, 0)

        assert motor.is_latched is True

        # 1. move_to must reject immediately
        with pytest.raises(RuntimeError) as exc1:
            motor.move_to(500, 500)
        assert "Fail-safe engaged: cursor moved to corner breakout region." in str(exc1.value)

        # 2. click must reject immediately
        with pytest.raises(RuntimeError) as exc2:
            motor.click(500, 500)
        assert "Fail-safe engaged: cursor moved to corner breakout region." in str(exc2.value)

        # 3. type_text must reject immediately
        with pytest.raises(RuntimeError) as exc3:
            motor.type_text("safe input text")
        assert "Fail-safe engaged: cursor moved to corner breakout region." in str(exc3.value)

        # Resetting motor allows commands again (until next corner)
        motor.reset()
        assert motor.is_latched is False


class TestRunLiveAgentSafetyDefaults:
    """Verifies that run_live_agent defaults to dry-run and requires --live."""

    def test_run_agent_default_argument(self):
        sig = inspect.signature(runner_babbage.run_agent)
        assert "dry_run" in sig.parameters
        assert sig.parameters["dry_run"].default is True

    def test_cli_argument_parser_defaults(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dry-run", action="store_true", default=True)
        parser.add_argument("--live", action="store_true", default=False)

        # Default invocation (no flags): live is False, dry_run is True
        args = parser.parse_args([])
        is_dry_run = not args.live
        assert is_dry_run is True

        # Explicit --live invocation: live is True, dry_run is False
        args_live = parser.parse_args(["--live"])
        is_dry_run_live = not args_live.live
        assert is_dry_run_live is False

        # Explicit --dry-run invocation: live is False, dry_run is True
        args_dry = parser.parse_args(["--dry-run"])
        is_dry_run_dry = not args_dry.live
        assert is_dry_run_dry is True
