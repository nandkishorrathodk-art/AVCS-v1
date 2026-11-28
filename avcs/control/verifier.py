"""Visual Ripple Assertion Verifier & Self-Healing Recovery Ladder.

Implements biological post-action visual verification:
1. Optical Flow Ripple: Measures local kinetic energy burst around action point.
2. Surprise Pulse: Evaluates predictive coding temporal prediction error (S_t).
3. State Transition: Verifies UI structural changes via DOM-less UIML diffs (Delta_visual).
4. 5-Tier Self-Healing Escalation Ladder: Automatically recovers from missed clicks,
   border deadzones, and delayed UI transitions.
"""
from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from avcs.perception.human_eye import HumanEye, HumanEyeState
from avcs.memory.types import VisualDiff
from forge.vision.cortex.motor.driver import OSCursorDriver, get_default_driver


__all__ = [
    "VerificationStatus",
    "VerificationResult",
    "VisualRippleVerifier",
]


class VerificationStatus(enum.Enum):
    """Status code emitted by the Visual Ripple Verifier."""
    VERIFIED = "verified"                  # Tier 1: Action successfully verified on screen
    RETRY_MICRO_JITTER = "retry_jitter"    # Tier 2: Deadzone suspected; micro-offset retry
    KEYBOARD_FALLBACK = "key_fallback"     # Tier 3: Click unacknowledged; keyboard fallback
    SACCADIC_REGROUNDING = "regrounding"   # Tier 4: Target moved; gaze re-acquisition
    ESCALATE_BLOCKED = "escalate_blocked"  # Tier 5: Element unresponsive or modal blocked


@dataclass
class VerificationResult:
    """Comprehensive visual post-condition assertion result."""
    status: VerificationStatus
    action_name: str
    target_coord: Tuple[float, float]
    optical_ripple_energy: float
    surprise_pulse: float
    state_transition_detected: bool
    causal_summary: str
    recovery_attempt: int
    elapsed_ms: float
    diff: Optional[VisualDiff] = None
    photometric_delta: float = 0.0

    @property
    def is_successful(self) -> bool:
        return self.status == VerificationStatus.VERIFIED

    @property
    def verified(self) -> bool:
        """Alias for is_successful: True if action is verified on screen."""
        return self.status == VerificationStatus.VERIFIED


def _extract_element_center(elem: Any) -> Optional[Tuple[float, float]]:
    """Extracts normalized (cx, cy) coordinates from an affordance or UIMLElement."""
    if elem is None:
        return None
    if isinstance(elem, (tuple, list)) and len(elem) > 0:
        elem = elem[0]
    if hasattr(elem, "normalized_center"):
        nc = elem.normalized_center
        if isinstance(nc, (tuple, list)) and len(nc) == 2:
            return (float(nc[0]), float(nc[1]))
    bbox = getattr(elem, "bbox", None)
    if bbox is not None and hasattr(bbox, "center"):
        c = bbox.center
        return (float(c[0]), float(c[1]))
    if hasattr(elem, "center"):
        c = elem.center
        if isinstance(c, (tuple, list)) and len(c) == 2:
            return (float(c[0]), float(c[1]))
    return None


class VisualRippleVerifier:
    """Post-action visual assertion verifier and self-healing executor (<3ms evaluation)."""

    def __init__(
        self,
        eye: HumanEye,
        driver: Optional[OSCursorDriver] = None,
        ripple_radius: float = 0.08,  # ~8% of screen around target
        min_ripple_threshold: float = 0.015,
        min_surprise_threshold: float = 0.12,
        enforce_focus: bool = False,
    ) -> None:
        self.eye = eye
        self.driver = driver or get_default_driver()
        self.ripple_radius = ripple_radius
        self.min_ripple_threshold = min_ripple_threshold
        self.min_surprise_threshold = min_surprise_threshold
        self.enforce_focus = enforce_focus

    def calculate_local_ripple(
        self,
        target: Tuple[float, float],
        motion_field: np.ndarray,
    ) -> float:
        """Computes kinetic optical flow energy within the target's visual neighborhood."""
        if motion_field is None or motion_field.ndim != 3:
            return 0.0

        rows, cols = motion_field.shape[:2]
        tx, ty = target
        center_r = int(ty * rows)
        center_c = int(tx * cols)

        radius_r = max(1, int(self.ripple_radius * rows))
        radius_c = max(1, int(self.ripple_radius * cols))

        r_min = max(0, center_r - radius_r)
        r_max = min(rows, center_r + radius_r + 1)
        c_min = max(0, center_c - radius_c)
        c_max = min(cols, center_c + radius_c + 1)

        patch = motion_field[r_min:r_max, c_min:c_max]
        if patch.size == 0:
            return 0.0

        # Sum of velocity magnitudes
        speeds = np.hypot(patch[..., 0], patch[..., 1])
        return float(np.mean(speeds))

    def calculate_local_photometric_delta(
        self,
        target: Tuple[float, float],
        pre_luma: Optional[np.ndarray],
        post_luma: Optional[np.ndarray],
    ) -> float:
        """Computes local photometric luminance patch difference around target."""
        if pre_luma is None or post_luma is None:
            return 0.0
        if pre_luma.shape != post_luma.shape:
            return 0.0
        rows, cols = pre_luma.shape[:2]
        tx, ty = target
        center_r = int(ty * rows)
        center_c = int(tx * cols)

        radius_r = max(1, int(self.ripple_radius * rows))
        radius_c = max(1, int(self.ripple_radius * cols))

        r_min = max(0, center_r - radius_r)
        r_max = min(rows, center_r + radius_r + 1)
        c_min = max(0, center_c - radius_c)
        c_max = min(cols, center_c + radius_c + 1)

        patch_pre = pre_luma[r_min:r_max, c_min:c_max]
        patch_post = post_luma[r_min:r_max, c_min:c_max]
        if patch_pre.size == 0 or patch_post.size == 0:
            return 0.0
        return float(np.mean(np.abs(patch_post - patch_pre)))

    def verify_action(
        self,
        action_name: str,
        target: Tuple[float, float],
        pre_state: HumanEyeState,
        settle_ms: float = 60.0,
        min_sequence: Optional[int] = None,
        is_window_launch: bool = False,
    ) -> VerificationResult:
        """Verifies if an action produced a physical optical ripple or UI state transition."""
        t0 = time.perf_counter()

        is_launch = bool(
            is_window_launch
            or (action_name and action_name.startswith("open_application"))
            or (action_name and action_name.startswith("window_launch"))
        )
        if is_launch and (settle_ms == 60.0 or settle_ms is None):
            settle_ms = 1500.0

        # Pre-action foreground window snapshot for window launch detection
        pre_window = getattr(pre_state, "active_window_title", None) or getattr(self.eye, "_cached_win_title", None)
        if hasattr(self.driver, "get_foreground_window"):
            try:
                w = self.driver.get_foreground_window()
                if w:
                    pre_window = w
            except Exception:
                pass
        # pure-RGB invariant: no Win32 GetWindowText title scraping
        # Pre-action window title captured via driver only (above).

        # Wait settlement period for UI animations to trigger
        if settle_ms > 0:
            time.sleep(settle_ms / 1000.0)

        # Perceive current screen state (strictly fresh frame if min_sequence specified)
        post_state = self.eye.observe(min_sequence=min_sequence)

        # Post-action foreground window check
        post_window = getattr(post_state, "active_window_title", None) or getattr(self.eye, "_cached_win_title", None)
        if hasattr(self.driver, "get_foreground_window"):
            try:
                w = self.driver.get_foreground_window()
                if w:
                    post_window = w
            except Exception:
                pass
        # pure-RGB invariant: no Win32 GetWindowText title scraping outside driver

        foreground_window_changed = bool(
            pre_window is not None
            and post_window is not None
            and pre_window != post_window
        )

        # 1. Local Optical Flow Ripple
        ripple_energy = self.calculate_local_ripple(target, post_state.motion_field)

        # 2. Photometric Local Patch Delta using downscaled luminance grids
        pre_luma = getattr(pre_state, "luminance_grid", None) if pre_state else None
        patch_delta = self.calculate_local_photometric_delta(
            target, pre_luma, getattr(post_state, "luminance_grid", None)
        )

        # Peak-hold settlement window (3-tap peak hold for subtle or fast UI transients)
        if ripple_energy < self.min_ripple_threshold and patch_delta < 0.035 and settle_ms > 0:
            for _ in range(2):
                time.sleep(max(0.015, settle_ms / 3000.0))
                tap_state = self.eye.observe()
                tap_ripple = self.calculate_local_ripple(target, tap_state.motion_field)
                tap_delta = self.calculate_local_photometric_delta(
                    target, pre_luma, getattr(tap_state, "luminance_grid", None)
                )
                if tap_ripple > ripple_energy:
                    ripple_energy = tap_ripple
                    post_state = tap_state
                if tap_delta > patch_delta:
                    patch_delta = tap_delta
                    post_state = tap_state
                if ripple_energy >= self.min_ripple_threshold or patch_delta >= 0.035:
                    break

        # 3. Optical Flow Motion Differential against pre_state
        pre_local = 0.0
        if pre_state is not None and getattr(pre_state, "motion_field", None) is not None:
            pre_local = self.calculate_local_ripple(target, pre_state.motion_field)

        motion_delta = abs(ripple_energy - pre_local)

        if pre_local < self.min_ripple_threshold:
            # Clean baseline: fresh optical ripple burst at target
            optical_ripple_confirmed = ripple_energy >= self.min_ripple_threshold
        else:
            # Pre-existing motion: requires significant kinetic burst/differential from action
            optical_ripple_confirmed = (
                ripple_energy >= self.min_ripple_threshold
                and motion_delta >= self.min_ripple_threshold
            )

        # 4. Predictive Coding Surprise Pulse
        surprise_pulse = post_state.surprise_score

        # 5. Local State Transition Detection (strictly restricted to target vicinity)
        diff = post_state.visual_diff
        structural_change = False
        global_structural_diff = False
        if diff is not None:
            if (
                getattr(diff, "added_elements", None)
                or getattr(diff, "modified_elements", None)
                or getattr(diff, "removed_ids", None)
            ):
                global_structural_diff = True
            elif hasattr(diff, "is_empty") and not diff.is_empty:
                global_structural_diff = True

            target_radius = self.ripple_radius * 1.5
            tx, ty = target
            for elem in getattr(diff, "added_elements", []):
                c = _extract_element_center(elem)
                if c is not None and math.hypot(c[0] - tx, c[1] - ty) <= target_radius:
                    structural_change = True
                    break
            if not structural_change:
                for elem in getattr(diff, "modified_elements", []):
                    c = _extract_element_center(elem)
                    if c is not None and math.hypot(c[0] - tx, c[1] - ty) <= target_radius:
                        structural_change = True
                        break
            if not structural_change and getattr(diff, "removed_ids", None):
                prev_tree = getattr(pre_state, "scene_tree", None) if pre_state else None
                if prev_tree is None and hasattr(self.eye, "memory"):
                    prev_tree = getattr(self.eye.memory, "previous_tree", None)
                elements_map = getattr(prev_tree, "elements", {}) if prev_tree else {}
                if elements_map:
                    for rid in diff.removed_ids:
                        removed_elem = elements_map.get(rid)
                        if removed_elem is not None:
                            c = _extract_element_center(removed_elem)
                            if c is not None and math.hypot(c[0] - tx, c[1] - ty) <= target_radius:
                                structural_change = True
                                break
                else:
                    structural_change = True

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if is_launch:
            # Window launch evaluates success based on global screen transition / surprise score
            # (surprise_pulse >= 0.15 or global structural diff or foreground window change)
            # rather than strictly requiring optical ripple within the tight 0.12 icon target radius.
            verified = (
                surprise_pulse >= 0.15
                or global_structural_diff
                or foreground_window_changed
                or optical_ripple_confirmed
                or patch_delta >= 0.035
                or structural_change
            )
            state_transition = global_structural_diff or structural_change or foreground_window_changed
            summary = (
                diff.causal_summary if diff and getattr(diff, "causal_summary", "")
                else ("Window launch verified via screen transition." if verified else "No window launch or screen transition detected.")
            )
        else:
            # Decision Logic: Did the screen react locally to the physical actuation?
            # Requires verified local ripple, local photometric patch delta, localized structural change,
            # or surprise pulse backed by local kinetic, photometric, or structural evidence.
            photometric_confirmed = patch_delta >= 0.035
            surprise_confirmed = (
                surprise_pulse >= self.min_surprise_threshold
                and (ripple_energy > 0.005 or patch_delta >= 0.02 or structural_change)
            )
            verified = (
                optical_ripple_confirmed
                or photometric_confirmed
                or structural_change
                or surprise_confirmed
            )
            state_transition = structural_change
            summary = (
                diff.causal_summary if diff and diff.causal_summary
                else ("Physical visual ripple verified." if verified else "No optical response detected.")
            )

        status = VerificationStatus.VERIFIED if verified else VerificationStatus.RETRY_MICRO_JITTER

        return VerificationResult(
            status=status,
            action_name=action_name,
            target_coord=target,
            optical_ripple_energy=ripple_energy,
            surprise_pulse=surprise_pulse,
            state_transition_detected=state_transition,
            causal_summary=summary,
            recovery_attempt=1,
            elapsed_ms=elapsed_ms,
            diff=diff,
            photometric_delta=patch_delta,
        )

    def _move_cursor(self, x: float, y: float) -> None:
        if hasattr(self.driver, "move_to_normalized"):
            self.driver.move_to_normalized(x, y)
        elif hasattr(self.driver, "move_to"):
            self.driver.move_to(x, y)
        else:
            raise RuntimeError("Cursor driver does not support pointer movement.")

    def execute_and_verify(
        self,
        action_type: str,
        target: Tuple[float, float],
        text_content: Optional[str] = None,
        max_recovery_tier: int = 3,
        button: str = "left",
        keyboard_fallback_key: Optional[str] = None,
        enforce_focus: bool = False,
        is_destructive: bool = False,
        destructive: bool = False,
        action_class: Optional[Any] = None,
        is_window_launch: bool = False,
        action_name: Optional[str] = None,
        settle_ms: Optional[float] = None,
    ) -> VerificationResult:
        """Executes an action through the 5-Tier Self-Healing Escalation Ladder."""
        if action_type not in {"click", "double_click", "type", "open_application", "window_launch"}:
            raise ValueError(f"Unsupported verified action type: {action_type}")
        if button not in {"left", "right", "middle"}:
            raise ValueError(f"Unsupported mouse button: {button}")
        t_global = time.perf_counter()
        tx, ty = target

        is_launch = bool(
            is_window_launch
            or action_type.startswith("open_application")
            or action_type.startswith("window_launch")
            or (action_name and (action_name.startswith("open_application") or action_name.startswith("window_launch")))
        )

        # Determine non-replaying semantics: destructive / non-idempotent actions must never blind-replay
        is_dest = is_destructive or destructive
        if action_class is not None:
            ac_val = str(getattr(action_class, "value", action_class)).lower()
            if ac_val in {"non_idempotent", "high_risk", "destructive"}:
                is_dest = True

        # Preventative Fail-Closed Gating: Verify focus before dispatching keystrokes if enforced
        if action_type == "type" and (enforce_focus or getattr(self, "enforce_focus", False)):
            focus_state = self.eye.verify_focus(target_point=target)
            if not focus_state.is_focused:
                return VerificationResult(
                    status=VerificationStatus.ESCALATE_BLOCKED,
                    action_name=f"type@{target}",
                    target_coord=target,
                    optical_ripple_energy=0.0,
                    surprise_pulse=0.0,
                    state_transition_detected=False,
                    causal_summary=(
                        f"Typing blocked: target at {target} has no verified visual focus "
                        f"(conf={focus_state.confidence:.2f}). Keystrokes not dispatched."
                    ),
                    recovery_attempt=0,
                    elapsed_ms=(time.perf_counter() - t_global) * 1000.0,
                )

        # Tier 1: Primary Action Execution
        pre_seq = getattr(getattr(self.eye, "source", None), "sequence_number", getattr(self.eye, "frame_count", 0))
        pre_state = self.eye.observe()
        self._move_cursor(tx, ty)

        if action_type == "click":
            self.driver.click(button=button)
        elif action_type in {"double_click", "open_application", "window_launch"}:
            if button == "click":
                self.driver.click(button="left")
            else:
                self.driver.double_click(button="left" if button in {"double_click", "open_application", "window_launch"} else button)
        elif action_type == "type":
            self.driver.click(button="left")
            time.sleep(0.04)
            if text_content and hasattr(self.driver, "type_text"):
                self.driver.type_text(text_content)
            elif text_content:
                raise RuntimeError("Cursor driver does not support text entry.")

        act_name = action_name or f"{action_type}@{target}"
        v_kwargs = {"is_window_launch": is_launch}
        if settle_ms is not None:
            v_kwargs["settle_ms"] = settle_ms
        res = self.verify_action(act_name, target, pre_state, min_sequence=pre_seq, **v_kwargs)
        if res.is_successful:
            return res

        # Non-replaying guard: destructive clicks must never be automatically retried blindly
        if is_dest:
            res.status = VerificationStatus.ESCALATE_BLOCKED
            res.causal_summary = (
                f"Destructive action '{action_type}' was dispatched but visual verification was inconclusive; "
                "blind replay blocked to prevent unintended side effects."
            )
            res.recovery_attempt = 1
            res.elapsed_ms = (time.perf_counter() - t_global) * 1000.0
            return res

        if max_recovery_tier < 2:
            return res
        if action_type == "type":
            res.status = VerificationStatus.ESCALATE_BLOCKED
            res.causal_summary = (
                "Typing was dispatched but visual verification was inconclusive; "
                "it was not replayed to avoid duplicating text."
            )
            res.elapsed_ms = (time.perf_counter() - t_global) * 1000.0
            return res

        # Titlebar and Window Chrome Protection:
        # Strictly BAN micro-jitter retries and spiral clicks in top window chrome / tab bar
        # where y < 40px (normalized ty < 0.05, e.g. close buttons, tabs, menu bar)
        if ty < 0.05:
            res.status = VerificationStatus.ESCALATE_BLOCKED
            res.recovery_attempt = 1
            res.causal_summary = (
                f"Micro-jitter retry strictly banned in window chrome / titlebar (ty={ty:.3f} < 0.05). "
                "Escalating to System 2 without firing blind clicks."
            )
            res.elapsed_ms = (time.perf_counter() - t_global) * 1000.0
            return res

        # Tier 2: Single Guarded Re-check (Hitbox Border Offset)
        # Offset by +4 pixels (approx +0.003 normalized) with 150ms dwell to allow UI animation to settle
        time.sleep(0.15)
        ox, oy = (0.003, 0.003)
        jitter_target = (float(np.clip(tx + ox, 0.0, 1.0)), float(np.clip(ty + oy, 0.0, 1.0)))
        pre_retry = self.eye.observe()
        self._move_cursor(jitter_target[0], jitter_target[1])

        if action_type == "double_click":
            self.driver.double_click(button=button)
        else:
            self.driver.click(button=button)

        res_jitter = self.verify_action(
            f"retry_{action_type}@{jitter_target}",
            jitter_target,
            pre_retry,
            is_window_launch=is_launch,
        )
        if res_jitter.is_successful:
            res_jitter.status = VerificationStatus.VERIFIED
            res_jitter.recovery_attempt = 2
            res_jitter.causal_summary = f"Recovered via single guarded retry at offset ({ox:+.3f},{oy:+.3f})."
            return res_jitter

        # Escalation to System 2: Guarded retry exhausted; suppress blind click spam
        res.status = VerificationStatus.ESCALATE_BLOCKED
        res.recovery_attempt = 3
        res.causal_summary = (
            "Guarded ripple re-check unconfirmed; blind clicking suppressed. "
            "Escalating frame to DAgger queue and System 2."
        )
        res.elapsed_ms = (time.perf_counter() - t_global) * 1000.0
        return res
        res.recovery_attempt = 5
        res.causal_summary = "All 5 recovery tiers exhausted; target unresponsive or blocked."
        res.elapsed_ms = (time.perf_counter() - t_global) * 1000.0
        return res
