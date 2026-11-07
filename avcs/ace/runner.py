"""Autonomous Visual GUI Autopilot Runner (Ace-Control-Medium).

Executes the continuous observe() -> plan() -> execute() loop with:
1. Native Win32 5th-order minimum-jerk hardware actuation.
2. Direct Action Token decoding with zero CoT latency overhead.
3. Fail-safe corner breakout & emergency abort mechanisms.
4. Non-blocking natural speech feedback on lifecycle events.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import dataclasses
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from avcs.ace.actions import Action, ActionKind
from avcs.ace.agent import AceAgent
from avcs.ace.client import CustomLLMClient
from avcs.ace.computer import ExecutionSafetyFirewall, Observation, SafetyCheckResult, WindowsComputer
from avcs.ace.planner import CognitivePlanner, ReflectionDiagnostic, SubGoal
from avcs.ace.session import AceSession
from avcs.ace.skills_vault.cache import SkillCache, SkillLookupResult, UIStateAnchor
from avcs.ace.streamer import ScreenStreamer
from avcs.cortex.gate import StabilityGateResult, VisualCortexGate

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [ACE] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("avcs.ace.runner")


class SafetyBreakoutError(Exception):
    """Raised when emergency fail-safe is triggered by user intervention."""
    pass


_tts_speaker = None


def speak_feedback(text: str, enabled: bool = True) -> None:
    """Delivers natural speech feedback via AVCS TTS or graceful console fallback."""
    if not enabled or not text:
        return
    logger.info(f"VOICE: {text}")
    try:
        from avcs.voice.tts import Speaker
        global _tts_speaker
        if _tts_speaker is None:
            _tts_speaker = Speaker()
        _tts_speaker.speak_async(text)
    except Exception as e:
        logger.debug(f"Speech fallback: {e}")


@dataclasses.dataclass
class Stage1Result:
    """Stage 1 Physical Verification result (optical delta / click ripple probe)."""
    passed: bool
    optical_delta: float
    latency_ms: float
    ripple_detected: bool
    details: str = ""


@dataclasses.dataclass
class Stage2Result:
    """Stage 2 Semantic Verification result (OS window / process state)."""
    passed: bool
    matched_window: Optional[str] = None
    matched_process: Optional[str] = None
    latency_ms: float = 0.0
    details: str = ""


@dataclasses.dataclass
class DualStageVerificationResult:
    """Combined Dual-Stage Closed-Loop verification result."""
    stage1: Stage1Result
    stage2: Stage2Result
    overall_success: bool
    details: str = ""


class DualStageVerifier:
    """Dual-Stage Closed-Loop Verifier for Desktop Autopilot.

    Stage 1: Physical Verification (Optical delta / click ripple probe in <2 ms).
    Stage 2: Semantic Verification (Confirms intended OS window/process state emerged).
    """

    def __init__(
        self,
        computer: Optional[WindowsComputer] = None,
        cortex_gate: Optional[VisualCortexGate] = None,
    ) -> None:
        self.computer = computer
        self.cortex_gate = cortex_gate
        self.user32 = getattr(ctypes.windll, "user32", None) if sys.platform == "win32" else None

    def verify_physical(
        self,
        pre_obs: Optional[Observation],
        post_obs: Optional[Observation],
        action: Optional[Action] = None,
    ) -> Stage1Result:
        """Stage 1: Evaluates localized optical delta / click ripple in <2 ms."""
        if self.cortex_gate is not None:
            return self.cortex_gate.evaluate_ripple(pre_obs, post_obs, action=action)

        t0 = time.perf_counter()
        if pre_obs is None or post_obs is None:
            latency = (time.perf_counter() - t0) * 1000.0
            return Stage1Result(
                passed=False,
                optical_delta=0.0,
                latency_ms=round(latency, 2),
                ripple_detected=False,
                details="Missing pre- or post-observation frame.",
            )

        b1 = pre_obs.image_bytes
        b2 = post_obs.image_bytes

        if b1 == b2:
            latency = (time.perf_counter() - t0) * 1000.0
            return Stage1Result(
                passed=False,
                optical_delta=0.0,
                latency_ms=round(latency, 2),
                ripple_detected=False,
                details="Zero optical change detected.",
            )

        # Distributed optical delta probe across entire stream in <0.5 ms
        min_len = min(len(b1), len(b2))
        if min_len == 0:
            latency = (time.perf_counter() - t0) * 1000.0
            return Stage1Result(
                passed=False,
                optical_delta=0.0,
                latency_ms=round(latency, 2),
                ripple_detected=False,
                details="Empty frame byte stream.",
            )

        sample_step = max(1, min_len // 1024)
        s1 = b1[:min_len:sample_step]
        s2 = b2[:min_len:sample_step]
        diff_count = sum(1 for x, y in zip(s1, s2) if x != y)
        length_diff = abs(len(b1) - len(b2))

        delta_score = min(1.0, (diff_count / float(len(s1))) + (length_diff / float(max(len(b1), len(b2), 1))))
        latency = (time.perf_counter() - t0) * 1000.0

        has_ripple = diff_count > 0 or length_diff > 0

        return Stage1Result(
            passed=has_ripple,
            optical_delta=round(delta_score, 4),
            latency_ms=round(latency, 2),
            ripple_detected=has_ripple,
            details=f"Optical ripple {'confirmed' if has_ripple else 'not detected'} (delta={delta_score:.4f}) in {latency:.2f}ms.",
        )

    def get_active_window_title(self) -> str:
        """Returns the title of the active foreground window."""
        if not self.user32:
            return ""
        try:
            hwnd = self.user32.GetForegroundWindow()
            if not hwnd:
                return ""
            buf = ctypes.create_unicode_buffer(512)
            self.user32.GetWindowTextW(hwnd, buf, 512)
            return buf.value
        except Exception:
            return ""

    def get_all_window_titles(self) -> List[str]:
        """Enumerates visible top-level desktop windows."""
        titles: List[str] = []
        if not self.user32 or not hasattr(self.user32, "EnumWindows"):
            return titles
        try:
            EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

            def enum_proc(hwnd, lparam):
                if self.user32.IsWindowVisible(hwnd):
                    buf = ctypes.create_unicode_buffer(512)
                    length = self.user32.GetWindowTextW(hwnd, buf, 512)
                    if length > 0 and buf.value.strip():
                        titles.append(buf.value)
                return True

            self.user32.EnumWindows(EnumWindowsProc(enum_proc), 0)
        except Exception:
            pass
        return titles

    def get_foreground_process_name(self) -> str:
        """Returns the process executable name of the active foreground window."""
        if not self.user32:
            return ""
        try:
            hwnd = self.user32.GetForegroundWindow()
            if not hwnd:
                return ""
            pid = ctypes.wintypes.DWORD()
            self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value:
                try:
                    import psutil
                    return psutil.Process(pid.value).name().lower()
                except Exception:
                    pass
        except Exception:
            pass
        return ""

    def verify_semantic(
        self,
        expected_window_title: Optional[str] = None,
        expected_process_name: Optional[str] = None,
        goal: Optional[str] = None,
    ) -> Stage2Result:
        """Stage 2: Verifies intended semantic state (window or process emerged)."""
        t0 = time.perf_counter()
        fg_title = self.get_active_window_title()
        fg_proc = self.get_foreground_process_name()
        all_titles = self.get_all_window_titles()
        if fg_title and fg_title not in all_titles:
            all_titles.append(fg_title)

        # 1. Match explicit window title
        if expected_window_title:
            exp_wt = expected_window_title.lower()
            for t in all_titles:
                if exp_wt in t.lower():
                    latency = (time.perf_counter() - t0) * 1000.0
                    return Stage2Result(
                        passed=True,
                        matched_window=t,
                        matched_process=fg_proc or None,
                        latency_ms=round(latency, 2),
                        details=f"Semantic Verification passed: Window '{t}' matches expected '{expected_window_title}'.",
                    )

        # 2. Match explicit process name
        if expected_process_name:
            exp_p = expected_process_name.lower().replace(".exe", "")
            if fg_proc and exp_p in fg_proc:
                latency = (time.perf_counter() - t0) * 1000.0
                return Stage2Result(
                    passed=True,
                    matched_window=fg_title or None,
                    matched_process=fg_proc,
                    latency_ms=round(latency, 2),
                    details=f"Semantic Verification passed: Process '{fg_proc}' matches expected '{expected_process_name}'.",
                )

        # 3. Match from goal keywords and intent extraction
        if goal:
            apps = re.findall(r'\b(notepad|calculator|calc|paint|cmd|terminal|explorer|chrome|edge|settings|task\s*manager|spotify|word|excel|powerpoint|vlc|code|powershell)\b', goal, re.IGNORECASE)
            # Dynamic app target extraction: e.g. "Open <App>", "Launch <App>", "kholo <App>", "<App> open karo"
            dynamic_m = re.findall(r'(?:open|launch|start|run|focus|kholo)\s+([a-zA-Z0-9_\-\.\s]{2,24}?)(?:\s+(?:app|application|window|karo|kijiye|aur|and|on|in|$))', goal, re.IGNORECASE)
            apps.extend(dynamic_m)
            hinglish_m = re.findall(r'([a-zA-Z0-9_\-\.]{2,20})\s+(?:kholo|open\s+karo|launch\s+karo)', goal, re.IGNORECASE)
            apps.extend(hinglish_m)

            for app in apps:
                app_clean = app.strip().lower()
                if not app_clean or app_clean in ("the", "a", "an", "window", "app"):
                    continue
                for t in all_titles:
                    if app_clean in t.lower():
                        latency = (time.perf_counter() - t0) * 1000.0
                        return Stage2Result(
                            passed=True,
                            matched_window=t,
                            matched_process=fg_proc or None,
                            latency_ms=round(latency, 2),
                            details=f"Semantic Verification passed: Application '{app_clean}' detected in window '{t}'.",
                        )
                if fg_proc and app_clean in fg_proc:
                    latency = (time.perf_counter() - t0) * 1000.0
                    return Stage2Result(
                        passed=True,
                        matched_window=fg_title or None,
                        matched_process=fg_proc,
                        latency_ms=round(latency, 2),
                        details=f"Semantic Verification passed: Application '{app_clean}' detected in process '{fg_proc}'.",
                    )

        latency = (time.perf_counter() - t0) * 1000.0
        return Stage2Result(
            passed=False,
            matched_window=fg_title or None,
            matched_process=fg_proc or None,
            latency_ms=round(latency, 2),
            details="Semantic Verification failed: Target window or process did not emerge on desktop.",
        )


class TrajectoryLogger:
    """Serializes golden successful trajectories and negative failure trajectories for SFT/DPO."""

    def __init__(self, vault_dir: Optional[Union[str, Path]] = None) -> None:
        if vault_dir is None:
            vault_dir = Path(__file__).resolve().parent / "skills_vault"
        self.vault_dir = Path(vault_dir)
        self.trajectories_dir = self.vault_dir / "trajectories"
        self.golden_dir = self.trajectories_dir / "golden"
        self.failures_dir = self.trajectories_dir / "failures"
        self.golden_dir.mkdir(parents=True, exist_ok=True)
        self.failures_dir.mkdir(parents=True, exist_ok=True)

    def log_trajectory(
        self,
        session: AceSession,
        verified_success: bool,
        failure_reason: str = "",
        stuck_points: Optional[List[Dict[str, Any]]] = None,
        verification_result: Optional[DualStageVerificationResult] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Serializes session trajectory into JSON in golden/ or failures/."""
        dest_dir = self.golden_dir if verified_success else self.failures_dir
        timestamp = time.time()
        record = {
            "session_id": session.session_id,
            "goal": session.goal,
            "status": "success" if verified_success else "failure",
            "verified_success": verified_success,
            "failure_reason": failure_reason or (session.stop_reason if not verified_success else ""),
            "stuck_points": stuck_points or [],
            "total_steps": session.step_count,
            "is_complete": session.is_complete,
            "stop_reason": session.stop_reason,
            "dry_run": session.dry_run,
            "steps": [s.__dict__ if hasattr(s, "__dict__") else s for s in session.steps],
            "metrics": session.get_metrics(),
            "created_at": timestamp,
            "metadata": metadata or {},
        }
        if verification_result:
            record["verification"] = {
                "stage1_passed": verification_result.stage1.passed,
                "stage1_delta": verification_result.stage1.optical_delta,
                "stage2_passed": verification_result.stage2.passed,
                "stage2_details": verification_result.stage2.details,
            }

        filename = f"{session.session_id}_{int(timestamp)}.json"
        out_path = dest_dir / filename
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.info(f"Trajectory logged ({'GOLDEN' if verified_success else 'FAILURE'}): {out_path}")
        return str(out_path)


class AceRunner:
    """Continuous Autonomous Execution Loop for Desktop Autopilot."""

    def __init__(
        self,
        agent: Optional[AceAgent] = None,
        computer: Optional[WindowsComputer] = None,
        enable_voice: bool = True,
        firewall: Optional[ExecutionSafetyFirewall] = None,
        verifier: Optional[DualStageVerifier] = None,
        trajectory_logger: Optional[TrajectoryLogger] = None,
        skill_cache: Optional[SkillCache] = None,
        require_confirmation: bool = True,
        planner: Optional[CognitivePlanner] = None,
        cortex_gate: Optional[VisualCortexGate] = None,
    ) -> None:
        self.computer = computer or WindowsComputer()
        self.agent = agent or AceAgent()
        self.enable_voice = enable_voice
        self.firewall = firewall or ExecutionSafetyFirewall(require_confirmation=require_confirmation)
        self.cortex_gate = cortex_gate
        self.verifier = verifier or DualStageVerifier(computer=self.computer, cortex_gate=self.cortex_gate)
        if self.cortex_gate is not None and getattr(self.verifier, "cortex_gate", None) is None:
            self.verifier.cortex_gate = self.cortex_gate
        self.trajectory_logger = trajectory_logger or TrajectoryLogger()
        self.skill_cache = skill_cache or SkillCache()
        self.planner = planner or CognitivePlanner()
        self.user32 = getattr(ctypes.windll, "user32", None) if sys.platform == "win32" else None
        self._last_actuation_coord: Optional[Tuple[int, int]] = None

    def check_safety_breakout(self, target_coord: Optional[Tuple[int, int]] = None) -> None:
        """Emergency failsafe check: corner mouse breakout in any of the 4 screen corners or ESC key.

        Exempts intentional corner actions targeting bottom-left (Start button) or
        top-right (window Close button).
        """
        if self.user32 is None:
            return

        # 1. 4-Corner mouse breakout (margin 15px)
        try:
            cx, cy = self.computer.get_cursor_pos()
            nw = self.computer.native_width
            nh = self.computer.native_height
            margin = 15

            top_left = 0 <= cx <= margin and 0 <= cy <= margin
            top_right = (nw - margin) <= cx <= nw and 0 <= cy <= margin
            bottom_left = 0 <= cx <= margin and (nh - margin) <= cy <= nh
            bottom_right = (nw - margin) <= cx <= nw and (nh - margin) <= cy <= nh

            # Check if this is an intentional corner action (e.g. Start button or Close button)
            exempt_bottom_left = False
            exempt_top_right = False

            check_coords = []
            if target_coord is not None:
                check_coords.append(target_coord)
            if self._last_actuation_coord is not None:
                check_coords.append(self._last_actuation_coord)

            for tx, ty in check_coords:
                if 0 <= tx <= margin * 4 and (nh - margin * 4) <= ty <= nh:
                    exempt_bottom_left = True
                if (nw - margin * 4) <= tx <= nw and 0 <= ty <= margin * 4:
                    exempt_top_right = True

            if top_left:
                raise SafetyBreakoutError(f"Safety breakout: Mouse pushed to top-left corner ({cx}, {cy}). Aborting immediately.")
            elif top_right and not exempt_top_right:
                raise SafetyBreakoutError(f"Safety breakout: Mouse pushed to top-right corner ({cx}, {cy}). Aborting immediately.")
            elif bottom_left and not exempt_bottom_left:
                raise SafetyBreakoutError(f"Safety breakout: Mouse pushed to bottom-left corner ({cx}, {cy}). Aborting immediately.")
            elif bottom_right:
                raise SafetyBreakoutError(f"Safety breakout: Mouse pushed to bottom-right corner ({cx}, {cy}). Aborting immediately.")
        except Exception as e:
            if isinstance(e, SafetyBreakoutError):
                raise

        # 2. ESC key emergency abort (VK_ESCAPE = 0x1B)
        try:
            if self.user32.GetAsyncKeyState(0x1B) & 0x8000:
                raise SafetyBreakoutError("Safety breakout: ESC key pressed. Aborting autopilot.")
        except Exception as e:
            if isinstance(e, SafetyBreakoutError):
                raise

    def run(
        self,
        goal: str,
        max_steps: int = 30,
        image_format: str = "jpeg",
        dry_run: bool = False,
        current_anchor: Optional[UIStateAnchor] = None,
        user_confirmed: bool = False,
        expected_window: Optional[str] = None,
        expected_process: Optional[str] = None,
        subgoals: Optional[List[SubGoal]] = None,
    ) -> AceSession:
        """Executes the autonomous loop until completion or maximum steps."""
        session = AceSession()
        session.start(goal=goal, max_steps=max_steps, dry_run=dry_run)

        # 1. Layer 1: Cognitive Intelligence & Sub-Goal Planner (0.5 - 1.0 Hz)
        if subgoals is None:
            subgoals = self.planner.decompose(goal)
        session.subgoals = [s.to_dict() if hasattr(s, "to_dict") else s for s in subgoals]
        logger.info(
            f"Layer 1 Cognitive Planner decomposed goal into {len(subgoals)} atomic sub-goals: "
            f"{[s.description for s in subgoals]}"
        )

        # 2. Layer 2 Anchor-Verified Skill Cache lookup for full composite goal (0 ms deterministic execution)
        lookup_anchor = current_anchor
        if lookup_anchor is None:
            # Auto-detect live desktop state anchor if none explicitly provided
            active_title = self.verifier.get_active_window_title()
            active_proc = self.verifier.get_foreground_process_name()
            if active_title or active_proc:
                lookup_anchor = UIStateAnchor(
                    window_title=active_title or None,
                    process_name=active_proc or None,
                )

        cache_lookup = self.skill_cache.lookup(goal, current_anchor=lookup_anchor)
        if cache_lookup.hit and cache_lookup.actions:
            logger.info(f"[SKILL CACHE HIT] {cache_lookup.reason}")
            speak_feedback(f"Executing verified skill for: {goal}", enabled=self.enable_voice)

            # Replay with firewall and breakout safety checks
            for cached_act in cache_lookup.actions:
                native_target = None
                if cached_act.coordinate:
                    native_target = self.computer._canonical_to_native(*cached_act.coordinate)
                elif cached_act.start_coordinate:
                    native_target = self.computer._canonical_to_native(*cached_act.start_coordinate)
                self.check_safety_breakout(target_coord=native_target)

                # Evaluate execution safety firewall during cache replay
                firewall_res = self.firewall.evaluate(cached_act, user_confirmed=user_confirmed)
                if not firewall_res.allowed:
                    logger.warning(f"[SKILL CACHE REPLAY BLOCKED] {firewall_res.reason}")
                    session.record_step(Action(kind=ActionKind.WAIT, duration=0.5, reason=firewall_res.reason), plan_latency_ms=0.0)
                    session.finish(f"Replay halted by firewall: {firewall_res.reason}")
                    self.trajectory_logger.log_trajectory(session, verified_success=False, failure_reason=firewall_res.reason)
                    return session

                session.record_step(cached_act, plan_latency_ms=0.0)
                if not dry_run:
                    self.computer.execute(cached_act, dry_run=False)
                else:
                    self.computer.execute(cached_act, dry_run=True)
                if cached_act.kind == ActionKind.STOP:
                    break

            for sg in subgoals:
                if hasattr(sg, "mark_completed"):
                    sg.mark_completed()
            session.finish("Skill cache execution completed successfully.")
            self.trajectory_logger.log_trajectory(session, verified_success=True)
            return session

        mode_str = "[DRY-RUN]" if dry_run else "[LIVE-ACTUATION]"
        logger.info(f"Starting Ace-Control-Medium Autopilot {mode_str}")
        logger.info(f"Goal: '{goal}' | Max Steps: {max_steps} | Model: {self.agent.client.model}")

        if dry_run:
            logger.warning(
                "[DRY-RUN MODE ACTIVE] Physical mouse and keyboard hardware events are intentionally BYPASSED. "
                "The desktop display will NOT change automatically unless you perform actions manually. "
                "To allow the autopilot to actuate on your live Windows desktop, run without --dry-run."
            )
            speak_feedback(
                "Dry run mode active. Mouse and keyboard actions are simulated and will not change your screen.",
                enabled=self.enable_voice,
            )
        else:
            speak_feedback(f"Starting desktop autopilot for: {goal}", enabled=self.enable_voice)

        streamer = ScreenStreamer(computer=self.computer, target_fps=30.0, image_format=image_format)
        step_idx = 1
        subgoal_idx = 0
        stuck_points: List[Dict[str, Any]] = []
        last_stage1_res: Optional[Stage1Result] = None

        try:
            streamer.start()
            while not session.is_complete and step_idx <= max_steps and subgoal_idx < len(subgoals):
                active_subgoal = subgoals[subgoal_idx]
                if hasattr(active_subgoal, "mark_in_progress"):
                    active_subgoal.mark_in_progress()

                logger.info(f"--- Sub-Goal {active_subgoal.index}/{len(subgoals)}: '{active_subgoal.description}' (Step {step_idx}/{max_steps}) ---")

                # 1. Safety Breakout Check
                self.check_safety_breakout()

                # 2. Visual Desktop Observation (Instant stream frame <0.1ms)
                obs = streamer.get_latest_observation() or self.computer.observe(format=image_format)
                logger.info(f"Observed desktop: {obs.width}x{obs.height} ({obs.format}) in {obs.capture_time_ms:.1f}ms")

                # 3. System 1 Sensory Stability Gate (Local Visual Cortex ConvGRU)
                # Must verify screen is stable before querying Layer 3 or replaying cache
                if self.cortex_gate is not None:
                    stab_res = self.cortex_gate.check_stability(
                        observation=obs,
                        max_settle_wait=0.5,
                        poll_interval=0.05,
                        observation_provider=lambda: streamer.get_latest_observation() or self.computer.observe(format=image_format),
                    )
                    logger.info(
                        f"Visual Cortex Stability Gate: is_stable={stab_res.is_stable} "
                        f"(surprise={stab_res.surprise_score:.4f}, spinner={stab_res.spinner_active}) "
                        f"in {stab_res.latency_ms:.1f}ms"
                    )
                    if not stab_res.is_stable:
                        logger.warning(
                            f"Cortex stability gate held: UI settling or loading spinner active ({stab_res.details}). Actuation gated."
                        )
                        wait_action = Action(kind=ActionKind.WAIT, duration=0.5, reason=f"Stability gate held: {stab_res.details}")
                        session.record_step(wait_action, observation=obs, plan_latency_ms=0.0)
                        step_idx += 1
                        continue

                # 4. Anchor-Verified Skill Cache Check for active Sub-Goal
                sg_lookup = self.skill_cache.lookup(active_subgoal.description, current_anchor=lookup_anchor)
                if sg_lookup.hit and sg_lookup.actions:
                    logger.info(f"[SKILL CACHE HIT (SUB-GOAL)] {sg_lookup.reason}")
                    replay_blocked = False
                    for cached_act in sg_lookup.actions:
                        native_target = None
                        if cached_act.coordinate:
                            native_target = self.computer._canonical_to_native(*cached_act.coordinate)
                        elif cached_act.start_coordinate:
                            native_target = self.computer._canonical_to_native(*cached_act.start_coordinate)
                        self.check_safety_breakout(target_coord=native_target)

                        firewall_res = self.firewall.evaluate(cached_act, user_confirmed=user_confirmed)
                        if not firewall_res.allowed:
                            logger.warning(f"[SKILL CACHE REPLAY BLOCKED] {firewall_res.reason}")
                            session.record_step(Action(kind=ActionKind.WAIT, duration=0.5, reason=firewall_res.reason), plan_latency_ms=0.0)
                            replay_blocked = True
                            break

                        session.record_step(cached_act, plan_latency_ms=0.0)
                        self.computer.execute(cached_act, dry_run=dry_run)
                        if cached_act.kind == ActionKind.STOP:
                            break

                    if not replay_blocked:
                        if hasattr(active_subgoal, "mark_completed"):
                            active_subgoal.mark_completed()
                        subgoal_idx += 1
                        step_idx += len(sg_lookup.actions)
                        continue

                # 5. Layer 3 Direct Action Token Inference (~533ms baseline)
                history_summary = session.get_history_summary()
                try:
                    action, plan_latency_ms = self.agent.step(
                        observation=obs,
                        goal=active_subgoal.description,
                        history=history_summary,
                        step_index=step_idx,
                    )
                except Exception as e:
                    logger.error(f"Planning error at step {step_idx}: {e}")
                    action = Action(kind=ActionKind.WAIT, duration=1.0, reason=f"Planning error: {e}")
                    plan_latency_ms = 0.0

                # 6. Repetition & Stuck Detection / Jittering & Dynamic Replanning
                repetition_count = session.get_repetition_count(new_action=action)
                recent_matches = session.count_recent_matches(action, window=5)
                effective_reps = max(repetition_count, recent_matches)
                if effective_reps >= 2:
                    stuck_points.append({
                        "step": step_idx,
                        "action": action.to_dict(),
                        "repetitions": effective_reps,
                        "subgoal": active_subgoal.description,
                    })
                    if dry_run:
                        logger.warning(
                            f"[DRY-RUN] Repetition detected ({effective_reps + 1}x for {action.kind.value} {action.coordinate or ''}). "
                            "Screen state remains static because hardware actuation is bypassed in dry-run mode."
                        )
                    if action.kind in (
                        ActionKind.LEFT_CLICK,
                        ActionKind.RIGHT_CLICK,
                        ActionKind.DOUBLE_CLICK,
                        ActionKind.TRIPLE_CLICK,
                    ) and action.coordinate:
                        orig_x, orig_y = action.coordinate
                        jitter_x = 10 if orig_x < 1270 else -10
                        jitter_y = -10 if orig_y >= 790 else 10
                        new_coord = (orig_x + jitter_x, orig_y + jitter_y)
                        logger.warning(
                            f"Loop detected: Click action repeated {effective_reps + 1} times. "
                            f"Jittering position from {action.coordinate} to {new_coord}."
                        )
                        action = Action(
                            kind=action.kind,
                            coordinate=new_coord,
                            reason=f"Loop breakout: Jittering click position from {action.coordinate} to {new_coord}",
                        )

                    diag = ReflectionDiagnostic(
                        failed_subgoal=active_subgoal,
                        failure_reason=f"Loop deadlock: action {action.kind.value} repeated {effective_reps} times without UI progress",
                        stuck_repetitions=effective_reps,
                        stuck_action=action.to_dict(),
                    )
                    replanned = self.planner.replan(
                        goal=goal,
                        failed_subgoal=active_subgoal,
                        failure_reason=diag.failure_reason,
                        diagnostic=diag,
                        stuck_points=stuck_points,
                    )
                    if replanned:
                        logger.warning(
                            f"Layer 1 Dynamic Replanning: Synthesized {len(replanned)} alternative sub-goals: "
                            f"{[r.description for r in replanned]}"
                        )
                        for offset, rsg in enumerate(replanned, start=active_subgoal.index):
                            rsg.index = offset
                        subgoals = subgoals[:subgoal_idx] + replanned
                        active_subgoal = subgoals[subgoal_idx]

                # 7. Layer 4 Execution Safety Firewall evaluation
                cls_name = ""
                if self.user32:
                    try:
                        fg_hwnd = self.user32.GetForegroundWindow()
                        if fg_hwnd:
                            buf = ctypes.create_unicode_buffer(256)
                            self.user32.GetClassNameW(fg_hwnd, buf, 256)
                            cls_name = buf.value
                    except Exception:
                        cls_name = ""

                firewall_res = self.firewall.evaluate(action, active_window_class=cls_name, user_confirmed=user_confirmed)
                if not firewall_res.allowed:
                    logger.warning(firewall_res.reason)
                    blocked_act = Action(
                        kind=ActionKind.WAIT,
                        duration=0.5,
                        reason=firewall_res.reason,
                    )
                    session.record_step(blocked_act, observation=obs, plan_latency_ms=plan_latency_ms)
                    # Trigger dynamic replan if firewall blocks critical action
                    if firewall_res.requires_confirmation and not user_confirmed:
                        diag = ReflectionDiagnostic(
                            failed_subgoal=active_subgoal,
                            failure_reason=firewall_res.reason,
                        )
                        replanned = self.planner.replan(
                            goal=goal,
                            failed_subgoal=active_subgoal,
                            failure_reason=firewall_res.reason,
                            diagnostic=diag,
                        )
                        if replanned:
                            for offset, rsg in enumerate(replanned, start=active_subgoal.index):
                                rsg.index = offset
                            subgoals = subgoals[:subgoal_idx] + replanned
                    step_idx += 1
                    continue

                logger.info(f"Planned Action: {action.kind.value} {action.coordinate or ''} | Reason: {action.reason} | Latency: {plan_latency_ms:.1f}ms")

                # 8. Record step in session
                is_subgoal_stop = (action.kind == ActionKind.STOP and (subgoal_idx + 1 < len(subgoals)))
                session.record_step(
                    action,
                    observation=obs,
                    plan_latency_ms=plan_latency_ms,
                    is_subgoal_stop=is_subgoal_stop,
                )

                # 9. Check Termination Condition & Sub-Goal Advancement with Layer 5 Dual-Stage Verifier
                if action.kind == ActionKind.STOP:
                    stage2_res = self.verifier.verify_semantic(
                        expected_window_title=expected_window,
                        expected_process_name=expected_process,
                        goal=active_subgoal.description,
                    )
                    if stage2_res.passed or dry_run:
                        if hasattr(active_subgoal, "mark_completed"):
                            active_subgoal.mark_completed()
                        logger.info(f"Sub-goal {active_subgoal.index} completed: {active_subgoal.description} ({action.reason})")
                        subgoal_idx += 1
                        if subgoal_idx >= len(subgoals):
                            logger.info(f"Goal completed: {action.reason}")
                            speak_feedback(f"Task completed: {action.reason}", enabled=self.enable_voice)
                            break
                        else:
                            logger.info(f"Advancing to next sub-goal {subgoal_idx + 1}/{len(subgoals)}: '{subgoals[subgoal_idx].description}'")
                            step_idx += 1
                            continue
                    else:
                        logger.warning(f"Sub-goal {active_subgoal.index} verification failed: {stage2_res.details}. Replanning...")
                        if hasattr(active_subgoal, "mark_failed"):
                            active_subgoal.mark_failed(stage2_res.details)
                        diag = ReflectionDiagnostic(
                            failed_subgoal=active_subgoal,
                            failure_reason=stage2_res.details,
                        )
                        replanned = self.planner.replan(
                            goal=goal,
                            failed_subgoal=active_subgoal,
                            failure_reason=stage2_res.details,
                            diagnostic=diag,
                        )
                        if replanned:
                            for offset, rsg in enumerate(replanned, start=active_subgoal.index):
                                rsg.index = offset
                            subgoals = subgoals[:subgoal_idx] + replanned
                        step_idx += 1
                        continue

                # 10. Physical Win32 Hardware Execution (Minimum-Jerk)
                native_target = None
                if action.coordinate:
                    native_target = self.computer._canonical_to_native(*action.coordinate)
                elif action.start_coordinate:
                    native_target = self.computer._canonical_to_native(*action.start_coordinate)

                self.check_safety_breakout(target_coord=native_target)

                if dry_run:
                    logger.info(f"[DRY-RUN] Simulated execution for {action.kind.value} {action.coordinate or ''} (hardware actuation bypassed)")
                    self.computer.execute(action, dry_run=True)
                else:
                    self.computer.execute(action, dry_run=False)

                if native_target is not None:
                    self._last_actuation_coord = native_target

                # Layer 5 Stage 1 Physical Verification (optical delta / click ripple)
                if not dry_run and action.kind in (
                    ActionKind.LEFT_CLICK,
                    ActionKind.RIGHT_CLICK,
                    ActionKind.DOUBLE_CLICK,
                    ActionKind.TRIPLE_CLICK,
                ):
                    time.sleep(0.04)
                    obs_after = self.computer.observe(format=image_format)
                    stage1_res = self.verifier.verify_physical(obs, obs_after, action=action)
                    last_stage1_res = stage1_res
                    if stage1_res.ripple_detected:
                        logger.info(
                            f"Visual closed-loop verification: Visual delta confirmed after {action.kind.value} at {action.coordinate}."
                        )
                    else:
                        logger.warning(
                            f"Visual closed-loop verification: No screen change detected after {action.kind.value} at {action.coordinate}."
                        )

                # Post-action settling delay
                if action.kind == ActionKind.KEY and action.keys and "win" in action.keys:
                    time.sleep(0.20)
                else:
                    time.sleep(0.08)
                step_idx += 1

        except Exception:
            raise
        finally:
            streamer.stop()
            try:
                self.computer.emergency_release_hardware()
            except Exception:
                pass

        if subgoal_idx >= len(subgoals) and not session.is_complete:
            session.finish("All planned sub-goals completed successfully.")
        elif not session.is_complete and step_idx > max_steps:
            session.finish(f"Maximum step limit ({max_steps}) reached.")
            speak_feedback("Autopilot reached maximum step limit.", enabled=self.enable_voice)

        session.subgoals = [s.to_dict() if hasattr(s, "to_dict") else s for s in subgoals]

        # Layer 5 Stage 2 Semantic Verification & Trajectory Logging
        stage2_res = self.verifier.verify_semantic(
            expected_window_title=expected_window,
            expected_process_name=expected_process,
            goal=goal,
        )
        verified_success = session.is_complete and not session.is_stuck(threshold=2) and (stage2_res.passed or dry_run)
        fail_reason = "" if verified_success else (stage2_res.details if not stage2_res.passed else session.stop_reason)

        if last_stage1_res is None:
            last_stage1_res = Stage1Result(
                passed=True,
                optical_delta=0.0,
                latency_ms=0.0,
                ripple_detected=False,
                details="Dry run or non-click session; physical verification bypassed.",
            )

        last_verification = DualStageVerificationResult(
            stage1=last_stage1_res,
            stage2=stage2_res,
            overall_success=verified_success,
            details=stage2_res.details,
        )

        metadata = {
            "subgoals": [s.to_dict() if hasattr(s, "to_dict") else s for s in subgoals],
        }

        self.trajectory_logger.log_trajectory(
            session=session,
            verified_success=verified_success,
            failure_reason=fail_reason,
            stuck_points=stuck_points,
            verification_result=last_verification,
            metadata=metadata,
        )

        metrics = session.get_metrics()
        logger.info(f"Session finished: {metrics}")
        return session


def main() -> None:
    """CLI Entrypoint for the Ace Autopilot."""
    parser = argparse.ArgumentParser(
        description="AVCS Ace-Control-Medium Autonomous Visual GUI Autopilot."
    )
    parser.add_argument("--goal", type=str, required=True, help="User goal to execute on the desktop.")
    parser.add_argument("--endpoint", type=str, default=None, help="Custom LLM API endpoint URL.")
    parser.add_argument("--model", type=str, default=None, help="Frontier model identifier (e.g. gpt-6-astra, fable-5.5).")
    parser.add_argument("--api-key", type=str, default=None, help="API key for custom LLM provider.")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum execution steps before aborting.")
    parser.add_argument("--format", type=str, default="jpeg", choices=["webp", "jpeg", "png"], help="Screen capture compression format.")
    parser.add_argument("--protocol", type=str, default=None, choices=["openai", "anthropic", "gemini", "auto"], help="API wire protocol.")
    parser.add_argument("--timeout", type=float, default=None, help="HTTP request timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Observe and plan without dispatching hardware mouse/keyboard events.")
    parser.add_argument("--no-voice", action="store_true", help="Disable natural voice speech feedback.")

    args = parser.parse_args()

    client = CustomLLMClient(
        endpoint=args.endpoint,
        model=args.model,
        api_key=args.api_key,
        protocol=args.protocol,
        timeout=args.timeout,
    )
    agent = AceAgent(client=client)
    computer = WindowsComputer()
    runner = AceRunner(agent=agent, computer=computer, enable_voice=not args.no_voice)

    try:
        runner.run(
            goal=args.goal,
            max_steps=args.max_steps,
            image_format=args.format,
            dry_run=args.dry_run,
        )
    except SafetyBreakoutError as e:
        logger.warning(f"EMERGENCY BREAKOUT: {e}")
        speak_feedback("Autopilot emergency breakout triggered by user.")
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Autopilot stopped by user via KeyboardInterrupt.")
        sys.exit(0)


if __name__ == "__main__":
    main()

