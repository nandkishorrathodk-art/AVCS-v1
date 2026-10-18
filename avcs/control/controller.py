"""Central AVCS Controller & Orchestration Hub.

Implements the unified System 1 (Eyes + Hands) interface for System 2 (LLM):
1. Observe -> Ground -> Validate -> Act -> Verify -> Describe -> Replan
2. Zero coordinates exposed to System 2 (internal geometry encapsulation)
3. Zero hallucination: honest, evidence-backed natural language descriptions
4. Dynamic neural grounding via GoalConditionedGrounder / Cross-Attention
5. Biological 5th-order minimum-jerk kinematics & fail-closed typing protection
6. Configurable local telemetry streaming
"""
from __future__ import annotations

import dataclasses
import logging
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from avcs.perception.events import VisualEvent, VisualEventStreamer, VisualEventType
from avcs.perception.focus import (
    FocusVerificationResult,
    VisualFocusDetector,
    VisualFocusState,
)
from avcs.goal_grounder import (
    GoalConditionedGrounder,
    load_grounding_checkpoint,
)
from avcs.perception.grounding import FovealInspection, GroundingResult
from avcs.perception.human_eye import HumanEye, HumanEyeState
from avcs.control.servoing import (
    NeuromuscularServoingController,
    NeuromuscularServoingResult,
)
from avcs.control.substep import (
    AutonomousGoalExecutor,
    GoalExecutionResult,
    SubstepTrace,
)
from avcs.stream.neuro_stream import NeuroPacket, NeuroStreamBus
from avcs.control.verifier import (
    VerificationResult,
    VerificationStatus,
    VisualRippleVerifier,
)
from avcs.observation import EntitySummary, SceneObservation, VisualObservationEngine
from forge.vision.capture import ScreenCapture, SyntheticScreen, open_capture
from forge.vision.cortex.motor import get_default_driver
from forge.vision.cortex.sdk import UnvalidatedGroundingError

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ActionResult:
    """Action outcome returned to System 2 without exposing raw coordinates."""
    action: str
    target_query: str
    success: bool
    verified: bool
    message: str
    latency_ms: float
    verification: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "action": self.action,
            "target_query": self.target_query,
            "success": self.success,
            "verified": self.verified,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.verification:
            res["verification"] = self.verification
        return res


@dataclasses.dataclass(frozen=True)
class System2Turn:
    """A text-only reasoning round trip; the returned goal is not auto-executed."""

    observation: SceneObservation
    goal: str


# Ergonomic & backward compatibility aliases
ActionOutcome = ActionResult
ScreenObservation = SceneObservation

__all__ = [
    "SceneObservation",
    "ScreenObservation",
    "ActionResult",
    "ActionOutcome",
    "System2Turn",
    "SubstepTrace",
    "GoalExecutionResult",
    "AVCSController",
]


class AVCSController:
    """The central orchestrator for AVCS System 1 (Eyes & Hands).
    
    Coordinates perception, neural grounding, foveal acuity, biological motor
    actuation, postcondition verification, and locally measured streaming.
    """

    def __init__(
        self,
        source: Optional[ScreenCapture | str] = None,
        weights_path: Optional[str] = "runs/weights/goal_grounder_best.pt",
        device: str = "cpu",
        target_fps: int = 60,
        force_virtual_motor: bool = False,
        enable_streaming: bool = True,
        require_real_capture: bool = False,
        motor_driver: Optional[Any] = None,
    ) -> None:
        if isinstance(target_fps, bool) or not isinstance(target_fps, int) or target_fps <= 0:
            raise ValueError("target_fps must be a positive integer")
        if require_real_capture and source is not None and source != "desktop":
            raise ValueError("require_real_capture is only valid for the default desktop source")
        self.device = torch.device(device)
        self.target_fps = target_fps
        self.force_virtual_motor = force_virtual_motor
        self.weights_path = Path(weights_path) if weights_path else None

        # 1. Screen Capture
        if source is None or source == "desktop":
            self.source = open_capture(
                prefer_real=True,
                threaded=True,
                allow_synthetic_fallback=not require_real_capture,
            )
        elif hasattr(source, "grab"):
            self.source = source
        else:
            self.source = open_capture(prefer_real=False, threaded=False, allow_synthetic_fallback=True)

        # 2. Perception (Eyes)
        self.eye = HumanEye(source=self.source, device=device)

        # 3. Motor Actuation (Hands)
        if motor_driver is not None:
            self.motor_driver = motor_driver
        else:
            self.motor_driver = get_default_driver(force_virtual=force_virtual_motor)
        self.servoing = NeuromuscularServoingController(driver=self.motor_driver, eye=self.eye)

        # 4. Postcondition Verifier & Action Broker
        self.verifier = VisualRippleVerifier(eye=self.eye, driver=self.motor_driver)
        from avcs.control.action_broker import ActionBroker, ActionSafetyController
        self.safety = ActionSafetyController()
        self.broker = ActionBroker(eye=self.eye, driver=self.motor_driver, safety_controller=self.safety)

        # 5. Real-Time Telemetry Bus & Differential Events
        self.stream_bus = NeuroStreamBus(buffer_size=120) if enable_streaming else None
        if self.stream_bus is not None:
            if hasattr(self.eye, "attach_stream_bus"):
                self.eye.attach_stream_bus(self.stream_bus)
            if hasattr(self.servoing, "attach_stream_bus"):
                self.servoing.attach_stream_bus(self.stream_bus)
        self.event_streamer = VisualEventStreamer()
        self.observation_engine = VisualObservationEngine()

        # 6. Neural Grounding Model (Cross-Attention / GoalConditionedGrounder)
        self.grounder: Optional[GoalConditionedGrounder] = None
        self._load_grounder_weights()

        self._frame_seq = 0
        self._last_observation: Optional[SceneObservation] = None
        self._stream_stop = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_error: Optional[BaseException] = None
        self._stream_started_at: Optional[float] = None
        self._stream_ended_at: Optional[float] = None
        self._stream_frames_processed = 0
        self._stream_frames_dropped = 0
        self._stream_tick_latencies_ms: deque[float] = deque(maxlen=512)
        self._stream_tick_intervals_ms: deque[float] = deque(maxlen=512)
        self._stream_has_source_sequence = (
            callable(getattr(self.source, "grab_with_sequence", None))
            and not isinstance(self.source, SyntheticScreen)
        )

    @property
    def capture_backend(self) -> str:
        """Names the actual selected capture backend without exposing frame data."""
        backend = getattr(self.source, "source", self.source)
        return type(backend).__name__

    @property
    def stream_error(self) -> Optional[BaseException]:
        """Returns a streaming worker failure for explicit caller handling."""
        return self._stream_error

    @property
    def stream_metrics(self) -> Dict[str, Any]:
        """Returns measured throughput and bounded full-tick latency percentiles."""
        end = self._stream_ended_at or time.perf_counter()
        elapsed = (
            max(0.0, end - self._stream_started_at)
            if self._stream_started_at is not None
            else 0.0
        )
        tick_latencies = tuple(self._stream_tick_latencies_ms)
        tick_intervals = tuple(self._stream_tick_intervals_ms)
        latency_percentiles = (
            np.percentile(tick_latencies, (50, 95)).tolist()
            if tick_latencies
            else (None, None)
        )
        interval_percentiles = (
            np.percentile(tick_intervals, (50, 95)).tolist()
            if tick_intervals
            else (None, None)
        )
        return {
            "frames_processed": self._stream_frames_processed,
            "frames_dropped": (
                self._stream_frames_dropped
                if self._stream_has_source_sequence
                else None
            ),
            "elapsed_seconds": elapsed,
            "processing_fps": (
                self._stream_frames_processed / elapsed if elapsed > 0 else 0.0
            ),
            "processing_tick_latency_ms": {
                "p50": latency_percentiles[0],
                "p95": latency_percentiles[1],
                "samples_retained": len(tick_latencies),
            },
            "processing_interval_ms": {
                "p50": interval_percentiles[0],
                "p95": interval_percentiles[1],
                "samples_retained": len(tick_intervals),
            },
            "capture_backend": self.capture_backend,
        }

    def _load_grounder_weights(self) -> bool:
        """Loads trained neural grounding weights if present and valid."""
        candidates = []
        if self.weights_path is not None:
            candidates.append(self.weights_path)
            default_paths = {
                Path("runs/weights/candidates/candidate_m3_highres_v2.pt"),
                Path("runs/weights/goal_grounder_best.pt"),
            }
            if self.weights_path in default_paths:
                for alt in (
                    Path("runs/weights/candidates/candidate_m3_highres_v2.pt"),
                    Path("runs/weights/goal_grounder_best.pt"),
                    Path("runs/weights/human_eye_weights.pt"),
                ):
                    if alt not in candidates:
                        candidates.append(alt)
        else:
            candidates.extend([
                Path("runs/weights/candidates/candidate_m3_highres_v2.pt"),
                Path("runs/weights/goal_grounder_best.pt"),
                Path("runs/weights/human_eye_weights.pt"),
            ])
        for p in candidates:
            if p and p.is_file():
                try:
                    self.grounder = load_grounding_checkpoint(p, device=self.device)
                    if self.grounder.is_trained:
                        return True
                except Exception:
                    self.grounder = None
        return False

    def load_grounder_weights(self) -> bool:
        """Loads trained neural grounding weights if present and valid."""
        return self._load_grounder_weights()

    @property
    def is_grounder_trained(self) -> bool:
        """Returns True if a trained neural grounding model is active."""
        return self.grounder is not None and getattr(self.grounder, "is_trained", False)

    # --- System 2 Intent Interface ---

    def describe_screen(
        self,
        frame: Optional[np.ndarray] = None,
        *,
        frame_sequence: Optional[int] = None,
    ) -> SceneObservation:
        """Observes the current display and emits an honest, evidence-backed description.
        
        Zero hallucination: only reports verified UI categories, optical motion,
        and visible interactive affordances.
        """
        t0 = time.time()
        if frame is None:
            grab_with_sequence = getattr(self.source, "grab_with_sequence", None)
            if callable(grab_with_sequence) and not isinstance(self.source, SyntheticScreen):
                frame, captured_sequence = grab_with_sequence()
                if frame_sequence is None:
                    frame_sequence = captured_sequence
            else:
                frame = self.source.grab()
        self._frame_seq += 1
        state = self.eye.observe(frame=frame, frame_sequence=frame_sequence)

        cat = getattr(state, "primary_category", "unknown")
        cat_clean = cat.replace("_", " ").title()
        stability = getattr(state, "stability", getattr(state, "stability_state", "stable"))
        has_modal = "modal" in cat or getattr(state, "has_modal_dialog", False)
        affordances = getattr(state, "affordance_hotspots", getattr(state, "affordances", []))
        inf_ms = getattr(state, "latency_ms", getattr(state, "inference_time_ms", 10.0))

        # Broadcast telemetry packet with local and capture sequence provenance.
        # Broadcast telemetry packet with local and capture sequence provenance (if eye did not broadcast it).
        if self.stream_bus is not None and getattr(self.eye, "stream_bus", None) is not self.stream_bus:
            packet = NeuroPacket(
                timestamp=t0,
                seq=self._frame_seq,
                fixation=self.eye.pursuit.current_gaze,
                motor_flow=(0.0, 0.0),
                contact_acquired=False,
                primary_category=cat,
                stability_state=stability,
                surprise_score=state.surprise_score,
                inference_ms=inf_ms,
                action_hotspots=affordances,
                source_frame_sequence=frame_sequence,
            )
            self.stream_bus.publish(packet)

        # Process differential events
        self.event_streamer.process_state(state)

        # Synthesize truthful natural language observation
        obs = self.observation_engine.generate(
            state=state,
            frame=frame,
            prior_action="observe",
        )
        if frame_sequence is not None:
            obs = dataclasses.replace(
                obs,
                frame_sequence=frame_sequence,
                frame_id=f"frame_{frame_sequence}",
            )
        self._last_observation = obs
        return obs

    def observe(
        self,
        frame: Optional[np.ndarray] = None,
        *,
        frame_sequence: Optional[int] = None,
    ) -> SceneObservation:
        """Observes the display and returns a privacy-preserving SceneObservation to System 2."""
        return self.describe_screen(frame=frame, frame_sequence=frame_sequence)

    def consult_system2(
        self,
        task: str,
        client: Any,
        *,
        consent_confirmed: bool = False,
    ) -> System2Turn:
        """Send an allowlisted text observation to System 2 and return its text goal.

        This method never dispatches mouse or keyboard input. The caller must
        review and separately authorize any resulting local action.
        """
        if consent_confirmed is not True:
            raise PermissionError(
                "System 2 sharing requires explicit user consent from the calling application"
            )
        observation = self.describe_screen()
        goal = client.request_goal(
            task,
            observation,
            consent_confirmed=True,
        )
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("System 2 client returned an empty or invalid goal")
        return System2Turn(observation=observation, goal=goal.strip())

    def start_streaming(self) -> None:
        """Continuously perceive new local frames and publish compact telemetry.

        Threaded capture sources are gated by their monotonically increasing
        frame sequence, so the same buffered frame is never reported twice.
        """
        if self.stream_bus is None:
            raise RuntimeError("Streaming is disabled for this controller")
        if self._stream_thread is not None and self._stream_thread.is_alive():
            raise RuntimeError("AVCS streaming is already running")

        self._stream_stop.clear()
        self._stream_error = None
        self._stream_started_at = time.perf_counter()
        self._stream_ended_at = None
        self._stream_frames_processed = 0
        self._stream_frames_dropped = 0
        self._stream_tick_latencies_ms.clear()
        self._stream_tick_intervals_ms.clear()
        self._stream_has_source_sequence = (
            callable(getattr(self.source, "grab_with_sequence", None))
            and not isinstance(self.source, SyntheticScreen)
        )
        self._stream_thread = threading.Thread(
            target=self._stream_worker,
            name="AVCS-System1-Stream",
            daemon=True,
        )
        self._stream_thread.start()

    def _stream_worker(self) -> None:
        interval = 1.0 / self.target_fps
        last_tick = 0.0
        last_tick_started: Optional[float] = None
        last_capture_sequence: Optional[int] = None
        grab_with_sequence = getattr(self.source, "grab_with_sequence", None)
        try:
            while not self._stream_stop.is_set():
                delay = interval - (time.perf_counter() - last_tick)
                if delay > 0 and self._stream_stop.wait(delay):
                    break

                tick_started = time.perf_counter()
                if callable(grab_with_sequence) and not isinstance(self.source, SyntheticScreen):
                    frame, capture_sequence = grab_with_sequence()
                    if last_capture_sequence is not None:
                        if capture_sequence <= last_capture_sequence:
                            self._stream_stop.wait(min(interval / 4.0, 0.002))
                            continue
                        self._stream_frames_dropped += max(
                            0, capture_sequence - last_capture_sequence - 1
                        )
                    last_capture_sequence = capture_sequence
                    self.describe_screen(
                        frame=frame,
                        frame_sequence=capture_sequence,
                    )
                else:
                    frame = self.source.grab()
                    self.describe_screen(frame=frame)

                last_tick = time.perf_counter()
                self._stream_tick_latencies_ms.append(
                    (last_tick - tick_started) * 1000.0
                )
                if last_tick_started is not None:
                    self._stream_tick_intervals_ms.append(
                        (tick_started - last_tick_started) * 1000.0
                    )
                last_tick_started = tick_started
                self._stream_frames_processed += 1
        except Exception as exc:
            self._stream_error = exc
            self._stream_stop.set()
            logger.exception("AVCS local streaming worker failed")
        finally:
            self._stream_ended_at = time.perf_counter()

    def stop_streaming(self, timeout: float = 5.0) -> None:
        """Stops the local stream and surfaces worker failures to the caller."""
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive number")
        thread = self._stream_thread
        if thread is None:
            return
        self._stream_stop.set()
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise TimeoutError("AVCS streaming worker did not stop before the timeout")
        self._stream_thread = None
        if self._stream_error is not None:
            raise RuntimeError("AVCS streaming worker failed") from self._stream_error

    def locate(self, query: str, frame: Optional[np.ndarray] = None) -> GroundingResult:
        """Locates a natural-language query on screen using the trained neural grounder.
        
        Coordinates remain internal to AVCS. System 2 does not receive or manage them.
        """
        clean_q = query.strip()
        if not clean_q:
            raise ValueError("Query string cannot be empty")

        t0 = time.perf_counter()
        if frame is None:
            frame = self.source.grab()

        if self.grounder is not None:
            prediction = self.grounder.predict(frame, clean_q)
            if not prediction.action_authorized or prediction.candidate_point_norm is None:
                raise UnvalidatedGroundingError(
                    f"Cannot locate '{clean_q}' for motor action: the grounder returned "
                    "an unauthorized candidate. Calibrated target-absence and "
                    "abstention evidence is required before actions are enabled."
                )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return GroundingResult(
                query=clean_q,
                target_coord=prediction.candidate_point_norm,
                confidence=prediction.peak_probability,
                relevance_map=prediction.heatmap.cpu().numpy(),
                snapped_to_affordance=False,
                latency_ms=dt_ms,
            )
        else:
            raise UnvalidatedGroundingError(
                f"Cannot locate '{clean_q}': trained neural grounding checkpoint is not loaded. "
                "Train the model first."
            )

    def click_target(
        self,
        query: str,
        double_click: bool = False,
        verify_action: bool = True,
        enable_micro_tremor: Optional[bool] = None,
    ) -> ActionResult:
        """Locates target, moves naturally (5th-order minimum-jerk), clicks, and verifies postcondition.
        
        System 2 supplies only the high-level intent query. Geometry is encapsulated.
        """
        t0 = time.perf_counter()
        clean_q = query.strip()

        # 1. Ground target
        try:
            grounding = self.locate(clean_q)
        except UnvalidatedGroundingError as exc:
            if not self.is_grounder_trained:
                raise
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action="double_click" if double_click else "click",
                target_query=clean_q,
                success=False,
                verified=False,
                message=f"Target '{clean_q}' could not be localized with sufficient confidence ({exc}).",
                latency_ms=dt_ms,
            )
        if grounding.confidence < 0.20:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action="double_click" if double_click else "click",
                target_query=clean_q,
                success=False,
                verified=False,
                message=f"Target '{clean_q}' could not be localized with sufficient confidence ({grounding.confidence:.2f} < 0.20).",
                latency_ms=dt_ms,
            )

        tx, ty = grounding.target_coord

        # 2. Foveal refinement for sub-pixel boundary docking
        try:
            foveal_res = self.eye.inspect_fovea(centroid=(tx, ty))
            if foveal_res.confidence >= 0.5:
                # Snap to refined foveal centroid
                tx, ty = foveal_res.refined_centroid
        except Exception:
            pass

        # 3. Smooth Biological Motor Movement
        if enable_micro_tremor is not None:
            try:
                self.servoing.servo_to((tx, ty), realtime=False, enable_micro_tremor=enable_micro_tremor)
            except TypeError:
                self.servoing.servo_to((tx, ty), realtime=False)
        else:
            self.servoing.servo_to((tx, ty), realtime=False)

        # 4. Dispatch once through the verifier, which owns the action lifecycle.
        verification_dict = None
        verified = False
        if verify_action:
            v_res = self.verifier.execute_and_verify(
                action_type="double_click" if double_click else "click",
                target=(tx, ty),
                # Clicks can submit or toggle state, so never replay them automatically.
                max_recovery_tier=1,
            )
            verified = v_res.status == VerificationStatus.VERIFIED
            verification_dict = {
                "status": v_res.status.value,
                "action_name": v_res.action_name,
                "optical_ripple_energy": v_res.optical_ripple_energy,
                "surprise_pulse": v_res.surprise_pulse,
                "state_transition_detected": v_res.state_transition_detected,
                "causal_summary": v_res.causal_summary,
                "recovery_attempt": v_res.recovery_attempt,
                "elapsed_ms": v_res.elapsed_ms,
            }
        elif double_click:
            self.motor_driver.double_click(button="left")
        else:
            self.motor_driver.click(button="left")

        dt_ms = (time.perf_counter() - t0) * 1000.0
        status_msg = (
            f"Successfully clicked and verified target '{clean_q}'."
            if verified else
            (
                f"Action for '{clean_q}' was dispatched without verification; "
                "completion is unconfirmed."
                if not verify_action else
                f"Action for '{clean_q}' was dispatched, but postcondition verification "
                "did not confirm completion."
            )
        )

        return ActionResult(
            action="double_click" if double_click else "click",
            target_query=clean_q,
            success=verified,
            verified=verified,
            message=status_msg,
            latency_ms=dt_ms,
            verification=verification_dict,
        )

    def open_application(self, app_name: str) -> ActionResult:
        """High-level intent to open an application by icon or menu, and verify window launch."""
        res = self.click_target(f"{app_name} icon", double_click=True, verify_action=True)
        return ActionResult(
            action="open_application",
            target_query=app_name,
            success=res.success,
            verified=res.verified,
            message=res.message,
            latency_ms=res.latency_ms,
            verification=res.verification,
        )

    def verify_focus(
        self,
        target_region: Optional[Tuple[float, float, float, float]] = None,
        foreground_app: Optional[str] = None,
        target_point: Optional[Tuple[float, float]] = None,
    ) -> FocusVerificationResult:
        """Evaluates the dual-condition typing safety gate:
        Condition A (OS State): Foreground window matches foreground_app (if given).
        Condition B (Visual Pixels): Target region/point exhibits visual focus (conf >= 0.70 and caret blinking or focus ring).
        """
        t0 = time.perf_counter()

        # 1. Condition A: OS / Window State Check
        os_verified = True
        active_win = None
        if hasattr(self.motor_driver, "get_foreground_window"):
            active_win = self.motor_driver.get_foreground_window()
        elif sys.platform == "win32" and not self.force_virtual_motor:
            try:
                import win32gui
                hwnd = win32gui.GetForegroundWindow()
                if hwnd:
                    active_win = win32gui.GetWindowText(hwnd)
            except Exception:
                pass

        if foreground_app is not None:
            if active_win is None or foreground_app.lower() not in active_win.lower():
                os_verified = False

        # 2. Condition B: Pure-Pixel Visual Focus Check
        point = target_point
        if point is None:
            if target_region is not None:
                x1, y1, x2, y2 = target_region
                point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            else:
                point = getattr(self.eye.pursuit, "current_gaze", (0.5, 0.5))

        st = self.eye.verify_focus(target_point=point)
        visual_is_focused = getattr(st, "is_focused", False)
        visual_conf = getattr(st, "confidence", 0.0)
        caret_blink = getattr(st, "caret_blinking", True)

        visual_verified = bool(visual_is_focused and visual_conf >= 0.70 and caret_blink)

        verified = bool(os_verified and visual_verified)
        final_conf = visual_conf if verified else min(visual_conf, 0.69)

        if verified:
            reason = "Focus confirmed"
        elif not os_verified:
            reason = f"Window mismatch (expected '{foreground_app}', got '{active_win}')"
        elif not caret_blink:
            reason = f"visual focus verification failed closed (caret blink absent, conf={visual_conf:.2f})"
        else:
            reason = f"visual focus verification failed closed (conf={visual_conf:.2f})"

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return FocusVerificationResult(
            verified=verified,
            confidence=final_conf,
            os_window_verified=os_verified,
            visual_focus_verified=visual_verified,
            active_window_title=active_win,
            expected_window_title=foreground_app,
            caret_detected=getattr(st, "caret_detected", False),
            caret_blinking=caret_blink,
            focus_ring_detected=getattr(st, "focus_ring_detected", False),
            caret_coord=getattr(st, "caret_coord", None),
            target_region=target_region,
            reason=reason,
            latency_ms=dt_ms,
        )

    def type_text(
        self,
        text: str,
        verify_focus: bool = True,
        clear_first: bool = False,
        press_enter: bool = False,
        target_field: Optional[str] = None,
        target_query: Optional[str] = None,
        verify_action: bool = False,
        foreground_app: Optional[str] = None,
    ) -> ActionResult:
        """Types text with fail-closed visual focus verification."""
        t0 = time.perf_counter()
        field = target_field or target_query

        # If a target input field was named, locate and click into it first
        if field:
            try:
                click_res = self.click_target(field)
            except UnvalidatedGroundingError as exc:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                return ActionResult(
                    action="type_text",
                    target_query=field,
                    success=False,
                    verified=False,
                    message=f"Typing blocked: target input field could not be grounded ({exc}).",
                    latency_ms=dt_ms,
                )
            if not click_res.success:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                return ActionResult(
                    action="type_text",
                    target_query=field,
                    success=False,
                    verified=False,
                    message=f"Could not focus target input field '{field}'.",
                    latency_ms=dt_ms,
                )

        current_gaze = self.eye.pursuit.current_gaze
        # Dual-Condition Focus Gate
        if verify_focus:
            foc_res = self.verify_focus(foreground_app=foreground_app, target_point=current_gaze)
            if not foc_res.verified:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                return ActionResult(
                    action="type_text",
                    target_query=field or "current_focus",
                    success=False,
                    verified=False,
                    message=f"Typing blocked: {foc_res.reason}.",
                    latency_ms=dt_ms,
                    verification=foc_res.to_dict(),
                )

        # Safe Clearing (Eliminate blind Ctrl+A + Backspace; Terminal Protection Guard)
        if clear_first:
            active_win = None
            if hasattr(self.motor_driver, "get_foreground_window"):
                active_win = self.motor_driver.get_foreground_window()
            elif sys.platform == "win32" and not self.force_virtual_motor:
                try:
                    import win32gui
                    hwnd = win32gui.GetForegroundWindow()
                    if hwnd:
                        active_win = win32gui.GetWindowText(hwnd)
                except Exception:
                    pass
            active_str = (active_win or foreground_app or "").lower()
            is_terminal = any(term in active_str for term in ["terminal", "powershell", "cmd", "bash", "zsh"])
            if is_terminal:
                self.motor_driver.press_hotkey("ctrl", "u")
            else:
                self.motor_driver.press_hotkey("ctrl", "a")
                self.motor_driver.press_key("backspace")

        pre_state = self.eye.observe() if verify_action else None

        # Mid-typing focus loss check callback
        def _focus_check() -> bool:
            if foreground_app is not None:
                cur_win = None
                if hasattr(self.motor_driver, "get_foreground_window"):
                    cur_win = self.motor_driver.get_foreground_window()
                if cur_win is not None and foreground_app.lower() not in cur_win.lower():
                    return False
            return True

        if hasattr(self.motor_driver, "type_text"):
            try:
                self.motor_driver.type_text(text, focus_check_fn=_focus_check)
            except TypeError:
                self.motor_driver.type_text(text)
        if press_enter:
            self.motor_driver.press_key("enter")

        verification_dict = None
        verified = False
        if verify_action:
            v_res = self.verifier.verify_action(
                action_name="type_text",
                target=current_gaze,
                pre_state=pre_state,
                settle_ms=40.0,
            )
            verified = v_res.status == VerificationStatus.VERIFIED
            verification_dict = {
                "status": v_res.status.value,
                "action_name": v_res.action_name,
                "optical_ripple_energy": v_res.optical_ripple_energy,
                "surprise_pulse": v_res.surprise_pulse,
                "state_transition_detected": v_res.state_transition_detected,
                "causal_summary": v_res.causal_summary,
                "recovery_attempt": v_res.recovery_attempt,
                "elapsed_ms": v_res.elapsed_ms,
            }

        dt_ms = (time.perf_counter() - t0) * 1000.0
        status_msg = (
            f"Successfully typed and verified text input into '{field or 'current_focus'}'."
            if verified else
            (
                "Text input was dispatched after a visual focus check, but the "
                "resulting screen content was not verified."
                if not verify_action else
                "Text input was dispatched after a visual focus check, but postcondition "
                "verification did not confirm content rendering."
            )
        )

        return ActionResult(
            action="type_text",
            target_query=target_field or "current_focus",
            success=verified,
            verified=verified,
            message=status_msg,
            latency_ms=dt_ms,
            verification=verification_dict,
        )

    def execute_goal(
        self,
        goal_description: str,
        max_substeps: int = 10,
        timeout_s: float = 30.0,
        context: Optional[Dict[str, Any]] = None,
        substep_callback: Optional[Callable[[SubstepTrace], None]] = None,
    ) -> GoalExecutionResult:
        """Autonomously executes a high-level goal through local substeps (Look -> Ground -> Act -> Verify).

        System 2 delegates only the high-level natural language goal. AVCS System 1
        autonomously figures out the required on-screen substeps, executes them with
        its hands, verifies each step visually, and returns a natural language summary.

        Zero coordinates are ever exposed to System 2.
        """
        executor = AutonomousGoalExecutor()
        return executor.execute(
            controller=self,
            goal_description=goal_description,
            max_substeps=max_substeps,
            timeout_s=timeout_s,
            context=context,
            substep_callback=substep_callback,
        )

    def get_telemetry_stream(self) -> Optional[NeuroStreamBus]:
        """Provides the local telemetry bus; measured rate depends on fresh frames."""
        return self.stream_bus

    def close(self) -> None:
        """Releases all capture and model resources."""
        try:
            self.stop_streaming()
        finally:
            self.eye.close()
