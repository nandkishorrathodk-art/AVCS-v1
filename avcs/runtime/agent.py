"""Ace-Control-Medium Cognitive Agent.

Orchestrates the visual perception to direct action token generation pipeline:
1. Fast Brain: Local GoalConditionedGrounder for sub-15ms direct spatial actuation.
2. Gated Actuation: Directly delegates when grounder peak_probability >= min_peak_probability (0.75).
3. Slow Brain: Fallback to Cloud LLM for complex reasoning, typing, and recovery.
4. Shadow Mode: Runs local grounder in shadow, compares coordinates against Cloud LLM, and logs agreement.
5. Canvas Invariant: Enforces canonical 1280x800 coordinate boundaries and zero center heuristics.
"""
from __future__ import annotations

import io
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from avcs.protocol.actions import Action, ActionKind
from avcs.runtime.client import CustomLLMClient
from avcs.os_io.computer import Observation

logger = logging.getLogger("avcs.ace.agent")


class AceAgentSession:
    """Stateful Agent Session implementing the generalagents-python session contract."""

    def __init__(
        self,
        agent: "AceAgent",
        instruction: str = "",
        max_steps: int = 30,
        dry_run: bool = False,
    ) -> None:
        self.agent = agent
        self.instruction = instruction
        self.max_steps = max_steps
        self.dry_run = dry_run
        self.step_count = 0
        self.is_closed = False
        self.is_complete = False
        self.stop_reason = ""
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.steps: List[Dict[str, Any]] = []
        self.last_observation: Optional[Observation] = None
        self.last_action: Optional[Action] = None

    @property
    def history(self) -> List[Dict[str, Any]]:
        """History of steps taken in this session."""
        return self.steps

    def start(self, instruction: str) -> "AceAgentSession":
        """Re-initializes session with new instruction/goal."""
        self.instruction = instruction
        self.step_count = 0
        self.is_closed = False
        self.is_complete = False
        self.stop_reason = ""
        self.start_time = time.time()
        self.end_time = None
        self.steps = []
        self.last_observation = None
        self.last_action = None
        return self

    def plan(self, observation: Observation) -> Action:
        """Plans the next physical Action given desktop Observation, tracking state."""
        if self.is_closed:
            raise RuntimeError("Cannot plan on a closed session. Call reset() or start() first.")

        if self.is_complete:
            return Action(kind=ActionKind.STOP, reason=self.stop_reason or "Session already complete.")

        self.last_observation = observation
        step_idx = self.step_count + 1

        history = [
            {"step": s["step"], "action": s["action"], "reason": s.get("reason", "")}
            for s in self.steps[-5:]
        ]

        action, latency_ms = self.agent.step(
            observation=observation,
            goal=self.instruction,
            history=history,
            step_index=step_idx,
        )

        self.step_count = step_idx
        self.last_action = action
        self.steps.append({
            "step": self.step_count,
            "action": action.to_dict(),
            "latency_ms": latency_ms,
            "timestamp": time.time(),
            "reason": action.reason,
        })

        if action.kind == ActionKind.STOP:
            self.is_complete = True
            self.stop_reason = action.reason or "Task finished."
            self.end_time = time.time()
        elif self.step_count >= self.max_steps:
            self.is_complete = True
            self.stop_reason = f"Max step limit ({self.max_steps}) reached."
            self.end_time = time.time()

        return action

    def reset(self) -> None:
        """Resets session state, step counts, and history."""
        self.step_count = 0
        self.is_closed = False
        self.is_complete = False
        self.stop_reason = ""
        self.start_time = time.time()
        self.end_time = None
        self.steps = []
        self.last_observation = None
        self.last_action = None

    def close(self) -> None:
        """Closes the session and marks it inactive."""
        self.is_closed = True
        self.is_complete = True
        if not self.stop_reason:
            self.stop_reason = "Session closed by caller"
        self.end_time = time.time()

    def __enter__(self) -> "AceAgentSession":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class AceAgent:
    """Pure-Vision GUI Autopilot Agent implementing the Ace-Control-Medium operational model."""

    CANONICAL_WIDTH = 1280
    CANONICAL_HEIGHT = 800

    def __init__(
        self,
        client: Optional[CustomLLMClient] = None,
        grounder: Optional[Any] = None,
        enable_grounder: bool = False,
        shadow_mode: bool = False,
        min_peak_probability: float = 0.75,
        checkpoint_path: Optional[str] = "runs/weights/goal_grounder_best.pt",
        embedded_runtime: Optional[Any] = None,
        vla_client: Optional[Any] = None,
        vla_server_url: Optional[str] = None,
    ) -> None:
        self.client = client or CustomLLMClient()
        self.grounder = grounder
        self.enable_grounder = enable_grounder or (grounder is not None)
        self.shadow_mode = shadow_mode
        self.min_peak_probability = min_peak_probability
        self.checkpoint_path = checkpoint_path
        self._grounder_load_attempted = False
        self._embedded_runtime = embedded_runtime
        self._embedded_runtime_attempted = False
        self.shadow_history: List[Dict[str, Any]] = []
        self.current_session: Optional[AceAgentSession] = None

        # Fast Brain remote VLA client (/ws/act persistent streaming)
        self.vla_client = vla_client
        if self.vla_client is None:
            v_url = vla_server_url or os.environ.get("AVCS_VLA_SERVER_URL", "")
            if v_url:
                try:
                    from avcs.runtime.vla_client import VLAClient
                    self.vla_client = VLAClient(server_url=v_url, use_websocket=True)
                    logger.info(f"AceAgent: Initialized Fast Brain WebSocket VLA client to {v_url} (/ws/act)")
                except Exception as e:
                    logger.warning(f"AceAgent: Failed to initialize VLAClient for {v_url}: {e}")

    def _get_embedded_runtime(self) -> Optional[Any]:
        """Lazily initializes the in-process EmbeddedVAMRuntime for fast reflexive actions."""
        if getattr(self, "_embedded_runtime", None) is not None:
            return self._embedded_runtime
        if getattr(self, "_embedded_runtime_attempted", False):
            return None
        self._embedded_runtime_attempted = True
        try:
            from avcs.models.local_model import EmbeddedVAMRuntime
            self._embedded_runtime = EmbeddedVAMRuntime(checkpoint_path=self.checkpoint_path)
            logger.info("AceAgent: Initialized EmbeddedVAMRuntime for fast reflexive execution")
            return self._embedded_runtime
        except Exception as e:
            logger.debug(f"AceAgent: EmbeddedVAMRuntime load failed: {e}")
            return None

    def _get_grounder(self) -> Optional[Any]:
        """Lazily loads the local GoalConditionedGrounder checkpoint."""
        if self.grounder is not None:
            return self.grounder
        if not self.enable_grounder and not self.shadow_mode:
            return None
        if self._grounder_load_attempted:
            return None
        self._grounder_load_attempted = True
        if self.checkpoint_path:
            p = Path(self.checkpoint_path)
            if p.exists():
                try:
                    from avcs.goal_grounder import load_grounding_checkpoint
                    self.grounder = load_grounding_checkpoint(str(p), device="cpu")
                    self.grounder.eval()
                    logger.info(f"AceAgent: Loaded local GoalConditionedGrounder from {p}")
                except Exception as e:
                    logger.warning(f"AceAgent: Failed to load grounder checkpoint {p}: {e}")
        return self.grounder

    def predict_with_grounder(
        self,
        observation: Observation,
        goal: str,
        step_index: int = 1,
    ) -> Tuple[Optional[Action], float]:
        """Runs local GoalConditionedGrounder inference on visual observation and natural language goal."""
        grounder = self._get_grounder()
        has_image = bool(
            observation is not None
            and (
                getattr(observation, "image_bytes", None)
                or (hasattr(observation, "raw_numpy") and observation.raw_numpy is not None)
            )
        )
        if observation is None or not has_image:
            return None, 0.0

        # Fast path rejection: Local Grounder only performs spatial pointing/clicking.
        # Pure keyboard actions, waiting, scrolling, or typing must be delegated to Cloud LLM / Slow Brain.
        g_clean = (goal or "").strip()
        is_pure_non_pointing = bool(
            re.search(
                r'^\s*(?:type|write|input|enter\s+text)\b'
                r'|^\s*(?:press|hit|hotkey|shortcut|key)\s+(?:key\s+)?(?:enter|return|esc|escape|win|windows|super|tab|backspace|delete|space|ctrl|alt|shift|up|down|left|right|home|end|pageup|pagedown|pgup|pgdn|insert|capslock|numlock|f[1-9]|f1[0-2]|prtscr|printscreen)\b'
                r'|^\s*(?:press|hit|hotkey|shortcut)\s+(?:ctrl|alt|shift|win|windows)\s*[\+\-]'
                r'|^\s*(?:wait|sleep|delay)\b'
                r'|^\s*(?:scroll|wheel)\s+(?:up|down|left|right)\b'
                r'|^\s*(?:stop|finish|done|exit|quit|complete)\b',
                g_clean,
                re.IGNORECASE,
            )
        )
        if is_pure_non_pointing:
            logger.debug(f"predict_with_grounder: Bypassing local grounder for non-pointing goal '{goal}'")
            return None, 0.0

        t0 = time.perf_counter()
        try:
            if hasattr(observation, "raw_numpy") and isinstance(observation.raw_numpy, np.ndarray):
                frame_np = observation.raw_numpy
            else:
                try:
                    img = Image.open(io.BytesIO(observation.image_bytes)).convert("RGB")
                    frame_np = np.array(img)
                except Exception:
                    w = getattr(observation, "width", 1280) or 1280
                    h = getattr(observation, "height", 800) or 800
                    frame_np = np.zeros((h, w, 3), dtype=np.uint8)

            # Clean query by stripping action verbs, prepositions, and articles
            clean_query = (goal or "").strip()
            clean_query = re.sub(
                r'^\s*(?:double[-\s]*click|right[-\s]*click|triple[-\s]*click|click|press|select|open|launch|hit|tap|focus)\s+(?:(?:on|at|to|in)\s+)?(?:(?:the|a|an)\s+)?',
                '',
                clean_query,
                flags=re.IGNORECASE,
            ).strip()
            clean_query = re.sub(
                r'\s+(?:double[-\s]*click|right[-\s]*click|triple[-\s]*click|click|press|select|open|launch|hit|tap|focus)\s*$',
                '',
                clean_query,
                flags=re.IGNORECASE,
            ).strip()

            clean_query = clean_query.strip(" '\".,;:-")

            if not clean_query:
                clean_query = (goal or "").strip()

            cand_pt = None
            peak_prob = 0.0

            if grounder is not None:
                if hasattr(grounder, "predict"):
                    # GoalConditionedGrounder.predict signature: (frame_rgb: np.ndarray, query: str)
                    try:
                        pred = grounder.predict(frame_np, clean_query)
                    except TypeError:
                        try:
                            pred = grounder.predict(frame_rgb=frame_np, query=clean_query)
                        except TypeError:
                            pred = grounder.predict(clean_query, frame=frame_np)
                    cand_pt = getattr(pred, "candidate_point_norm", None)
                    raw_prob = getattr(pred, "peak_probability", 0.0)
                    peak_prob = float(raw_prob) if raw_prob is not None else 0.0
                elif hasattr(grounder, "locate"):
                    try:
                        target = grounder.locate(clean_query, frame=frame_np)
                    except TypeError:
                        target = grounder.locate(clean_query)
                    if target is not None:
                        cand_pt = target.centroid if getattr(target, "centroid", None) is not None else None
                        raw_prob = getattr(target, "confidence", 0.0)
                        peak_prob = float(raw_prob) if raw_prob is not None else 0.0

            if cand_pt is not None:
                cx_norm, cy_norm = cand_pt
                if (
                    cx_norm is not None
                    and cy_norm is not None
                    and not math.isnan(cx_norm)
                    and not math.isnan(cy_norm)
                    and not math.isinf(cx_norm)
                    and not math.isinf(cy_norm)
                ):
                    pred_x = int(np.clip(round(cx_norm * self.CANONICAL_WIDTH), 0, self.CANONICAL_WIDTH - 1))
                    pred_y = int(np.clip(round(cy_norm * self.CANONICAL_HEIGHT), 0, self.CANONICAL_HEIGHT - 1))

                    is_triple = bool(re.search(r'\b(?:triple[-\s]*click|tripleclick)\b', goal, re.IGNORECASE))
                    is_double = bool(re.search(r'\b(?:double[-\s]*click|doubleclick)\b', goal, re.IGNORECASE))
                    is_right = bool(re.search(r'\b(?:right[-\s]*click|rightclick)\b', goal, re.IGNORECASE))
                    act_kind = (
                        ActionKind.TRIPLE_CLICK if is_triple else (
                            ActionKind.DOUBLE_CLICK if is_double else (
                                ActionKind.RIGHT_CLICK if is_right else ActionKind.LEFT_CLICK
                            )
                        )
                    )

                    act = Action(
                        kind=act_kind,
                        coordinate=(pred_x, pred_y),
                        confidence=peak_prob,
                        reason=f"Local Grounder target '{clean_query}' (peak_prob={peak_prob:.3f})",
                        metadata={
                            "fast_brain": True,
                            "grounder_peak_prob": round(peak_prob, 4),
                            "clean_query": clean_query,
                        },
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    return act, latency_ms

            # Fast reflexive execution via EmbeddedVAMRuntime before slow Cloud LLM fallback
            runtime = self._get_embedded_runtime()
            if runtime is not None:
                try:
                    img_bytes = getattr(observation, "image_bytes", None)
                    if not img_bytes and hasattr(observation, "raw_numpy") and isinstance(observation.raw_numpy, np.ndarray):
                        buf = io.BytesIO()
                        Image.fromarray(observation.raw_numpy).save(buf, format="JPEG")
                        img_bytes = buf.getvalue()

                    if img_bytes:
                        ref_act, ref_lat = runtime.predict(img_bytes, goal, step_index=step_index)
                        if (
                            ref_act is not None
                            and ref_act.kind != ActionKind.WAIT
                            and not str(getattr(ref_act, "reason", "")).startswith("Awaiting UI response")
                            and not (ref_act.kind == ActionKind.STOP and str(getattr(ref_act, "reason", "")).startswith("Goal completed:"))
                        ):
                            if getattr(ref_act, "metadata", None) is None:
                                ref_act.metadata = {}
                            ref_act.metadata["fast_brain"] = True
                            ref_act.metadata["reflexive"] = True
                            latency_ms = (time.perf_counter() - t0) * 1000.0
                            return ref_act, latency_ms
                except Exception as e:
                    logger.debug(f"predict_with_grounder: EmbeddedVAMRuntime reflexive execution error: {e}")

            latency_ms = (time.perf_counter() - t0) * 1000.0
            return None, latency_ms
        except Exception as e:
            logger.debug(f"predict_with_grounder exception: {e}")
            return None, (time.perf_counter() - t0) * 1000.0

    def start(
        self,
        instruction: str = "",
        max_steps: int = 30,
        dry_run: bool = False,
    ) -> AceAgentSession:
        """Initiates a stateful session matching the generalagents-python contract."""
        session = AceAgentSession(
            agent=self,
            instruction=instruction,
            max_steps=max_steps,
            dry_run=dry_run,
        )
        self.current_session = session
        return session

    def plan(self, observation: Observation) -> Action:
        """Plans next action using the active session."""
        if self.current_session is None or self.current_session.is_closed:
            self.current_session = self.start(instruction="")
        return self.current_session.plan(observation)

    def reset(self) -> None:
        """Resets active session state."""
        if self.current_session is not None:
            self.current_session.reset()

    def close(self) -> None:
        """Closes active session."""
        if self.current_session is not None:
            self.current_session.close()
            self.current_session = None

    def clamp_coordinate(self, coord: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Clamps coordinate within [0, 1280] x [0, 800] canvas without center snapping."""
        if coord is None:
            return None
        try:
            if not isinstance(coord, (list, tuple)) or len(coord) != 2:
                return None
            x, y = coord
            if x is None or y is None:
                return None
            x_val = float(x)
            y_val = float(y)
            if math.isnan(x_val) or math.isnan(y_val) or math.isinf(x_val) or math.isinf(y_val):
                return None
            cx = max(0, min(self.CANONICAL_WIDTH, int(round(x_val))))
            cy = max(0, min(self.CANONICAL_HEIGHT, int(round(y_val))))
            return cx, cy
        except (ValueError, TypeError):
            return None

    def validate_action(self, action: Action) -> Action:
        """Validates action parameters and enforces canvas constraints without center heuristics."""
        if action is None:
            return Action(kind=ActionKind.WAIT, duration=0.5, reason="Null action received.")

        # 1. Clamp single coordinates
        if action.coordinate is not None:
            action.coordinate = self.clamp_coordinate(action.coordinate)

        # 2. Clamp drag coordinates and chunks
        if action.start_coordinate is not None:
            action.start_coordinate = self.clamp_coordinate(action.start_coordinate)
        if action.end_coordinate is not None:
            action.end_coordinate = self.clamp_coordinate(action.end_coordinate)
        if action.drag_chunks:
            clamped_chunks = []
            for pt in action.drag_chunks:
                c = self.clamp_coordinate(pt)
                if c is not None:
                    clamped_chunks.append(c)
            action.drag_chunks = clamped_chunks if clamped_chunks else None

        # 3. Guard against ungrounded spatial actions (Zero center heuristic mandate)
        # Includes LEFT, RIGHT, DOUBLE, TRIPLE, MIDDLE click, and MOUSE_MOVE
        if action.kind in (
            ActionKind.LEFT_CLICK,
            ActionKind.RIGHT_CLICK,
            ActionKind.DOUBLE_CLICK,
            ActionKind.TRIPLE_CLICK,
            ActionKind.MIDDLE_CLICK,
            ActionKind.MOUSE_MOVE,
        ) and action.coordinate is None:
            logger.warning(f"Action {action.kind.value} rejected due to missing coordinates. Never snapping to center.")
            return Action(
                kind=ActionKind.WAIT,
                duration=0.5,
                reason=f"Ungrounded action {action.kind.value}: no coordinates provided.",
            )

        # 4. Guard against ungrounded drag actions
        if action.kind in (ActionKind.LEFT_CLICK_DRAG,) and (
            action.start_coordinate is None or action.end_coordinate is None
        ):
            logger.warning(f"Action {action.kind.value} rejected due to missing drag coordinates. Never snapping to center.")
            return Action(
                kind=ActionKind.WAIT,
                duration=0.5,
                reason=f"Ungrounded action {action.kind.value}: start or end coordinate missing.",
            )

        # 5. Guard against ungrounded scroll actions
        if action.kind == ActionKind.SCROLL and action.scroll_delta is None and action.coordinate is None:
            logger.warning("Action scroll rejected due to missing scroll_delta and coordinate.")
            return Action(
                kind=ActionKind.WAIT,
                duration=0.5,
                reason="Ungrounded action scroll: neither scroll_delta nor coordinate provided.",
            )

        # 6. Sanitize text and key payloads
        if action.kind in (ActionKind.TYPE, ActionKind.TYPE_TEXT):
            if action.text is None:
                action.text = ""

        if action.kind == ActionKind.KEY:
            if action.keys is None:
                action.keys = []
            elif isinstance(action.keys, str):
                action.keys = [action.keys]

        # 7. Sanitize wait duration
        if action.kind == ActionKind.WAIT:
            try:
                action.duration = max(0.0, float(action.duration))
            except (ValueError, TypeError):
                action.duration = 0.5

        return action

    def step(
        self,
        observation: Observation,
        goal: str,
        history: Optional[List[Dict[str, Any]]] = None,
        step_index: int = 1,
        blacklisted_coordinates: Optional[Any] = None,
    ) -> Tuple[Action, float]:
        """Processes visual observation and returns next physical Action with latency in ms."""
        # 1. Gated Actuation Mode: Remote WebSocket VLA (Fast Brain) first if configured
        if self.vla_client is not None and not self.shadow_mode:
            t0 = time.perf_counter()
            try:
                frame_input = getattr(observation, "raw_numpy", None)
                if frame_input is None:
                    raw_bytes = getattr(observation, "image_bytes", None)
                    if raw_bytes:
                        # Zero-copy passthrough if already binary JPEG bytes
                        if len(raw_bytes) >= 2 and raw_bytes[0] == 0xFF and raw_bytes[1] == 0xD8:
                            frame_input = raw_bytes
                        else:
                            frame_input = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

                w = getattr(observation, "width", self.CANONICAL_WIDTH) or self.CANONICAL_WIDTH
                h = getattr(observation, "height", self.CANONICAL_HEIGHT) or self.CANONICAL_HEIGHT

                vla_res = self.vla_client.act(
                    frame=frame_input,
                    instruction=goal,
                    width=w,
                    height=h,
                )
                vla_lat = (time.perf_counter() - t0) * 1000.0

                action_name = str(vla_res.get("action", "left_click")).lower().strip()
                coord = vla_res.get("coordinate")
                target_coord = (int(coord[0]), int(coord[1])) if (isinstance(coord, (list, tuple)) and len(coord) == 2) else None

                is_blacklisted = False
                if target_coord and blacklisted_coordinates:
                    gx, gy = target_coord
                    is_blacklisted = any(
                        abs(gx - bx) <= 5 and abs(gy - by) <= 5
                        for bx, by in blacklisted_coordinates
                    )

                if is_blacklisted:
                    logger.warning(
                        f"[FAST BRAIN VLA FALLBACK] Target coordinate {target_coord} is blacklisted due to prior deadlock. "
                        "Falling back to Cloud LLM."
                    )
                else:
                    act_kind_map = {
                        "left_click": ActionKind.LEFT_CLICK,
                        "right_click": ActionKind.RIGHT_CLICK,
                        "double_click": ActionKind.DOUBLE_CLICK,
                        "triple_click": ActionKind.TRIPLE_CLICK,
                        "mouse_move": ActionKind.MOUSE_MOVE,
                        "wait": ActionKind.WAIT,
                        "stop": ActionKind.STOP,
                        "key": ActionKind.KEY,
                        "type": ActionKind.TYPE,
                        "type_text": ActionKind.TYPE,
                    }
                    act_kind = act_kind_map.get(action_name, ActionKind.LEFT_CLICK)
                    vla_act = Action(
                        kind=act_kind,
                        coordinate=target_coord,
                        confidence=1.0,
                        reason=f"WebSocket VLA Fast Brain '{action_name}' {target_coord} ({vla_res.get('roundtrip_ms', vla_lat):.1f}ms)",
                        metadata={
                            "fast_brain": True,
                            "vla_streaming": True,
                            "roundtrip_ms": vla_res.get("roundtrip_ms", vla_lat),
                            "latency_ms": vla_res.get("latency_ms", 0.0),
                        },
                    )
                    validated = self.validate_action(vla_act)
                    if validated.kind != ActionKind.WAIT or act_kind == ActionKind.WAIT:
                        logger.info(
                            f"[FAST BRAIN DIRECT ACTUATION] WebSocket VLA predicted {validated.kind.value} {validated.coordinate} "
                            f"in {vla_lat:.1f}ms (bypassing Cloud LLM)."
                        )
                        return validated, vla_lat
            except Exception as e:
                logger.warning(f"[FAST BRAIN FALLBACK] WebSocket VLA failed ({e}). Falling back to Cloud LLM.")

        # Local Grounder fallback only if explicitly enabled
        elif self.enable_grounder and not self.shadow_mode:
            grounder_act, grounder_lat = self.predict_with_grounder(
                observation, goal, step_index=step_index
            )

            # Check if Grounder's predicted coordinate is blacklisted due to prior deadlock
            is_blacklisted = False
            if grounder_act and grounder_act.coordinate and blacklisted_coordinates:
                gx, gy = grounder_act.coordinate
                is_blacklisted = any(
                    abs(gx - bx) <= 5 and abs(gy - by) <= 5
                    for bx, by in blacklisted_coordinates
                )

            grounder_conf = (
                float(grounder_act.confidence)
                if (grounder_act and grounder_act.confidence is not None)
                else 0.0
            )

            if is_blacklisted:
                logger.warning(
                    f"[FAST BRAIN FALLBACK] Grounder target coordinate {grounder_act.coordinate} is blacklisted due to prior deadlock. "
                    "Falling back to Cloud LLM for complex recovery."
                )
            elif grounder_act is not None and grounder_conf >= self.min_peak_probability:
                validated = self.validate_action(grounder_act)
                # If validation degraded the action to WAIT (e.g. invalid coordinates), fall back to Cloud LLM
                if validated.kind == ActionKind.WAIT and grounder_act.kind != ActionKind.WAIT:
                    logger.warning(
                        "[FAST BRAIN FALLBACK] Grounder action failed validation constraints. "
                        "Falling back to Cloud LLM."
                    )
                else:
                    logger.info(
                        f"[FAST BRAIN DIRECT ACTUATION] Grounder peak_probability={grounder_conf:.3f} >= {self.min_peak_probability}. "
                        f"Delegating direct action {validated.kind.value} {validated.coordinate} in {grounder_lat:.1f}ms (bypassing Cloud LLM)."
                    )
                    return validated, grounder_lat
            else:
                logger.info(
                    f"[FAST BRAIN FALLBACK] Grounder peak_probability={grounder_conf:.3f} < {self.min_peak_probability}. "
                    "Falling back to Cloud LLM."
                )

        # 2. Cloud LLM Inference (Slow Brain)
        action, latency_ms = self.client.predict_action(
            image_bytes=observation.image_bytes,
            goal=goal,
            history=history,
            step_index=step_index,
            format=observation.format,
        )
        validated_action = self.validate_action(action)

        # 3. Shadow Mode: Run grounder in shadow, compare predictions, and log agreement
        if self.shadow_mode:
            grounder_act, grounder_lat = self.predict_with_grounder(
                observation, goal, step_index=step_index
            )
            llm_coord = validated_action.coordinate or validated_action.start_coordinate
            grounder_coord = grounder_act.coordinate if grounder_act else None
            grounder_conf = (
                float(grounder_act.confidence)
                if (grounder_act and grounder_act.confidence is not None)
                else 0.0
            )

            distance_px = None
            agreed = False
            if llm_coord is not None and grounder_coord is not None:
                try:
                    d = math.hypot(llm_coord[0] - grounder_coord[0], llm_coord[1] - grounder_coord[1])
                    if not math.isnan(d) and not math.isinf(d):
                        distance_px = float(d)
                        agreed = (distance_px <= 50.0)
                except (TypeError, ValueError):
                    pass

            shadow_record = {
                "goal": goal,
                "llm_action": validated_action.kind.value,
                "llm_coordinate": list(llm_coord) if llm_coord else None,
                "grounder_coordinate": list(grounder_coord) if grounder_coord else None,
                "grounder_peak_prob": round(grounder_conf, 4),
                "distance_px": round(distance_px, 2) if distance_px is not None else None,
                "agreed": agreed,
                "grounder_latency_ms": round(grounder_lat, 2),
            }
            self.shadow_history.append(shadow_record)
            logger.info(
                f"[SHADOW MODE] Goal='{goal}' | LLM={llm_coord} vs Grounder={grounder_coord} "
                f"(peak_prob={grounder_conf:.3f}) | distance={distance_px if distance_px is not None else 'N/A'}px | agreed={agreed}"
            )
            if getattr(validated_action, "metadata", None) is None:
                validated_action.metadata = {}
            validated_action.metadata["shadow_comparison"] = shadow_record

        return validated_action, latency_ms

