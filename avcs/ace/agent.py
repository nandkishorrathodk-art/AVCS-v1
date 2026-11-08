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
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from avcs.ace.actions import Action, ActionKind
from avcs.ace.client import CustomLLMClient
from avcs.ace.computer import Observation

logger = logging.getLogger("avcs.ace.agent")


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
    ) -> None:
        self.client = client or CustomLLMClient()
        self.grounder = grounder
        self.enable_grounder = enable_grounder or (grounder is not None)
        self.shadow_mode = shadow_mode
        self.min_peak_probability = min_peak_probability
        self.checkpoint_path = checkpoint_path
        self._grounder_load_attempted = False
        self.shadow_history: List[Dict[str, Any]] = []

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
    ) -> Tuple[Optional[Action], float]:
        """Runs local GoalConditionedGrounder inference on visual observation and natural language goal."""
        grounder = self._get_grounder()
        if grounder is None or observation is None or not observation.image_bytes:
            return None, 0.0

        # Fast path rejection: Local Grounder only performs spatial pointing/clicking.
        # Pure keyboard actions, waiting, or typing must be delegated to Cloud LLM / Slow Brain.
        g_clean = (goal or "").strip()
        is_pure_non_pointing = bool(
            re.search(
                r'^\s*(?:press|hit)\s+(?:enter|return|esc|escape|win|windows|super|tab|backspace|delete|space|ctrl|alt|shift)\b',
                g_clean,
                re.IGNORECASE,
            )
            or re.search(r'^\s*(?:wait|sleep|delay)\b', g_clean, re.IGNORECASE)
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

            # Clean query by stripping action verbs
            clean_query = re.sub(
                r'^(?:click|press|select|open|double\s*click|right\s*click|triple\s*click|hit|tap)\s+(?:on\s+)?(?:the\s+)?',
                '',
                goal,
                flags=re.IGNORECASE,
            ).strip()
            # Also clean Hindi/Hinglish action directives
            clean_query = re.sub(
                r'\b(?:open\s*karo|kholo|chalao|daba\s*do|dabao|press\s*karo|click\s*karo|kijiye)\b',
                '',
                clean_query,
                flags=re.IGNORECASE,
            ).strip()
            if not clean_query:
                clean_query = goal

            cand_pt = None
            peak_prob = 0.0

            if hasattr(grounder, "predict"):
                # GoalConditionedGrounder.predict signature: (frame_rgb: np.ndarray, query: str)
                try:
                    pred = grounder.predict(frame_np, clean_query)
                except TypeError:
                    try:
                        pred = grounder.predict(frame_rgb=frame_np, query=clean_query)
                    except TypeError:
                        pred = grounder.predict(clean_query, frame=frame_np)
                cand_pt = pred.candidate_point_norm
                peak_prob = float(pred.peak_probability)
            elif hasattr(grounder, "locate"):
                try:
                    target = grounder.locate(clean_query, frame=frame_np)
                except TypeError:
                    target = grounder.locate(clean_query)
                if target is not None:
                    cand_pt = target.centroid if getattr(target, "centroid", None) is not None else None
                    peak_prob = float(getattr(target, "confidence", 0.0))

            latency_ms = (time.perf_counter() - t0) * 1000.0

            if cand_pt is None:
                return None, latency_ms

            cx_norm, cy_norm = cand_pt
            pred_x = int(np.clip(round(cx_norm * self.CANONICAL_WIDTH), 0, self.CANONICAL_WIDTH - 1))
            pred_y = int(np.clip(round(cy_norm * self.CANONICAL_HEIGHT), 0, self.CANONICAL_HEIGHT - 1))

            is_triple = bool(re.search(r'\btriple\b', goal, re.IGNORECASE))
            is_double = bool(re.search(r'\bdouble\b', goal, re.IGNORECASE))
            is_right = bool(re.search(r'\bright\b', goal, re.IGNORECASE))
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
            return act, latency_ms
        except Exception as e:
            logger.debug(f"predict_with_grounder exception: {e}")
            return None, (time.perf_counter() - t0) * 1000.0

    def clamp_coordinate(self, coord: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Clamps coordinate within [0, 1280] x [0, 800] canvas without center snapping."""
        if coord is None:
            return None
        x, y = coord
        cx = max(0, min(self.CANONICAL_WIDTH, x))
        cy = max(0, min(self.CANONICAL_HEIGHT, y))
        return cx, cy

    def validate_action(self, action: Action) -> Action:
        """Validates action parameters and enforces canvas constraints without center heuristics."""
        # 1. Clamp single coordinates
        if action.coordinate is not None:
            action.coordinate = self.clamp_coordinate(action.coordinate)

        # 2. Clamp drag coordinates
        if action.start_coordinate is not None:
            action.start_coordinate = self.clamp_coordinate(action.start_coordinate)
        if action.end_coordinate is not None:
            action.end_coordinate = self.clamp_coordinate(action.end_coordinate)

        # 3. Guard against ungrounded click actions (Zero center heuristic mandate)
        if action.kind in (
            ActionKind.LEFT_CLICK,
            ActionKind.RIGHT_CLICK,
            ActionKind.DOUBLE_CLICK,
            ActionKind.TRIPLE_CLICK,
        ) and action.coordinate is None:
            logger.warning(f"Action {action.kind.value} rejected due to missing coordinates. Never snapping to center.")
            return Action(
                kind=ActionKind.WAIT,
                duration=0.5,
                reason=f"Ungrounded action {action.kind.value}: no coordinates provided.",
            )

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
        # 1. Gated Actuation Mode: Local Fast Brain first if enabled and not in shadow mode
        if self.enable_grounder and not self.shadow_mode:
            grounder_act, grounder_lat = self.predict_with_grounder(observation, goal)

            # Check if Grounder's predicted coordinate is blacklisted due to prior deadlock
            is_blacklisted = False
            if grounder_act and grounder_act.coordinate and blacklisted_coordinates:
                gx, gy = grounder_act.coordinate
                is_blacklisted = any(
                    abs(gx - bx) <= 5 and abs(gy - by) <= 5
                    for bx, by in blacklisted_coordinates
                )

            if is_blacklisted:
                logger.warning(
                    f"[FAST BRAIN FALLBACK] Grounder target coordinate {grounder_act.coordinate} is blacklisted due to prior deadlock. "
                    "Falling back to Cloud LLM for complex recovery."
                )
            elif grounder_act is not None and getattr(grounder_act, "confidence", 0.0) >= self.min_peak_probability:
                logger.info(
                    f"[FAST BRAIN DIRECT ACTUATION] Grounder peak_probability={grounder_act.confidence:.3f} >= {self.min_peak_probability}. "
                    f"Delegating direct action {grounder_act.kind.value} {grounder_act.coordinate} in {grounder_lat:.1f}ms (bypassing Cloud LLM)."
                )
                validated = self.validate_action(grounder_act)
                return validated, grounder_lat
            else:
                conf = getattr(grounder_act, "confidence", 0.0) if grounder_act else 0.0
                logger.info(
                    f"[FAST BRAIN FALLBACK] Grounder peak_probability={conf:.3f} < {self.min_peak_probability}. "
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
            grounder_act, grounder_lat = self.predict_with_grounder(observation, goal)
            llm_coord = validated_action.coordinate
            grounder_coord = grounder_act.coordinate if grounder_act else None
            grounder_conf = getattr(grounder_act, "confidence", 0.0) if grounder_act else 0.0

            distance_px = None
            agreed = False
            if llm_coord is not None and grounder_coord is not None:
                distance_px = float(math.hypot(llm_coord[0] - grounder_coord[0], llm_coord[1] - grounder_coord[1]))
                agreed = (distance_px <= 50.0)

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
            if hasattr(validated_action, "metadata"):
                validated_action.metadata["shadow_comparison"] = shadow_record

        return validated_action, latency_ms

