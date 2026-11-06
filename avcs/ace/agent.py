"""Ace-Control-Medium Cognitive Agent.

Orchestrates the visual perception to direct action token generation pipeline.
Enforces canonical 1280x800 coordinate boundaries and zero center heuristics.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from avcs.ace.actions import Action, ActionKind
from avcs.ace.client import CustomLLMClient
from avcs.ace.computer import Observation

logger = logging.getLogger("avcs.ace.agent")


class AceAgent:
    """Pure-Vision GUI Autopilot Agent implementing the Ace-Control-Medium operational model."""

    CANONICAL_WIDTH = 1280
    CANONICAL_HEIGHT = 800

    def __init__(self, client: Optional[CustomLLMClient] = None) -> None:
        self.client = client or CustomLLMClient()

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
    ) -> Tuple[Action, float]:
        """Processes visual observation and returns next physical Action with latency in ms."""
        action, latency_ms = self.client.predict_action(
            image_bytes=observation.image_bytes,
            goal=goal,
            history=history,
            step_index=step_index,
            format=observation.format,
        )
        validated_action = self.validate_action(action)
        return validated_action, latency_ms

