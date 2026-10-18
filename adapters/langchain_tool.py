"""LangChain & CrewAI Universal Visual Perception Tools.

Provides plug-and-play Agentic Tools for LangChain, CrewAI, AutoGen, and OpenAI
Function Calling architectures. Allows LLM agents to interact with screens
sensory-first without needing heavy screenshot attachments on every step.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from avcs import ScreenEye
from forge.vision.cortex.sdk import UnvalidatedGroundingError


class AVCSScreenTools:
    """Suite of high-level agent tools for LLM frameworks."""

    def __init__(self, eye: Optional[ScreenEye] = None) -> None:
        self.eye = eye or ScreenEye()

    def look_at_element(self, element_description: str) -> str:
        """Directs visual gaze onto a described UI element.

        Args:
            element_description: Natural language description (e.g. 'Submit button', 'Search bar').
        Returns:
            JSON string with normalized (x, y) coordinates and screen category.
        """
        x, y = self.eye.look_at(element_description)
        state = self.eye.current_state or self.eye.perceive_now()
        return json.dumps({
            "target_coordinates": [round(x, 4), round(y, 4)],
            "screen_category": state.primary_category,
            "status": "experimental_unvalidated_gaze_estimate",
        })

    def check_screen_stability(self) -> str:
        """Checks if current screen has finished loading and settled.

        Returns:
            JSON string stating whether screen is stable and safe to interact with.
        """
        is_stable = self.eye.is_stable()
        state = self.eye.current_state or self.eye.perceive_now()
        return json.dumps({
            "is_stable": is_stable,
            "stability_state": state.stability_state,
            "surprise_score": round(state.surprise_score, 4),
        })

    def get_interactive_buttons(self) -> str:
        """Discovers clickable buttons and interactive controls on screen.

        Returns:
            JSON array of top clickable elements with coordinates and confidence.
        """
        buttons = self.eye.clickable_hotspots()
        payload = [
            {"x": round(x, 4), "y": round(y, 4), "confidence": round(conf, 3)}
            for x, y, conf in buttons
        ]
        return json.dumps({"clickable_elements": payload})

    def steer_and_click(self, element_description: str) -> str:
        """Fails closed until semantic grounding is validated for motor actions."""
        raise UnvalidatedGroundingError(
            "Natural-language targets cannot drive motor actions: validated trained "
            "semantic grounding is unavailable."
        )

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Returns standard OpenAI Function Calling schema tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "look_at_element",
                    "description": "Return an experimental, unvalidated gaze estimate for a natural-language query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element_description": {"type": "string", "description": "Name or visual description of the target element."}
                        },
                        "required": ["element_description"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_screen_stability",
                    "description": "Verify if animations or web page loads have finished settling before clicking.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_interactive_buttons",
                    "description": "List top interactive buttons and controls detected by the dorsal visual cortex.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
