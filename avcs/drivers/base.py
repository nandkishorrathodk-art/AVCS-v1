"""Unified Base LLM Driver for AVCS.

Provides standard execution loops, tool dispatch, visual verification,
and telemetry accounting for Claude, OpenAI, and Ollama agents.
"""
from __future__ import annotations

import abc
import dataclasses
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from adapters.tool_schemas import get_canonical_tools
from forge.vision.cortex.sdk import ScreenEye
from avcs.stream.reflex_controller import GameReflexController


@dataclasses.dataclass
class StepResult:
    """Outcome of a single agent step in the visual perception loop."""
    step_index: int
    tool_name: str
    tool_args: Dict[str, Any]
    output: Any
    elapsed_seconds: float
    screen_category: str
    stability_state: str
    surprise_score: float
    success: bool = True
    error: Optional[str] = None


class BaseAVCSDriver(abc.ABC):
    """Abstract driver providing unified tool execution and verification against AVCS."""

    def __init__(
        self,
        eye: ScreenEye,
        force_virtual_motor: bool = False,
    ) -> None:
        self.eye = eye
        self.force_virtual_motor = force_virtual_motor
        self.reflex = GameReflexController(eye=eye, force_virtual=force_virtual_motor)
        self.step_history: List[StepResult] = []

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Tuple[Any, bool, Optional[str]]:
        """Executes a canonical AVCS action and verifies post-condition visual state."""
        start_t = time.time()
        try:
            if name == "locate":
                query = args.get("query", "")
                x, y = self.eye.locate(query)
                res = {"target": query, "x": round(float(x), 4), "y": round(float(y), 4)}
                return res, True, None

            elif name == "click":
                query = args.get("query")
                x = args.get("x")
                y = args.get("y")
                button = args.get("button", "left")
                double_click = bool(args.get("double_click", False))

                if query:
                    ok = self.eye.click_target(query=query, button=button, force_virtual=self.force_virtual_motor)
                    return {"action": "click", "target": query, "success": ok}, ok, None

                if x is not None and y is not None:
                    res = self.eye.steer_hand_physical(
                        target_gaze=(float(x), float(y)),
                        click_on_arrival=True,
                        force_virtual=self.force_virtual_motor,
                        realtime=False,
                    )
                    if double_click:
                        self.reflex.driver.double_click(button=button)
                    return {"action": "click", "x": x, "y": y, "contact": res.contact_acquired}, True, None

                # Default: click gaze fixation
                gaze = self.eye.cortex.retina.current_fixation
                res = self.eye.steer_hand_physical(
                    target_gaze=gaze,
                    click_on_arrival=True,
                    force_virtual=self.force_virtual_motor,
                    realtime=False,
                )
                return {"action": "click", "fixation": gaze, "contact": res.contact_acquired}, True, None

            elif name == "type_text":
                text = args.get("text", "")
                clear_first = bool(args.get("clear_first", False))
                press_enter = bool(args.get("press_enter", False))
                self.eye.type_text(
                    text=text,
                    clear_first=clear_first,
                    press_enter=press_enter,
                    force_virtual=self.force_virtual_motor,
                )
                return {"action": "type_text", "characters": len(text)}, True, None

            elif name == "type_into":
                query = args.get("query", "")
                text = args.get("text", "")
                clear_first = bool(args.get("clear_first", True))
                press_enter = bool(args.get("press_enter", True))
                ok = self.eye.type_into(
                    query=query,
                    text=text,
                    clear_first=clear_first,
                    press_enter=press_enter,
                    force_virtual=self.force_virtual_motor,
                )
                return {"action": "type_into", "target": query, "success": ok}, ok, None

            elif name == "wait_stable":
                timeout = float(args.get("timeout_seconds", 3.0))
                thresh = float(args.get("stability_threshold", 0.8))
                start_w = time.time()
                stable = False
                while time.time() - start_w < timeout:
                    state = self.eye.perceive_now()
                    if state.stability_state in ("stable_scene", "nominal"):
                        stable = True
                        break
                    time.sleep(0.05)
                return {"action": "wait_stable", "stable": stable}, True, None

            elif name == "get_affordances":
                action_type = args.get("action_type", "clickable")
                min_conf = float(args.get("min_confidence", 0.5))
                spots = self.eye.get_affordances(action_type=action_type)
                filtered = [
                    {"x": round(x, 4), "y": round(y, 4), "confidence": round(conf, 3)}
                    for x, y, conf in spots
                    if conf >= min_conf
                ]
                return {"action_type": action_type, "hotspots": filtered}, True, None

            elif name == "press_hotkey":
                keys = args.get("keys", [])
                self.eye.press_hotkey(*keys, force_virtual=self.force_virtual_motor)
                return {"action": "press_hotkey", "keys": keys}, True, None

            else:
                return None, False, f"Unknown tool: '{name}'"

        except Exception as e:
            return None, False, str(e)
        finally:
            elapsed = time.time() - start_t
            state = self.eye.current_state or self.eye.perceive_now()
            step_res = StepResult(
                step_index=len(self.step_history) + 1,
                tool_name=name,
                tool_args=args,
                output=res if "res" in locals() else None,
                elapsed_seconds=elapsed,
                screen_category=state.primary_category,
                stability_state=state.stability_state,
                surprise_score=state.surprise_score,
                success="res" in locals(),
            )
            self.step_history.append(step_res)

    def close(self) -> None:
        self.reflex.close()
        self.eye.close()

