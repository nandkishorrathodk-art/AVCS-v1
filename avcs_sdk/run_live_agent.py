"""[6] run_live_agent.py: End-to-End Autonomous Local VLA Agent Runner.

Single-command entrypoint to launch the 100% offline Single-Brain Local VLA pipeline:
1. EyeStream (100 FPS Live Retina + ConvGRU Surprise Detector)
2. FoveaReader (High-acuity central pixel zoom; pixel-only, no OCR)
3. LiveNarrator (Local sensory commentary for state telemetry)
4. HandMotor (Biological Minimum-Jerk Mouse Servoing + Ripple Verifier)
5. LLMBridge (Local VLA Engine Bridge — zero external cloud LLM calls)
6. CognitivePlanner (Hierarchical Goal Decomposition & Multi-Step Reasoning)
7. WorkflowCoordinator (Cross-Application Context & Window Switching)
8. ObstacleRecoveryManager (Dynamic Recovery on Modals, Locks & Syntax Errors)

Usage:
    python run_live_agent.py --goal "Check screen and describe active window"
    python run_live_agent.py --monitor --seconds 10
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import List, Optional, Tuple

# Ensure safe UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .eye_stream import EyeStream, UI_CATEGORIES
from .fovea_reader import FovealElement, FoveaReader
from .hand_motor import HandMotor
from .llm_bridge import LLMBridge
from .narrator import LiveNarrator
from .cognitive import (
    ApplicationRole,
    CognitivePlanner,
    ObstacleRecoveryManager,
    ObstacleType,
    WorkflowCoordinator,
)

# Category tokens the eye stream actually emits. The scoring below used to test
# for "text_like"/"graphic", which no producer emits, so those branches were
# dead. Derive from the single vocabulary source so the two cannot drift.
TEXT_CATEGORIES = frozenset({"text_input"})
INTERACTIVE_CATEGORIES = UI_CATEGORIES - TEXT_CATEGORIES - {"content_region"}


def resolve_semantic_target(
    target_query: Optional[str],
    elements: List[FovealElement],
) -> Optional[FovealElement]:
    """Semantically resolve the best matching FovealElement for a cognitive directive.

    Matches query keywords against element categories, visual texture metrics
    (glyphs, edges, contrast), and spatial quadrants without hardcoded coordinates.
    """
    if not elements:
        return None
    if not target_query:
        return elements[0]

    q_lower = target_query.lower()

    def score_element(elem: FovealElement) -> float:
        score = 0.0
        # 1. Category matching
        if elem.category.lower() in q_lower:
            score += 3.0

        # 2. Text / code / syntax matching. Dense, high-contrast detail is the
        # pixel evidence for glyph-like content; both metrics come from the
        # foveal crop, not from a phantom attribute.
        if any(w in q_lower for w in ("code", "syntax", "text", "line", "type", "script")):
            if elem.category in TEXT_CATEGORIES:
                score += 3.0
            score += elem.edge_density * 0.15
            score += elem.contrast * 0.05

        # 3. Interactive controls / buttons / dialogs
        if any(w in q_lower for w in ("button", "dismiss", "ok", "cancel", "close", "dialog", "click", "control")):
            if elem.category in INTERACTIVE_CATEGORIES:
                score += 3.0
            score += elem.contrast * 0.05

        # 4. Spatial quadrant matching
        cx, cy = elem.center_norm
        if "top" in q_lower and cy < 0.35:
            score += 2.0
        if "bottom" in q_lower and cy > 0.65:
            score += 2.0
        if "left" in q_lower and cx < 0.35:
            score += 2.0
        if "right" in q_lower and cx > 0.65:
            score += 2.0
        if "center" in q_lower and (0.25 <= cx <= 0.75 and 0.25 <= cy <= 0.75):
            score += 2.0

        # 5. Base detection confidence
        score += float(elem.confidence)
        return score

    scored = [(score_element(e), e) for e in elements]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1]


def _resolve_screen_point(
    eye: EyeStream,
    elements: List[FovealElement],
    coord: Optional[Tuple[int, int]],
    target_desc: Optional[str],
) -> Optional[Tuple[Tuple[int, int], Tuple[float, float], str]]:
    """Resolve canonical VLA coordinates or semantic foveal elements to screen pixels."""
    if coord is not None and len(coord) >= 2:
        norm_x = max(0.0, min(1.0, float(coord[0]) / 1280.0))
        norm_y = max(0.0, min(1.0, float(coord[1]) / 800.0))
        sx, sy = eye.geometry.to_screen(norm_x, norm_y)
        return (sx, sy), (norm_x, norm_y), f"vla_coord at ({norm_x:.2f}, {norm_y:.2f})"

    if elements:
        target_elem = resolve_semantic_target(target_desc, elements) or elements[0]
        sx, sy = eye.geometry.to_screen(target_elem.center_norm[0], target_elem.center_norm[1])
        return (sx, sy), target_elem.center_norm, target_elem.describe()

    return None


def run_agent(goal: str, max_turns: int = 10, dry_run: bool = True, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
    print("\n[AVCS Agent] Initializing Autonomous Live Agent...")
    print(f"[Objective]: {goal}")
    if dry_run:
        print("[Safety Mode]: DRY-RUN active (Physical actuation bypassed). Corner breakout active.\n")
    else:
        print("[Safety Mode]: LIVE ACTUATION active! Move cursor to any screen corner to abort.\n")

    eye = EyeStream()
    fovea = FoveaReader()
    narrator = LiveNarrator()
    motor = HandMotor()
    brain = LLMBridge(api_key=api_key, base_url=base_url)

    # Prefrontal Cognitive Mission Operators
    planner = CognitivePlanner()
    workflow = WorkflowCoordinator()
    recovery_mgr = ObstacleRecoveryManager()

    plan_graph = planner.decompose_goal(goal)
    print(f"[Cognitive Planner] Goal decomposed into {len(plan_graph.steps)} steps:")
    print(plan_graph.to_summary_prompt())
    print()

    print(f"[AVCS Eye] Weights: {eye.weights_status}")
    if sys.platform != "win32":
        print("[Hand Motor] Non-Windows host: motor actuation is disabled (read-only perception).")

    last_action = None
    last_ripple = None

    for turn in range(1, max_turns + 1):
        print(f"--- [Turn {turn}/{max_turns}] ---")

        # Check if plan is already completed
        if planner.state.status == "completed" or plan_graph.is_complete():
            print("[AVCS Agent] All mission steps successfully completed!")
            break

        current_step = planner.get_next_step()
        step_desc = current_step.description if current_step else goal
        if current_step:
            print(f"[Active Plan Step {current_step.id}]: {current_step.description}")

        # 1. Retina Observation (<6ms)
        eye_frame = eye.step()
        active_window = ""

        # 2. Foveal Inspection of Interactive Elements
        elements = fovea.inspect(eye_frame.raw_frame, eye_frame.hotspots, max_inspect=5)
        fovea_patches = [e.patch for e in elements if e.patch is not None]

        # 3. Dynamic Obstacle Detection
        obstacle = recovery_mgr.detect_obstacle(
            surprise_score=eye_frame.surprise_score,
            has_popup=eye_frame.has_popup,
            ripple_verified=last_ripple.verified if last_ripple else None,
            is_stable=eye_frame.is_stable,
        )
        if obstacle:
            print(f"[Obstacle Alert]: {obstacle.description} (type={obstacle.obstacle_type.value}, surprise={obstacle.surprise_magnitude:.2f})")
            rec_directive = recovery_mgr.generate_recovery(obstacle)
            print(f"[Recovery Directive]: Action='{rec_directive.action}', Strategy='{rec_directive.strategy}', Reason='{rec_directive.reason}'")
            if rec_directive.action == "abort":
                print(f"[AVCS Agent] Aborting execution due to unresolvable obstacle: {rec_directive.reason}")
                planner.state.status = "failed"
                break
            planner.reformulate_plan(obstacle.description)

        # 4. Live Commentary (AVCS speaks to LLM)
        commentary = narrator.narrate(
            eye_frame=eye_frame,
            active_window="",
            foveal_elements=elements,
            ripple_verified=last_ripple.verified if last_ripple else None,
            prior_action=last_action,
        )

        if commentary:
            print(commentary.to_llm_prompt())
        else:
            print("[AVCS Live Eye]: Screen static, holding state.")

        # 5. Local VLA Directive (Single-Brain Local VLA Engine)
        if commentary:
            directive = brain.plan_next_step(
                step_desc,
                commentary,
                fovea_patches=fovea_patches,
                frame=eye_frame.raw_frame,
            )
            print(
                f"[Local VLA Order]: Action='{directive.action}', Target='{directive.target}', "
                f"Reason='{directive.reason}' (planner={directive.source})"
            )
            if directive.source not in ("local_vla", "llm"):
                print("[LLMBridge] No local VLA weights loaded; using offline heuristic planner.")

            act = directive.action.lower()

            if act in ("finish", "stop", "abort"):
                print(f"[AVCS Agent] Task {act}: {directive.reason}")
                if act != "abort" and current_step:
                    planner.mark_step_complete(current_step.id, directive.reason)
                elif act == "abort":
                    planner.state.status = "failed"
                break

            if act == "wait":
                print("[AVCS Agent] Waiting for screen content to settle...")
                time.sleep(1.0)
                if current_step and current_step.action_type == "wait":
                    planner.mark_step_complete(current_step.id, "wait completed")
                continue

            if act == "switch_app":
                target_role = directive.target or (current_step.app_role if current_step else "editor")
                switch_cmd = workflow.switch_window(target_role)
                print(f"[Workflow Coordinator] Switching application context to '{target_role}': {switch_cmd}")
                if current_step:
                    planner.mark_step_complete(current_step.id, f"switched to {target_role}")

            elif act in ("click", "left_click", "right_click", "double_click", "triple_click", "mouse_move"):
                resolved = _resolve_screen_point(eye, elements, directive.coordinate, directive.target)
                if resolved is not None:
                    (screen_x, screen_y), norm_pt, desc = resolved
                    print(f"[Hand Motor] {act} at ({screen_x}, {screen_y}) targeting {desc}...")

                    if not dry_run:
                        pre_frame = eye_frame.raw_frame
                        try:
                            if act in ("click", "left_click"):
                                success = motor.click(screen_x, screen_y)
                            elif act == "right_click":
                                success = motor.right_click(screen_x, screen_y)
                            elif act == "double_click":
                                success = motor.double_click(screen_x, screen_y)
                            elif act == "triple_click":
                                success = motor.triple_click(screen_x, screen_y)
                            else:
                                success = motor.move_to(screen_x, screen_y)
                        except RuntimeError as exc:
                            print(f"[SAFETY] Motor movement halted by fail-safe: {exc}")
                            break
                        if success:
                            if act == "mouse_move":
                                last_action = f"moved to {desc}"
                                if current_step:
                                    planner.mark_step_complete(current_step.id, "mouse_move completed")
                            else:
                                last_ripple = motor.verify_ripple(eye.grab_screen, norm_pt, pre_frame)
                                last_action = f"{act} on {desc}"
                                print(f"[Ripple Verifier]: Verified={last_ripple.verified}, Energy={last_ripple.surprise_magnitude}")
                                if last_ripple.verified:
                                    recovery_mgr.record_outcome("unresponsive_target", True)
                                    if current_step:
                                        planner.mark_step_complete(current_step.id, f"{act} registered")
                                else:
                                    recovery_mgr.record_outcome("unresponsive_target", False)
                        else:
                            print("[WARNING] Motor movement halted by corner breakout failsafe.")
                            break
                    else:
                        print(f"[DRY RUN] Physical {act} bypassed (pass --live to authorize physical mouse actions).")
                        if current_step:
                            planner.mark_step_complete(current_step.id, f"dry run {act}")

            elif act in ("left_click_drag", "drag", "drag_start"):
                start_res = _resolve_screen_point(
                    eye, elements, directive.start_coordinate or directive.coordinate, directive.target
                )
                end_res = _resolve_screen_point(eye, elements, directive.end_coordinate, directive.target)
                if start_res is not None and end_res is not None:
                    (sx, sy), _, start_desc = start_res
                    (ex, ey), end_norm, end_desc = end_res
                    print(f"[Hand Motor] Dragging from ({sx}, {sy}) [{start_desc}] to ({ex}, {ey}) [{end_desc}]...")
                    if not dry_run:
                        pre_frame = eye_frame.raw_frame
                        try:
                            success = motor.drag(sx, sy, ex, ey)
                        except RuntimeError as exc:
                            print(f"[SAFETY] Drag halted by fail-safe: {exc}")
                            break
                        if success:
                            last_ripple = motor.verify_ripple(eye.grab_screen, end_norm, pre_frame)
                            last_action = f"dragged from ({sx},{sy}) to ({ex},{ey})"
                            if current_step:
                                planner.mark_step_complete(current_step.id, "drag completed")
                        else:
                            print("[WARNING] Drag halted by corner breakout failsafe.")
                            break
                    else:
                        print("[DRY RUN] Physical drag bypassed (pass --live to authorize physical drag).")
                        if current_step:
                            planner.mark_step_complete(current_step.id, "dry run drag")

            elif act == "scroll":
                delta = -3
                if directive.scroll_delta:
                    dx, dy = directive.scroll_delta
                    delta = dy if dy != 0 else dx
                resolved = _resolve_screen_point(eye, elements, directive.coordinate, directive.target) if directive.coordinate else None
                sx, sy = resolved[0] if resolved is not None else (None, None)
                print(f"[Hand Motor] Scrolling wheel delta={delta}" + (f" at ({sx}, {sy})" if sx is not None else ""))
                if not dry_run:
                    try:
                        success = motor.scroll(delta, x=sx, y=sy)
                        if success:
                            last_action = f"scrolled delta={delta}"
                    except RuntimeError as exc:
                        print(f"[SAFETY] Scroll halted by fail-safe: {exc}")
                        break
                else:
                    print("[DRY RUN] Physical scroll bypassed (pass --live to authorize scroll).")
                if current_step:
                    planner.mark_step_complete(current_step.id, f"scrolled delta={delta}")

            elif act in ("hotkey", "key"):
                keys = directive.keys or ([directive.target] if directive.target else ([directive.text] if directive.text else []))
                if keys:
                    print(f"[Hand Motor] Sending key/hotkey: {keys}")
                    if not dry_run:
                        try:
                            if len(keys) == 1:
                                motor.key(keys[0])
                            else:
                                motor.hotkey(*keys)
                            last_action = f"hotkey {'+'.join(keys)}"
                        except RuntimeError as exc:
                            print(f"[SAFETY] Key injection halted by fail-safe: {exc}")
                            break
                    else:
                        print("[DRY RUN] Physical hotkey bypassed (pass --live to authorize real keystrokes).")
                    if current_step:
                        planner.mark_step_complete(current_step.id, f"hotkey {'+'.join(keys)}")

            elif act in ("type", "type_text") and directive.text:
                print(f"[Hand Motor] Typing text: '{directive.text}'")
                if not dry_run:
                    try:
                        motor.type_text(directive.text)
                        last_action = f"typed '{directive.text}'"
                    except RuntimeError as exc:
                        print(f"[SAFETY] Keyboard typing halted by fail-safe: {exc}")
                        break
                else:
                    print("[DRY RUN] Physical keyboard typing bypassed (pass --live to authorize real keystrokes).")
                if current_step:
                    planner.mark_step_complete(current_step.id, f"typed '{directive.text}'")

        time.sleep(0.5)

    eye.close()


def run_monitor(seconds: int = 10, api_key: Optional[str] = None) -> None:
    print(f"\n[AVCS Monitor] Streaming live screen commentary for {seconds} seconds...")
    eye = EyeStream()
    fovea = FoveaReader()
    narrator = LiveNarrator()
    motor = HandMotor()

    print(f"[AVCS Eye] Weights: {eye.weights_status}")

    t_end = time.time() + seconds
    while time.time() < t_end:
        eye_frame = eye.step()
        elements = fovea.inspect(eye_frame.raw_frame, eye_frame.hotspots, max_inspect=4)
        comm = narrator.narrate(eye_frame, "", elements)

        if comm:
            print(f"[{time.strftime('%H:%M:%S')}] {comm.narrative} | Latency: {eye_frame.latency_ms}ms")
        time.sleep(0.8)

    eye.close()


def run_check(api_key: Optional[str] = None) -> int:
    print("[AVCS Diagnostic Health Check]")
    print("Version: 0.2.0")
    print(f"Host OS: {sys.platform}")

    # 1. System 1 EyeStream & Retina
    weights_loaded = False
    try:
        eye = EyeStream()
        frame = eye.step()
        print(f"[*] System 1 Retina: OK ({frame.raw_frame.shape[1]}x{frame.raw_frame.shape[0]} px, latency={frame.latency_ms:.2f}ms)")
        weights_loaded = bool(eye.weights_loaded)
        if weights_loaded:
            print(f"[*] Model Weights: OK ({eye.weights_status})")
        else:
            print(f"[!] Model Weights: NOT LOADED ({eye.weights_status})")
        eye.close()
    except Exception as e:
        print(f"[!] System 1 Retina: FAILED ({e})")
        return 1

    # 2. Fovea Reader
    try:
        reader = FoveaReader()
        print(f"[*] Fovea Reader (Pure-Pixel): OK (crop_size={reader.crop_size}px)")
    except Exception as e:
        print(f"[!] Fovea Reader: FAILED ({e})")
        return 1

    # 3. Hand Motor & Win32
    try:
        motor = HandMotor()
        screen_w, screen_h = motor.get_screen_size()
        print(f"[*] Hand Motor & Desktop Geometry: OK (Screen: {screen_w}x{screen_h}, Pure-Pixel Mode)")
    except Exception as e:
        print(f"[!] Hand Motor: FAILED ({e})")
        return 1

    # 4. Cognitive Planner
    try:
        planner = CognitivePlanner()
        plan = planner.decompose_goal("Inspect test report")
        print(f"[*] Cognitive Planner: OK (Generated {len(plan.steps)} plan steps)")
    except Exception as e:
        print(f"[!] Cognitive Planner: FAILED ({e})")
        return 1

    # 5. Local VLA Bridge
    try:
        bridge = LLMBridge()
        print(f"[*] Local VLA Bridge: OK (Endpoint: {bridge.base_url}, Model: {bridge.model_name}, Mode: 100% offline local VLA)")
    except Exception as e:
        print(f"[!] Local VLA Bridge: FAILED ({e})")
        return 1

    if not weights_loaded:
        print(
            "\n[CHECKS INCOMPLETE] Plumbing is wired, but no trained checkpoint was "
            "loaded, so perception is untrained. This is not a passing perception "
            "check. See 'Model Weights' above."
        )
        return 2

    print("\n[ALL CHECKS PASSED] AVCS is operational with a loaded checkpoint.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="AVCS Autonomous Live Agent Runner")
    parser.add_argument("--version", action="version", version="AVCS-SDK v0.2.0 (Parth Labs)")
    parser.add_argument("--check", action="store_true", help="Run system diagnostic health check")
    parser.add_argument("--goal", type=str, default="Inspect screen and summarize active window state", help="Goal for the agent")
    parser.add_argument("--monitor", action="store_true", help="Run live commentary monitor")
    parser.add_argument("--seconds", type=int, default=8, help="Duration for monitor mode")
    parser.add_argument("--dry-run", action="store_true", help="Bypass physical mouse clicks and keyboard typing (simulation mode)")
    parser.add_argument("--live", action="store_true", help="Authorize live physical Win32 mouse clicks and keyboard typing (default: active)")
    parser.add_argument("--api-key", type=str, default=None, help="Authentication key for encrypted .avcs model vault")
    parser.add_argument("--base-url", type=str, default=None, help="Custom LLM API Base URL")
    args = parser.parse_args()

    if args.api_key:
        os.environ["AVCS_API_KEY"] = args.api_key

    if args.check:
        sys.exit(run_check(api_key=args.api_key))
    elif args.monitor:
        run_monitor(args.seconds, api_key=args.api_key)
    else:
        is_dry_run = bool(args.dry_run and not args.live)
        run_agent(args.goal, dry_run=is_dry_run, api_key=args.api_key, base_url=args.base_url)


if __name__ == "__main__":
    main()

