"""[6] run_live_agent.py: End-to-End Autonomous AVCS Agent Runner.

Single-command entrypoint to launch the full AVCS pipeline:
1. EyeStream (100 FPS Live Retina + ConvGRU Surprise Detector)
2. FoveaReader (High-acuity central pixel zoom; pixel-only, no OCR)
3. LiveNarrator (Live English Commentary for LLM)
4. HandMotor (Biological Minimum-Jerk Mouse Servoing + Ripple Verifier)
5. LLMBridge (Continuous 2-way conversation stream with System 2)
6. CognitivePlanner (Hierarchical Goal Decomposition & Multi-Step Reasoning)
7. WorkflowCoordinator (Cross-Application Context & Window Switching)
8. ObstacleRecoveryManager (Dynamic Recovery on Modals, Locks & Syntax Errors)

Usage:
    python run_live_agent.py --goal "Check screen and describe active window"
    python run_live_agent.py --monitor --seconds 10
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

# Ensure safe UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .eye_stream import EyeStream
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

        # 2. Text / code / syntax matching
        if any(w in q_lower for w in ("code", "syntax", "text", "line", "type", "script")):
            if elem.category == "text_like":
                score += 3.0
            score += elem.glyph_likeness * 2.0
            score += elem.edge_density * 1.5

        # 3. Interactive controls / buttons / dialogs
        if any(w in q_lower for w in ("button", "dismiss", "ok", "cancel", "close", "dialog", "click", "control")):
            if elem.category in ("graphic", "interactive_control"):
                score += 3.0
            score += elem.contrast * 1.5

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


def run_agent(goal: str, max_turns: int = 10, dry_run: bool = False) -> None:
    print("\n[AVCS Agent] Initializing Autonomous Live Agent...")
    print(f"[Objective]: {goal}")
    print("[Safety Mode]: Corner breakout active (Move cursor to top-left corner to abort)\n")

    eye = EyeStream()
    fovea = FoveaReader()
    narrator = LiveNarrator()
    motor = HandMotor()
    brain = LLMBridge()

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
        active_window = motor.get_foreground_title()
        if active_window:
            workflow.register_window(active_window)
            workflow.set_focused_context(active_window)

        # 2. Foveal Inspection of Interactive Elements
        elements = fovea.inspect(eye_frame.raw_frame, eye_frame.hotspots, max_inspect=5)
        fovea_patches = [e.patch for e in elements] if elements else None

        # 3. Dynamic Obstacle Detection
        obstacle = recovery_mgr.detect_obstacle(
            surprise_score=eye_frame.surprise_score,
            has_popup=eye_frame.has_popup,
            ripple_verified=last_ripple.verified if last_ripple else None,
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
            active_window=active_window,
            foveal_elements=elements,
            ripple_verified=last_ripple.verified if last_ripple else None,
            prior_action=last_action,
        )

        if commentary:
            print(commentary.to_llm_prompt())
        else:
            print("[AVCS Live Eye]: Screen static, holding state.")

        # 5. LLM Cognitive Directive (LLM orders next action)
        if commentary:
            directive = brain.plan_next_step(step_desc, commentary, fovea_patches=fovea_patches)
            print(
                f"[LLM Order]: Action='{directive.action}', Target='{directive.target}', "
                f"Reason='{directive.reason}' (planner={directive.source})"
            )
            if directive.source != "llm":
                print("[LLMBridge] No model reachable; running the offline heuristic planner.")

            if directive.action == "finish":
                print(f"[AVCS Agent] Task complete: {directive.reason}")
                if current_step:
                    planner.mark_step_complete(current_step.id, directive.reason)
                break

            if directive.action == "wait":
                print("[AVCS Agent] Waiting for screen content to settle...")
                time.sleep(1.0)
                if current_step and current_step.action_type == "wait":
                    planner.mark_step_complete(current_step.id, "wait completed")
                continue

            if directive.action == "switch_app":
                target_role = directive.target or (current_step.app_role if current_step else "editor")
                switch_cmd = workflow.switch_window(target_role)
                print(f"[Workflow Coordinator] Switching application context to '{target_role}': {switch_cmd}")
                if current_step:
                    planner.mark_step_complete(current_step.id, f"switched to {target_role}")

            elif directive.action == "click" and elements:
                # Semantic Target Matching
                target_elem = resolve_semantic_target(directive.target, elements)
                if target_elem is None:
                    target_elem = elements[0]

                screen_x, screen_y = eye.geometry.to_screen(target_elem.center_norm[0], target_elem.center_norm[1])
                print(f"[Hand Motor] Moving mouse to ({screen_x}, {screen_y}) semantically targeting {target_elem.describe()}...")

                if not dry_run:
                    pre_frame = eye_frame.raw_frame
                    success = motor.click(screen_x, screen_y)
                    if success:
                        last_ripple = motor.verify_ripple(eye.grab_screen, target_elem.center_norm, pre_frame)
                        last_action = f"click on {target_elem.category} at ({target_elem.center_norm[0]:.2f},{target_elem.center_norm[1]:.2f})"
                        print(f"[Ripple Verifier]: Verified={last_ripple.verified}, Energy={last_ripple.surprise_magnitude}")
                        if last_ripple.verified:
                            recovery_mgr.record_outcome("unresponsive_target", True)
                            if current_step:
                                planner.mark_step_complete(current_step.id, "click registered")
                        else:
                            recovery_mgr.record_outcome("unresponsive_target", False)
                    else:
                        print("[WARNING] Motor movement halted by corner breakout failsafe.")
                        break
                else:
                    print("[DRY RUN] Physical click bypassed.")
                    if current_step:
                        planner.mark_step_complete(current_step.id, "dry run click")

            elif directive.action == "type" and directive.text:
                print(f"[Hand Motor] Typing text: '{directive.text}'")
                if not dry_run:
                    motor.type_text(directive.text)
                    last_action = f"typed '{directive.text}'"
                else:
                    print("[DRY RUN] Physical keyboard typing bypassed (remove --dry-run to actuate real keyboard).")
                if current_step:
                    planner.mark_step_complete(current_step.id, f"typed '{directive.text}'")

        time.sleep(0.5)

    eye.close()


def run_monitor(seconds: int = 10) -> None:
    print(f"\n[AVCS Monitor] Streaming live screen commentary for {seconds} seconds...")
    eye = EyeStream()
    fovea = FoveaReader()
    narrator = LiveNarrator()
    motor = HandMotor()

    print(f"[AVCS Eye] Weights: {eye.weights_status}")

    t_end = time.time() + seconds
    while time.time() < t_end:
        eye_frame = eye.step()
        active_win = motor.get_foreground_title()
        elements = fovea.inspect(eye_frame.raw_frame, eye_frame.hotspots, max_inspect=4)
        comm = narrator.narrate(eye_frame, active_win, elements)

        if comm:
            print(f"[{time.strftime('%H:%M:%S')}] {comm.narrative} (Window: '{active_win}') | Latency: {eye_frame.latency_ms}ms")
        time.sleep(0.8)

    eye.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="AVCS Autonomous Live Agent Runner")
    parser.add_argument("--goal", type=str, default="Inspect screen and summarize active window state", help="Goal for the agent")
    parser.add_argument("--monitor", action="store_true", help="Run live commentary monitor")
    parser.add_argument("--seconds", type=int, default=8, help="Duration for monitor mode")
    parser.add_argument("--dry-run", action="store_true", help="Bypass physical mouse clicks")
    args = parser.parse_args()

    if args.monitor:
        run_monitor(args.seconds)
    else:
        run_agent(args.goal, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

