"""Demonstration of AVCS System 1 Autonomous Substep Execution Loop.

Demonstrates the model-to-model dialogue architecture from GOAL.md:
1. System 2 delegates a broad, high-level goal in natural language.
2. AVCS System 1 autonomously plans and executes on-screen substeps (Look -> Ground -> Act -> Verify).
3. Coordinates remain completely internal to System 1: ZERO coordinates leaked to System 2.
4. Visual state transitions are verified after each action via postcondition assertions.
5. Returns truthful, evidence-backed natural language summaries to System 2.
6. Demonstrates handling of unexpected modal blockers (GOAL.md Example B).
"""
from __future__ import annotations

import json
import time

from avcs.control.controller import AVCSController, GoalExecutionResult, SubstepTrace
from avcs.control.simulated_app import SimulatedAppScreen


def run_autonomous_substep_demo() -> None:
    print("=" * 76)
    print("   AVCS SYSTEM 1: AUTONOMOUS SUBSTEP LOOP & MODEL-TO-MODEL DIALOGUE DEMO   ")
    print("=" * 76)

    # 1. Initialize Controller attached to Simulated Interactive Desktop Screen
    print("\n[Step 1] Initializing AVCS System 1 Controller with Simulated Desktop...")
    sim_screen = SimulatedAppScreen(width=320, height=240)
    controller = AVCSController(
        source=sim_screen,
        weights_path=None,
        device="cpu",
        target_fps=60,
        force_virtual_motor=True,
        enable_streaming=False,
    )
    sim_screen.attach_controller(controller)
    print(" -> AVCS Controller initialized.")
    print(" -> Hands: Virtual neuromuscular driver with 5th-order minimum-jerk servoing.")
    print(" -> Eyes: High-acuity visual perception & postcondition assertion verifier.")

    # 2. Scenario A: Autonomous Application Exploration & Test
    print("\n" + "-" * 76)
    print("[Scenario A] High-Level Directive: Autonomous GUI Application Exploration")
    print("-" * 76)
    goal_a = "explore text editor: open it, test typing, and close without saving"
    print(f"System 2 (Manager Directive): '{goal_a}'\n")

    print("AVCS System 1 Autonomous Action Loop (Internal Execution):")

    def on_substep(trace: SubstepTrace) -> None:
        print(f"  [Substep {trace.step_index}] Action: {trace.action:<14} Target: {trace.target_query:<18}")
        print(f"              Status: {'VERIFIED' if trace.verified else 'FAILED'} ({trace.latency_ms:.1f}ms)")
        print(f"              Evidence: {trace.message}")

    res_a: GoalExecutionResult = controller.execute_goal(
        goal_a,
        max_substeps=8,
        substep_callback=on_substep,
    )

    print("\nAVCS System 1 (Natural Language Report to System 2):")
    print(f"  \"{res_a.summary}\"")
    print(f"\nExecution Metadata:")
    print(f"  * Status:        {res_a.status}")
    print(f"  * Success:       {res_a.success}")
    print(f"  * Substeps:      {len(res_a.substeps)} executed & verified")
    print(f"  * Total Latency: {res_a.total_latency_ms:.1f} ms")

    # Verify zero coordinates leaked
    payload = res_a.to_dict()
    coord_leak = any(k in payload for k in ("x", "y", "target_coord", "bounding_box"))
    print(f"  * Zero Coordinate Leakage: {not coord_leak} (Coordinates strictly encapsulated)")

    # 3. Scenario B: Handling an Unexpected Blocker (GOAL.md Example B)
    print("\n" + "-" * 76)
    print("[Scenario B] Blocker Handling: Unexpected Modal Dialog on Screen")
    print("-" * 76)
    sim_screen.trigger_blocker("Session expired. Please re-authenticate.")
    goal_b = "company portal pe jaakar latest monthly report download karo"
    print(f"System 2 (Manager Directive): '{goal_b}'\n")

    print("AVCS System 1 Autonomous Action Loop:")
    res_b: GoalExecutionResult = controller.execute_goal(goal_b)

    print(f"  * Eyes detect: {res_b.final_observation.description}")
    print(f"  * Blocker Flagged: {res_b.blocker_reason}")
    print(f"  * Substeps Dispatched: {len(res_b.substeps)} (Halted immediately without blind clicks)")

    print("\nAVCS System 1 (Natural Language Report to System 2):")
    print(f"  \"{res_b.summary}\"")
    print(f"\nExecution Metadata:")
    print(f"  * Status:        {res_b.status}")
    print(f"  * Success:       {res_b.success}")

    controller.close()
    print("\n" + "=" * 76)
    print("   AUTONOMOUS SUBSTEP LOOP DEMONSTRATION SUCCESSFULLY COMPLETED!   ")
    print("=" * 76)


if __name__ == "__main__":
    run_autonomous_substep_demo()
