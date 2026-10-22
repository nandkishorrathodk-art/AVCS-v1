"""End-to-End System 2 (LLM) + System 1 (AVCS Eyes & Hands) Closed-Loop Demo.

Demonstrates the fail-closed wiring of the System 1 <-> System 2 boundary:
1. System 2 Intent Interface (Zero Coordinates Leaked to LLM)
2. Pure-Pixel Evidence-Backed Scene Understanding (Zero Hallucination)
3. Neural Grounding via GoalConditionedGrounder
4. Neuromuscular 5th-Order Minimum-Jerk Actuation & Biological Jitter
5. Fresh Post-Action Verification via VisualRippleVerifier
6. Real-Time Sensory-Motor Telemetry Stream
7. Fail-Closed Keyboard Focus Verification

Note: with no trained grounding checkpoint present the grounder abstains, so
every natural-language action below is expected to be refused. That refusal is
the correct behaviour, not a demo failure. The demo exits non-zero when any
step reports an unexpected outcome.
"""
from __future__ import annotations

import json
import time

from avcs.controller import AVCSController, ActionResult, SceneObservation
from forge.vision.capture import SyntheticScreen


def run_avcs_system2_demo() -> None:
    print("=" * 70)
    print("   AVCS SYSTEM 1 (EYES + HANDS) + SYSTEM 2 (LLM INTENT) CLOSED-LOOP DEMO   ")
    print("=" * 70)

    # 1. Initialize Central Orchestration Controller
    print("\n[Step 1] Initializing Central AVCSController on CPU...")
    source = SyntheticScreen(width=320, height=240)
    from pathlib import Path
    weights_to_use = (
        "runs/weights/goal_grounder_combined_best.pt"
        if Path("runs/weights/goal_grounder_combined_best.pt").is_file()
        else "runs/weights/goal_grounder_best.pt"
    )
    controller = AVCSController(
        source=source,
        weights_path=weights_to_use,
        device="cpu",
        target_fps=60,
        force_virtual_motor=True,
        enable_streaming=True,
    )
    print(f" -> Eye & Motor Subsystems Online: {controller.is_grounder_trained=}")
    print(f" -> Telemetry Streaming Bus: {controller.stream_bus is not None} (60 FPS)")

    # 2. Attach Telemetry Observer
    telemetry_packets = []
    unsub = controller.stream_bus.subscribe(lambda pkt: telemetry_packets.append(pkt))

    # 3. Step 1: System 2 asks "What is on the screen?"
    print("\n[Step 2] System 2 Intent: describe_screen()")
    print(" -> LLM asks: 'What is currently on screen?'")
    obs: SceneObservation = controller.describe_screen()
    print(" -> AVCS Observation returned to LLM (Zero coordinates leaked):")
    print(f"    * Context:     {obs.primary_category} ({obs.stability})")
    print(f"    * Description: {obs.description}")
    print(f"    * Confidence:  {obs.confidence:.2f} (Uncertainty: {obs.uncertainty:.2f})")
    print(f"    * Surprise:    {obs.surprise_score:.3f}")

    # 4. Step 2: System 2 asks "Open VLC Player"
    print("\n[Step 3] System 2 Intent: open_application('VLC Player')")
    print(" -> LLM sends high-level intent without any coordinate management.")
    act1: ActionResult = controller.open_application("VLC Player")
    print(" -> AVCS Action Execution Outcome:")
    print(f"    * Action:    {act1.action}")
    print(f"    * Target:    {act1.target_query}")
    print(f"    * Success:   {act1.success}")
    print(f"    * Verified:  {act1.verified}")
    print(f"    * Message:   {act1.message}")
    print(f"    * Latency:   {act1.latency_ms:.2f} ms")

    # 5. Step 3: System 2 asks "Click Play button"
    print("\n[Step 4] System 2 Intent: click_target('Play button')")
    act2: ActionResult = controller.click_target("Play button", verify_action=True)
    print(" -> AVCS Action Execution Outcome:")
    print(f"    * Action:    {act2.action}")
    print(f"    * Target:    {act2.target_query}")
    print(f"    * Success:   {act2.success}")
    print(f"    * Verified:  {act2.verified}")
    print(f"    * Message:   {act2.message}")
    print(f"    * Latency:   {act2.latency_ms:.2f} ms")

    # 6. Step 4: System 2 asks "Type search query"
    print("\n[Step 5] System 2 Intent: type_text('Inception.mp4', target_field='Search input')")
    act3: ActionResult = controller.type_text("Inception.mp4", target_query="Search input")
    print(" -> AVCS Fail-Closed Safety Outcome:")
    print(f"    * Action:    {act3.action}")
    print(f"    * Success:   {act3.success} (Strictly rejected if focus unverified)")
    print(f"    * Message:   {act3.message}")

    # 7. Step 5: Verify 60 FPS Telemetry Bus
    print("\n[Step 6] Verifying Live 60 FPS NeuroStream Telemetry Stream...")
    print(f" -> Captured {len(telemetry_packets)} high-frequency telemetry packets during interaction.")
    if telemetry_packets:
        last_pkt = telemetry_packets[-1]
        print(" -> Latest Packet Sample:")
        print(f"    * Seq:             {last_pkt.seq}")
        print(f"    * Gaze Fixation:   {last_pkt.fixation}")
        print(f"    * Primary Context: {last_pkt.primary_category}")
        print(f"    * Inference Time:  {last_pkt.inference_ms:.2f} ms")

    unsub()
    if hasattr(source, "close"):
        source.close()

    # A refusal is the expected outcome when no trained grounder is present.
    # The demo is only "successful" if the wiring behaved as declared: abstain
    # without a trained checkpoint, and never claim a verified action.
    refused = (not act1.success) and (not act2.success) and (not act3.success)
    print("\n" + "=" * 70)
    if refused:
        print("   CLOSED-LOOP WIRING VERIFIED: all actions fail-closed (no trained grounder).   ")
    else:
        print("   UNEXPECTED OUTCOME: an action reported success without a trained grounder.   ")
    print("=" * 70)
    if not refused:
        raise SystemExit(1)


main = run_avcs_system2_demo


if __name__ == "__main__":
    main()

