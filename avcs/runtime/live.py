"""AVCS Real-Time Desktop Stream: Integrated Biological Eyes + Hands Engine.

Provides an end-to-end, real-time streaming environment combining:
1. Biological Eyes: Continuous 30-60 FPS visual perception, retinal foveation,
   optical motion tracking, and dorsal affordance localization.
2. Neuromuscular Hands: Closed-loop minimum-jerk physical cursor steering,
   Fitts' Law velocity profiles, and optical ripple post-click verification.
3. Apprenticeship Mining: Discovers causal mouse clicks and UI state transitions
   from tutorial screencasts and video demonstrations.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any, Optional, Tuple

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import avcs
from avcs.gui import launch_gui
from avcs.control.servoing import NeuromuscularServoingController
from avcs.control.verifier import VisualRippleVerifier
from forge.vision.cortex.motor import get_default_driver


class CausalActionMiner:
    """Discovers causal user clicks and UI state transitions from demonstration streams.

    Principles:
    1. Cursor Dwell: When a user clicks, their cursor slows down and dwells at
       target coordinates (x, y) for a sustained period (typically 150ms - 500ms).
    2. Click Event: Following dwell, if the screen undergoes an optical ripple, visual delta,
       or scene stability transition (e.g., menu popped up, window opened, dialog appeared),
       a causal (State_before, Click_at, State_after) transition is identified.
    3. Self-Supervised Apprenticeship: Extracted clicks generate positive training signal
       for affordance detection and action forecasting without requiring manual labels.
    """

    def __init__(
        self,
        dwell_radius: float = 0.045,
        min_dwell_ticks: int = 4,
        max_dwell_ticks: int = 45,
        output_file: Optional[Path] = None,
    ) -> None:
        self.dwell_radius = dwell_radius
        self.min_dwell_ticks = min_dwell_ticks
        self.max_dwell_ticks = max_dwell_ticks
        self.output_file = output_file or Path("runs/apprenticeship_actions.jsonl")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.dwell_anchor: Optional[Tuple[float, float]] = None
        self.dwell_count: int = 0
        self.candidate_click: Optional[Tuple[float, float, int]] = None
        self.candidate_wait_ticks: int = 0
        self.cooldown_ticks: int = 0
        self.mined_count: int = 0

    def step(self, state: Any, raw_frame: Optional[Any] = None) -> Optional[dict]:
        gaze = state.gaze_fixation
        if self.cooldown_ticks > 0:
            self.cooldown_ticks -= 1

        # 1. Update Dwell Clustered Tracking
        if self.dwell_anchor is None:
            self.dwell_anchor = gaze
            self.dwell_count = 1
        else:
            dx = gaze[0] - self.dwell_anchor[0]
            dy = gaze[1] - self.dwell_anchor[1]
            dist = math.hypot(dx, dy)
            if dist <= self.dwell_radius:
                self.dwell_count += 1
            else:
                # Dwell ended: if dwell was sustained, register candidate click
                if self.min_dwell_ticks <= self.dwell_count <= self.max_dwell_ticks and self.cooldown_ticks == 0:
                    self.candidate_click = (self.dwell_anchor[0], self.dwell_anchor[1], self.dwell_count)
                    self.candidate_wait_ticks = 8
                self.dwell_anchor = gaze
                self.dwell_count = 1

        # 2. Check for Causal Visual Transition Trigger following candidate click
        if self.candidate_click is not None and self.candidate_wait_ticks > 0:
            self.candidate_wait_ticks -= 1
            cx, cy, dw_ticks = self.candidate_click

            has_transition = (
                state.stability in ("changed", "busy", "transition")
                or state.surprise_score > 0.10
                or state.motion_magnitude > 0.035
                or (hasattr(state, "visual_diff") and state.visual_diff and getattr(state.visual_diff, "appeared_count", 0) > 0)
            )

            if has_transition and self.cooldown_ticks == 0:
                self.mined_count += 1
                self.cooldown_ticks = 15  # Cooldown to avoid duplicate triggers from ongoing animations
                self.candidate_click = None
                self.candidate_wait_ticks = 0

                # Match nearest affordance if available
                nearest_kind = "clickable"
                nearest_conf = 0.50
                if state.affordance_hotspots:
                    for hx, hy, kind, conf in state.affordance_hotspots:
                        if math.hypot(cx - hx, cy - hy) < 0.08:
                            nearest_kind = kind
                            nearest_conf = conf
                            break

                summary = (
                    state.visual_diff.causal_summary
                    if hasattr(state, "visual_diff") and state.visual_diff and state.visual_diff.causal_summary
                    else f"UI state transition in {getattr(state, 'primary_category', 'desktop')}"
                )

                record = {
                    "action_id": self.mined_count,
                    "tick": state.frame_index,
                    "click_coords": [round(cx, 3), round(cy, 3)],
                    "dwell_ticks": dw_ticks,
                    "affordance_type": nearest_kind,
                    "affordance_confidence": round(nearest_conf, 2),
                    "stability": state.stability,
                    "surprise_score": round(state.surprise_score, 3),
                    "summary": summary,
                }

                try:
                    with open(self.output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                except Exception:
                    pass

                return record

        if self.candidate_wait_ticks <= 0:
            self.candidate_click = None

        return None


def run_eyes_and_hands_stream(
    seconds: float = 0.0,
    fps: int = 60,
    capture_mode: str = "desktop",
    auto_servo: bool = False,
    virtual_only: bool = False,
    video_path: Optional[str] = None,
    online_learning: bool = False,
    commentary: bool = False,
    fast: bool = False,
    loop_video: bool = False,
    stride: int = 1,
    device: Optional[str] = None,
) -> None:
    """Streams live desktop or video perception with continuous Eyes telemetry and Hand coordination."""
    print("=" * 78)
    print("    AVCS // REAL-TIME DESKTOP STREAM: BIOLOGICAL EYES + HANDS ENGINE    ")
    print("=" * 78)

    video_ended = False
    total_video_frames = 0
    stride = max(1, stride)
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if video_path:
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video stream or file: {video_path}")
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            v_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            dur_mins = (total_video_frames / v_fps) / 60.0 if total_video_frames > 0 else 0.0

            def grab_video_frame() -> np.ndarray:
                nonlocal video_ended
                if stride > 1:
                    for _ in range(stride - 1):
                        cap.grab()
                ret, frame = cap.read()
                if not ret:
                    if loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    else:
                        video_ended = True
                        return np.zeros((256, 256, 3), dtype=np.uint8)
                if not ret:
                    video_ended = True
                    return np.zeros((256, 256, 3), dtype=np.uint8)
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from forge.vision.capture import FrameSource
            source_to_use = FrameSource(grab_video_frame)
            mode_tag = "Fast Video Stream" if fast else f"Video Stream (Target: {fps} FPS)"
            print(f"[*] Initializing {mode_tag}: {video_path}...")
            print(f"[*] Video Scope: {total_video_frames:,} total frames (~{dur_mins:.1f} mins) | Device: {device.upper()} | Stride: {stride}x")
        except ImportError:
            raise ImportError("OpenCV (cv2) is required to stream video files. Install via: pip install opencv-python")
    else:
        source_to_use = capture_mode
        print(f"[*] Initializing Biological Vision Stream (Source: {capture_mode}, Target: {fps} FPS)...")

    client = avcs.eye(source=source_to_use, fps=fps, force_virtual_motor=virtual_only, device=device)
    state = client.perceive()
    print(f"[*] Human Eye Online. Initial Gaze Fixation: ({state.gaze_fixation[0]:.3f}, {state.gaze_fixation[1]:.3f})")
    print(f"[*] Motor Servoing Driver: {'Virtual Driver' if virtual_only else 'Native Win32 Driver'}")

    miner = CausalActionMiner()
    print(f"[*] Causal Action Miner: ACTIVE -> Recording to {miner.output_file}")

    temporal_model = None
    temporal_opt = None
    affordance_opt = None
    prev_frame_t = None
    pair_ring_buffer = collections.deque(maxlen=2)

    if online_learning:
        import torch
        from avcs.temporal_cortex import TemporalVisualCortex, temporal_loss_for_frames
        temporal_model = TemporalVisualCortex().to(client.device)
        temporal_model.train()
        temporal_opt = torch.optim.AdamW(temporal_model.parameters(), lr=1e-4)

        if hasattr(client.eye, "affordance_head"):
            client.eye.affordance_head.train()
            affordance_opt = torch.optim.AdamW(client.eye.affordance_head.parameters(), lr=1e-4)

        print("[*] Real-Time Online Apprenticeship Learning: ACTIVE (Temporal + Affordance Updates)")

    print("=" * 78)
    print("STREAMING LIVE DESKTOP FRAMES (Press Ctrl+C to stop)...")

    start_t = time.perf_counter()
    tick_count = 0
    last_servo_time = 0.0
    last_commentary_time = 0.0

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass

    try:
        while True:
            if video_path and video_ended:
                print("\n[*] Video demonstration stream completed. All frames processed!")
                break
            if seconds > 0 and (time.perf_counter() - start_t) >= seconds:
                break

            t0 = time.perf_counter()
            state = client.perceive()
            tick_count += 1

            moving_str = f"{len(state.moving_objects)} obj" if state.moving_objects else "static"
            aff_count = len(state.affordance_hotspots) if state.affordance_hotspots else 0
            raw_frame = getattr(client.eye, "_latest_raw_frame", None)

            # Step Causal Action Miner (detect tutorial tutor clicks)
            mined_action = miner.step(state, raw_frame)
            if mined_action:
                cx, cy = mined_action["click_coords"]
                lbl = mined_action["affordance_type"]
                summ = mined_action["summary"]
                sys.stdout.write(
                    f"\n  [ACTION MINED #{miner.mined_count}] [CLICK] Click @ ({cx:.2f}, {cy:.2f}) "
                    f"[{lbl}] -> {summ[:45]} (Dwell: {mined_action['dwell_ticks']} ticks)\n"
                )
                sys.stdout.flush()

                # Online reinforcement of affordance head on verified causal click target
                if online_learning and affordance_opt is not None and getattr(client.eye, "_last_feat", None) is not None:
                    try:
                        import torch
                        import torch.nn.functional as F
                        feat_t = client.eye._last_feat
                        logits = client.eye.affordance_head(feat_t)
                        gy = min(15, max(0, int(cy * 16.0)))
                        gx = min(15, max(0, int(cx * 16.0)))
                        target_t = torch.tensor([1], dtype=torch.long, device=client.device)
                        loss_aff = F.cross_entropy(logits[:, :, gy, gx], target_t)
                        loss_aff.backward()
                        affordance_opt.step()
                        affordance_opt.zero_grad()
                    except Exception:
                        pass

            # Online temporal self-supervised weight updates
            loss_str = ""
            if online_learning and temporal_model is not None:
                import torch
                import torch.nn.functional as F
                if raw_frame is not None and prev_frame_t is not None:
                    curr_t = torch.from_numpy(raw_frame).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(client.device)
                    if curr_t.shape[-2:] != (128, 128):
                        curr_t = F.interpolate(curr_t, size=(128, 128), mode="bilinear", align_corners=False)
                    pair_ring_buffer.append((prev_frame_t, curr_t))
                    prev_frame_t = curr_t.detach()
                    if len(pair_ring_buffer) >= 2:
                        batch_prev = torch.cat([p[0] for p in pair_ring_buffer], dim=0)
                        batch_curr = torch.cat([p[1] for p in pair_ring_buffer], dim=0)
                        loss_dict = temporal_loss_for_frames(temporal_model, batch_prev, batch_curr)
                        loss = loss_dict["total"]
                        loss.backward()
                        temporal_opt.step()
                        temporal_opt.zero_grad()
                        loss_str = f" | Loss: {loss.item():.4f}"
                elif raw_frame is not None:
                    prev_frame_t = torch.from_numpy(raw_frame).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(client.device)
                    if prev_frame_t.shape[-2:] != (128, 128):
                        prev_frame_t = F.interpolate(prev_frame_t, size=(128, 128), mode="bilinear", align_corners=False)

            # Real-time streaming status line
            mined_str = f" | Actions: {miner.mined_count}" if miner.mined_count > 0 else ""
            progress_str = ""
            if total_video_frames > 0:
                pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if 'cv2' in sys.modules else 0
                pct = min(100.0, (pos / total_video_frames) * 100.0) if total_video_frames > 0 else 0.0
                progress_str = f" | Video: {pct:4.1f}%"
            sys.stdout.write(
                f"\r[TICK #{state.frame_index:04d}] "
                f"FPS: {state.fps:4.1f} | "
                f"Latency: {state.latency_ms:4.1f}ms | "
                f"Gaze: ({state.gaze_fixation[0]:.2f}, {state.gaze_fixation[1]:.2f}) | "
                f"Affordances: {aff_count:2d} | "
                f"Motion: {state.motion_magnitude:.3f} ({moving_str}) | "
                f"Stability: {state.stability:<8}"
                f"{progress_str}"
                f"{loss_str}"
                f"{mined_str}"
            )
            sys.stdout.flush()

            # Live verbal commentary stream (talking while observing)
            if commentary and (time.perf_counter() - last_commentary_time) >= 4.0:
                last_commentary_time = time.perf_counter()
                try:
                    obs = state.scene_observation
                    if obs and obs.description:
                        sys.stdout.write(f"\n  [AI TALK] [EYE] {obs.primary_category.upper()}: {obs.description}\n")
                        sys.stdout.flush()
                except Exception:
                    pass

            # Auto-servo demonstration: smoothly steer hand to top affordance every 3 seconds
            if auto_servo and (time.perf_counter() - last_servo_time) >= 3.0:
                last_servo_time = time.perf_counter()
                if state.affordance_hotspots:
                    best = max(state.affordance_hotspots, key=lambda a: a[3])
                    target = (best[0], best[1])
                    sys.stdout.write(f"\n  [HAND] Neuromuscular servoing to '{best[2]}' at ({target[0]:.2f}, {target[1]:.2f})...\n")
                    res = client.move_to(target=target, realtime=True)
                    sys.stdout.write(f"  [HAND] Docked in {res.duration_ms:.0f}ms (Fitts ID: {res.fitts_index_of_difficulty:.2f} bits)\n")

            # Pacing: skip sleep if in fast compute mode
            if not fast:
                elapsed = time.perf_counter() - t0
                sleep_dt = max(0.0, (1.0 / fps) - elapsed)
                if sleep_dt > 0:
                    time.sleep(sleep_dt)

    except KeyboardInterrupt:
        print("\n\nStream paused by operator.")
    finally:
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass

    total_time = max(0.001, time.perf_counter() - start_t)
    effective_fps = tick_count / total_time
    usage = client.get_usage()
    receipt = client.generate_receipt()
    client.close()

    if online_learning and temporal_model is not None:
        save_path = Path("runs/weights/temporal_cortex_stream_learned.pt")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save(temporal_model.state_dict(), save_path)
        print(f"\n[*] Saved real-time stream learned weights to: {save_path}")

    print("\n" + "-" * 78)
    print("STREAM SESSION SUMMARY:")
    print(f"  * Runtime                 : {total_time:.2f} seconds")
    print(f"  * Total Processed Frames  : {tick_count} frames")
    print(f"  * Sustained Vision FPS    : {effective_fps:.1f} FPS")
    print(f"  * Median Latency          : {client.eye.p50_latency_ms:.2f} ms")
    print(f"  * Mined Tutor Actions     : {miner.mined_count} causal clicks discovered")
    print(f"  * Apprenticeship Actions  : {miner.output_file}")
    print(f"  * Cryptographic Receipt   : {receipt.receipt_id}")
    print("-" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AVCS Live Desktop Stream - Real-Time Biological Eyes + Hands Engine"
    )
    parser.add_argument("--gui", action="store_true", help="Launch native Desktop HUD workstation")
    parser.add_argument(
        "--allow-capture",
        "--no-exclude",
        dest="allow_capture",
        action="store_true",
        default=False,
        help="Allow desktop screen capture / screenshots of the GUI window (disables WDA_EXCLUDEFROMCAPTURE)",
    )
    parser.add_argument("--seconds", type=float, default=0.0, help="Stream duration in seconds (0 for full video or indefinite)")
    parser.add_argument("--fps", type=int, default=30, help="Perception stream frame rate (default: 30)")
    parser.add_argument("--auto-servo", action="store_true", help="Periodically steer hand to detected affordances")
    parser.add_argument("--virtual", action="store_true", help="Force virtual cursor motor driver")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic desktop frames instead of live monitor")
    parser.add_argument("--video", type=str, default=None, help="Stream from an MP4 screencast or video file instead of display")
    parser.add_argument("--learn", action="store_true", help="Enable online self-supervised stream learning in real time")
    parser.add_argument("--talk", "--commentary", dest="commentary", action="store_true", help="Live conversational commentary while watching video")
    parser.add_argument("--fast", action="store_true", help="Process video at maximum compute speed without artificial FPS delays")
    parser.add_argument("--loop", action="store_true", help="Loop video infinitely until --seconds duration is reached")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame (e.g. 2 or 3 to accelerate full-length video)")
    parser.add_argument("--device", type=str, default=None, help="Hardware device (cuda or cpu)")
    args = parser.parse_args()

    if args.gui:
        print("Launching AVCS Real-Time Desktop Cortex HUD...", flush=True)
        source = None
        if args.synthetic:
            from forge.vision.capture import SyntheticScreen

            source = SyntheticScreen()
        launch_gui(fps=args.fps, source=source, allow_capture=args.allow_capture)
    else:
        capture_mode = "synthetic" if args.synthetic else "desktop"
        run_eyes_and_hands_stream(
            seconds=args.seconds,
            fps=args.fps,
            capture_mode=capture_mode,
            auto_servo=args.auto_servo,
            virtual_only=args.virtual,
            video_path=args.video,
            online_learning=args.learn,
            commentary=args.commentary,
            fast=args.fast,
            loop_video=args.loop,
            stride=args.stride,
            device=args.device,
        )


if __name__ == "__main__":
    main()
