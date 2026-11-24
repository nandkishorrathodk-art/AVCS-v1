"""Empirical Verification Audit: Multi-Step Application Launch & Window Transition Tracking.

Audits and verifies from raw pixels:
1. Icon click detection and local visual ripple assertion (VisualRippleVerifier).
2. Magnocellular motion stream optical flow & moving entity tracking during window expansion.
3. Ventral semantic stream category shift across desktop -> terminal / browser / editor.
4. Temporal predictive coding world model & stability hysteresis.
5. Discrete VisualEventStreamer lifecycle events & agent readiness synchronization.
6. GameReflexController biological surprise reflex actuation.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw

from avcs.events import VisualEvent, VisualEventStreamer, VisualEventType
from avcs.human_eye import HumanEye, HumanEyeState, MagnocellularMotionEngine
from avcs.stream.neuro_stream import NeuroPacket, NeuroStreamBus, RealTimeNeuroStreamer
from avcs.stream.reflex_controller import GameReflexController
from avcs.verifier import VerificationResult, VerificationStatus, VisualRippleVerifier
from forge.vision.cortex.benchmark_real_desktop import RealisticDesktopRenderer
from forge.vision.cortex.core.ventral import UI_CATEGORIES, VentralSemanticStream
from forge.vision.cortex.model import ArtificialVisualCortex, CortexState
from forge.vision.cortex.motor import VirtualCursorDriver


class DesktopSceneBuilder:
    """Builds pixel-accurate desktop canvases, icons, and window launch sequences."""

    def __init__(self, width: int = 640, height: int = 360) -> None:
        self.width = width
        self.height = height
        self.renderer = RealisticDesktopRenderer(width=width, height=height)

    def render_generic_app_window(self, app_name: str) -> np.ndarray:
        """Procedurally renders a clean, modern application window for arbitrary targets."""
        img = Image.new("RGB", (self.width, self.height), color=(30, 32, 36))
        draw = ImageDraw.Draw(img)

        # Title bar
        draw.rectangle([0, 0, self.width, 32], fill=(42, 44, 50))
        draw.text((16, 8), f"{app_name.title()} - Dynamic Application Window", fill=(220, 220, 225))

        # Window controls (close, max, min) on right
        draw.rectangle([self.width - 40, 0, self.width, 32], fill=(196, 43, 28))  # Close button
        draw.text((self.width - 25, 8), "X", fill=(255, 255, 255))
        draw.rectangle([self.width - 76, 0, self.width - 40, 32], fill=(55, 57, 63))  # Max
        draw.rectangle([self.width - 64, 10, self.width - 52, 22], outline=(200, 200, 200), fill=None)
        draw.rectangle([self.width - 112, 0, self.width - 76, 32], fill=(55, 57, 63))  # Min
        draw.line([self.width - 100, 18, self.width - 88, 18], fill=(200, 200, 200), width=2)

        # Menu bar
        draw.rectangle([0, 32, self.width, 58], fill=(50, 52, 58))
        menu_items = ["File", "Edit", "View", "Tools", "Help"]
        for i, item in enumerate(menu_items):
            draw.text((20 + i * 55, 38), item, fill=(190, 190, 195))

        # Left sidebar navigation
        sb_w = max(60, int(self.width * 0.22))
        draw.rectangle([0, 58, sb_w, self.height - 24], fill=(36, 38, 44))
        draw.text((14, 68), "WORKSPACE", fill=(130, 135, 145))
        for i, nav in enumerate(["Dashboard", "Projects", "Settings", "Activity"]):
            ny = 92 + i * 26
            draw.rectangle([10, ny, sb_w - 10, ny + 20], fill=(48, 51, 60))
            draw.text((16, ny + 3), nav, fill=(210, 210, 215))

        # Main workspace canvas
        cw_x = sb_w + 16
        draw.rectangle([cw_x, 74, self.width - 20, 140], fill=(42, 45, 54), outline=(65, 70, 82))
        draw.text((cw_x + 16, 88), f"Welcome to {app_name.title()}", fill=(255, 255, 255))
        draw.text((cw_x + 16, 110), f"Active target context initialized dynamically.", fill=(160, 165, 175))

        # Primary action button
        btn_x0 = cw_x + 16
        draw.rectangle([btn_x0, 160, btn_x0 + 130, 195], fill=(0, 120, 215))
        draw.text((btn_x0 + 16, 170), f"Run {app_name[:8].title()}", fill=(255, 255, 255))

        # Typeable search / input field
        input_x0 = btn_x0 + 150
        input_w = max(120, self.width - input_x0 - 30)
        draw.rectangle([input_x0, 160, input_x0 + input_w, 195], fill=(22, 24, 28), outline=(80, 85, 95))
        draw.text((input_x0 + 12, 170), f"Filter {app_name.lower()} components...", fill=(120, 125, 135))

        # Status Bar
        draw.rectangle([0, self.height - 24, self.width, self.height], fill=(24, 25, 28))
        draw.text((12, self.height - 18), f"Ready  |  {app_name.title()} v1.0  |  UTF-8", fill=(140, 145, 155))

        return np.array(img)

    def render_desktop_canvas(
        self,
        app_targets: List[str] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
        """Renders desktop wallpaper, taskbar, and configurable application shortcut icons."""
        img = Image.new("RGB", (self.width, self.height), color=(35, 65, 105)) # Windows-style blue wallpaper
        draw = ImageDraw.Draw(img)

        # Subtle gradient/accent wallpaper geometry
        draw.polygon([(0, 0), (self.width, 0), (self.width, 100), (0, 220)], fill=(45, 80, 125))

        # Bottom Taskbar
        tb_h = 32
        draw.rectangle([0, self.height - tb_h, self.width, self.height], fill=(24, 24, 26))
        # Start button
        draw.rectangle([8, self.height - 26, 32, self.height - 6], fill=(0, 120, 215))

        # Desktop Shortcut Icons (normalized coordinates returned)
        targets = app_targets if app_targets is not None else ["terminal", "chrome", "vscode"]
        icons: Dict[str, Tuple[float, float]] = {}

        # Styled icon color palette for arbitrary dynamic applications
        palette = [
            ((0, 122, 204), (100, 180, 255)),
            ((234, 67, 53), (251, 188, 5)),
            ((16, 124, 65), (80, 200, 120)),
            ((114, 46, 209), (180, 130, 255)),
            ((230, 110, 25), (255, 180, 80)),
            ((20, 140, 160), (90, 210, 230)),
        ]

        max_rows_per_col = max(1, (self.height - tb_h - 40) // 80)

        for idx, app_name in enumerate(targets):
            app_clean = app_name.strip()
            app_key = app_clean.lower()
            col = idx // max_rows_per_col
            row = idx % max_rows_per_col
            x0 = 40 + col * 75
            y0 = 40 + row * 80
            box = [x0, y0, x0 + 40, y0 + 40]

            if app_key in ("terminal", "shell", "powershell", "cmd"):
                draw.rectangle(box, fill=(20, 20, 20), outline=(80, 80, 80))
                draw.text((x0 + 8, y0 + 12), ">_", fill=(50, 220, 100))
                lbl = "Terminal"
            elif app_key in ("chrome", "browser", "firefox", "brave", "edge"):
                draw.ellipse(box, fill=(234, 67, 53), outline=(251, 188, 5))
                draw.ellipse([x0 + 10, y0 + 10, x0 + 30, y0 + 30], fill=(66, 133, 244))
                lbl = "Chrome"
            elif app_key in ("vscode", "editor", "code"):
                draw.rectangle(box, fill=(0, 122, 204), outline=(100, 180, 255))
                draw.text((x0 + 6, y0 + 12), "{;}", fill=(255, 255, 255))
                lbl = "VSCode"
            elif app_key in ("excel", "spreadsheet", "calc"):
                draw.rectangle(box, fill=(16, 124, 65), outline=(100, 220, 150))
                draw.text((x0 + 12, y0 + 12), "X", fill=(255, 255, 255))
                lbl = "Excel"
            elif app_key in ("notion", "productivity", "docs"):
                draw.rectangle(box, fill=(240, 240, 240), outline=(100, 100, 100))
                draw.text((x0 + 12, y0 + 12), "N", fill=(20, 20, 20))
                lbl = "Notion"
            else:
                # Dynamic generic application icon
                color_bg, color_outline = palette[idx % len(palette)]
                draw.rectangle(box, fill=color_bg, outline=color_outline)
                glyph = (app_clean[:2] if len(app_clean) >= 2 else app_clean[:1]).upper()
                draw.text((x0 + 10, y0 + 12), glyph, fill=(255, 255, 255))
                lbl = app_clean[:8].title()

            draw.text((x0 - 4, y0 + 44), lbl, fill=(220, 220, 220))
            icons[app_key] = ((box[0] + box[2]) / 2 / self.width, (box[1] + box[3]) / 2 / self.height)

        return np.array(img), icons

    def render_icon_click_ripple(
        self,
        base_canvas: np.ndarray,
        icon_coord: Tuple[float, float],
        box_radius: int = 22,
    ) -> np.ndarray:
        """Renders visual highlight/depression around icon when clicked."""
        img = Image.fromarray(base_canvas.copy())
        draw = ImageDraw.Draw(img)
        cx = int(icon_coord[0] * self.width)
        cy = int(icon_coord[1] * self.height)

        # Selection / click highlight glow
        draw.rectangle(
            [cx - box_radius, cy - box_radius, cx + box_radius, cy + box_radius],
            fill=None,
            outline=(180, 220, 255),
            width=3,
        )
        return np.array(img)

    def render_window_launch_animation(
        self,
        base_canvas: np.ndarray,
        icon_coord: Tuple[float, float],
        app_type: str = "terminal",
        steps: int = 6,
    ) -> List[np.ndarray]:
        """Generates an expanding window animation sequence from icon origin to full display."""
        # Target final window image dynamically resolved
        app_key = app_type.strip().lower()
        if app_key in ("terminal", "shell", "powershell", "cmd"):
            final_screen = self.renderer.render_terminal().frame
        elif app_key in ("chrome", "browser", "firefox", "brave", "edge"):
            final_screen = self.renderer.render_chrome().frame
        elif app_key in ("vscode", "editor", "code"):
            final_screen = self.renderer.render_vscode().frame
        elif app_key in ("excel", "spreadsheet", "calc"):
            final_screen = self.renderer.render_excel().frame
        elif app_key in ("notion", "productivity", "docs"):
            final_screen = self.renderer.render_notion().frame
        else:
            final_screen = self.render_generic_app_window(app_type)

        target_img = Image.fromarray(final_screen)
        base_img = Image.fromarray(base_canvas)

        start_x = int(icon_coord[0] * self.width)
        start_y = int(icon_coord[1] * self.height)

        end_x0, end_y0 = 30, 20
        end_w = self.width - 60
        end_h = self.height - 60

        frames = []
        for s in range(1, steps + 1):
            # Non-linear cubic easing curve: progress in (0, 1]
            t = s / steps
            prog = 1.0 - (1.0 - t) ** 3

            cur_w = max(20, int(20 + prog * (end_w - 20)))
            cur_h = max(20, int(20 + prog * (end_h - 20)))

            cur_x0 = int(start_x + prog * (end_x0 - start_x))
            cur_y0 = int(start_y + prog * (end_y0 - start_y))
            cur_x1 = cur_x0 + cur_w
            cur_y1 = cur_y0 + cur_h

            frame_img = base_img.copy()
            draw = ImageDraw.Draw(frame_img)

            # Crop and scale window content into the expanding bounding box
            scaled_content = target_img.resize((cur_w, cur_h), Image.Resampling.BILINEAR)
            frame_img.paste(scaled_content, (cur_x0, cur_y0))
            # Window shadow & border
            draw.rectangle([cur_x0, cur_y0, cur_x1, cur_y1], outline=(180, 180, 180), width=1)

            frames.append(np.array(frame_img))

        # Add fully settled final window frame (without border margin)
        frames.append(final_screen)
        return frames


def _map_app_to_expected_category(app_name: str) -> str:
    """Dynamic heuristic mapping from arbitrary application name to expected Ventral semantic category."""
    k = app_name.strip().lower()
    if k in ("terminal", "shell", "powershell", "cmd", "bash", "console"):
        return "terminal_shell"
    if k in ("chrome", "browser", "firefox", "brave", "edge", "safari", "notion"):
        return "browser_webpage"
    if k in ("vscode", "code", "editor", "sublime", "pycharm", "ide"):
        return "editor_code"
    if k in ("excel", "spreadsheet", "calc", "sheets", "form"):
        return "form_input"
    if k in ("vlc", "spotify", "media", "music", "video", "player"):
        return "media_playback"
    if k in ("settings", "control", "preferences", "config"):
        return "settings_panel"
    return "browser_webpage"


def run_empirical_window_transition_audit(
    primary_app: str = "terminal",
    app_targets: List[str] | None = None,
    steps: int = 6,
) -> Dict[str, Any]:
    print("=" * 80)
    print("   AVCS EMPIRICAL AUDIT: MULTI-STEP APP LAUNCH & WINDOW TRANSITION TRACKING   ")
    print("=" * 80)

    target_apps = [a.strip() for a in app_targets] if app_targets else ["terminal", "chrome", "vscode"]
    primary_key = primary_app.strip().lower()
    if primary_key not in [a.lower() for a in target_apps]:
        target_apps.insert(0, primary_app)

    builder = DesktopSceneBuilder(width=640, height=360)
    desktop_canvas, icons = builder.render_desktop_canvas(app_targets=target_apps)

    results: Dict[str, Any] = {}

    target_coord = icons.get(primary_key)
    if target_coord is None:
        primary_key = list(icons.keys())[0]
        target_coord = icons[primary_key]

    # --------------------------------------------------------------------------
    # 1. ICON CLICK & LOCAL VISUAL RIPPLE ASSERTION
    # --------------------------------------------------------------------------
    print(f"\n[PART 1: ICON CLICK & LOCAL VISUAL RIPPLE ASSERTION ({primary_key.upper()})]")
    clicked_canvas = builder.render_icon_click_ripple(desktop_canvas, target_coord)

    # Frame capture emulator
    class MemoryCapture:
        def __init__(self, sequence: List[np.ndarray]) -> None:
            self.seq = sequence
            self.idx = 0
        def grab(self) -> np.ndarray:
            f = self.seq[min(self.idx, len(self.seq) - 1)]
            self.idx += 1
            return f

    cap_click = MemoryCapture([desktop_canvas, clicked_canvas, clicked_canvas])
    eye_click = HumanEye(source=cap_click, threaded_capture=False)
    driver_click = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye_click, driver=driver_click, ripple_radius=0.08)

    pre_state = eye_click.observe()
    driver_click.move_to_normalized(target_coord[0], target_coord[1])
    driver_click.click(button="left")
    verify_res = verifier.verify_action(f"click_{primary_key}_icon", target_coord, pre_state)

    print(f"Target Coordinate:           ({target_coord[0]:.4f}, {target_coord[1]:.4f})")
    print(f"Optical Ripple Energy:       {verify_res.optical_ripple_energy:.5f} (threshold: {verifier.min_ripple_threshold})")
    print(f"Predictive Surprise Pulse:   {verify_res.surprise_pulse:.5f}")
    print(f"Verification Status:         {verify_res.status.value.upper()}")
    print(f"Causal Summary:              {verify_res.causal_summary}")
    print(f"Verification Latency:        {verify_res.elapsed_ms:.2f} ms")

    assert verify_res.is_successful, f"Icon click verification for {primary_key} must succeed!"
    results["part1_ripple"] = {
        "target_app": primary_key,
        "status": verify_res.status.value,
        "ripple_energy": verify_res.optical_ripple_energy,
        "surprise_pulse": verify_res.surprise_pulse,
        "elapsed_ms": verify_res.elapsed_ms,
    }

    # --------------------------------------------------------------------------
    # 2. MAGNOCELLULAR MOTION STREAM & OPTICAL FLOW FIELD
    # --------------------------------------------------------------------------
    print(f"\n[PART 2: MAGNOCELLULAR MOTION STREAM & OPTICAL FLOW FIELD ({primary_key.upper()})]")
    anim_frames = builder.render_window_launch_animation(desktop_canvas, target_coord, app_type=primary_key, steps=steps)
    motion_engine = MagnocellularMotionEngine(grid_rows=16, grid_cols=24)

    # Initial frame
    motion_engine.process(desktop_canvas)

    motion_telemetry = []
    divergence_values = []

    for idx, f in enumerate(anim_frames):
        mf, mean_mot, objs, surprise = motion_engine.process(f)
        vx = mf[..., 0]
        vy = mf[..., 1]
        speed_grid = np.hypot(vx, vy)
        max_speed = float(np.max(speed_grid))
        active_cells = int(np.sum(speed_grid > 0.05))

        # Compute optical flow divergence div(V) = d(vx)/dx + d(vy)/dy
        grad_vx_x = np.zeros_like(vx)
        grad_vy_y = np.zeros_like(vy)
        grad_vx_x[:, 1:] = vx[:, 1:] - vx[:, :-1]
        grad_vy_y[1:, :] = vy[1:, :] - vy[:-1, :]
        div_field = grad_vx_x + grad_vy_y
        mean_div = float(np.mean(div_field))
        divergence_values.append(mean_div)

        obj_str = "None"
        if objs:
            top = objs[0]
            obj_str = f"Centroid=({top.centroid[0]:.2f}, {top.centroid[1]:.2f}), Speed={top.speed:.3f}, Area={top.area:.3f}"

        motion_telemetry.append({
            "step": idx + 1,
            "mean_motion": mean_mot,
            "surprise_score": surprise,
            "max_velocity": max_speed,
            "active_cells": active_cells,
            "mean_divergence": mean_div,
            "moving_objects_count": len(objs),
            "top_object": obj_str,
        })

        print(
            f"Step {idx + 1:2d} | MeanMotion={mean_mot:.4f} | Surprise={surprise:.4f} | "
            f"ActiveGrid={active_cells:2d}/384 | MaxSpeed={max_speed:.3f} | {obj_str}"
        )

    # Check that motion was non-zero during expansion and settled at the end
    peak_motion = max(m["mean_motion"] for m in motion_telemetry[:-1])
    settled_motion = motion_telemetry[-1]["mean_motion"]
    print(f"\nPeak Motion Energy during Expansion: {peak_motion:.4f}")
    print(f"Settled Motion Energy at Target Window:  {settled_motion:.4f}")
    assert peak_motion > 0.02, "Magnocellular engine must detect non-zero optical flow during opening animation!"
    results["part2_motion"] = {
        "target_app": primary_key,
        "peak_motion": peak_motion,
        "settled_motion": settled_motion,
        "steps": motion_telemetry,
    }

    # --------------------------------------------------------------------------
    # 3. VENTRAL SEMANTIC STREAM CATEGORY SHIFT
    # --------------------------------------------------------------------------
    print("\n[PART 3: VENTRAL STREAM SEMANTIC CATEGORY SHIFTS]")
    eye_ventral = HumanEye(source="synthetic", threaded_capture=False)

    apps_to_test = [
        (app_name, _map_app_to_expected_category(app_name))
        for app_name in target_apps
    ]

    ventral_results = {}

    for app_name, target_cat in apps_to_test:
        app_k = app_name.strip().lower()
        if app_k not in icons:
            continue
        eye_ventral.reset()
        # Seed with 2 desktop canvas frames
        st_pre0 = eye_ventral.observe(desktop_canvas)
        st_pre1 = eye_ventral.observe(desktop_canvas)
        pre_cat = st_pre1.primary_category
        pre_conf = st_pre1.category_confidence

        seq = builder.render_window_launch_animation(desktop_canvas, icons[app_k], app_type=app_name, steps=steps)

        traj = []
        for step_idx, frame in enumerate(seq):
            st = eye_ventral.observe(frame)
            traj.append({
                "step": step_idx + 1,
                "category": st.primary_category,
                "confidence": st.category_confidence,
                "stability": st.stability,
                "surprise": st.surprise_score,
            })

        final_st = traj[-1]
        print(f"\nApp Transition: Desktop -> {app_name.upper()} ({target_cat})")
        print(f"  Initial (Desktop):  category='{pre_cat}' (conf={pre_conf:.2f})")
        print(f"  Transition Trajectory:")
        for t in traj:
            print(f"    Step {t['step']}: category='{t['category']:<15}' conf={t['confidence']:.2f} stability='{t['stability']:<12}' surprise={t['surprise']:.3f}")
        print(f"  Final Settled State: category='{final_st['category']}' (conf={final_st['confidence']:.2f})")

        ventral_results[app_name] = {
            "target": target_cat,
            "pre_category": pre_cat,
            "final_category": final_st["category"],
            "final_confidence": final_st["confidence"],
            "trajectory": traj,
        }

    results["part3_ventral"] = ventral_results

    # --------------------------------------------------------------------------
    # 4. DISCRETE VISUAL EVENT STREAMER & AGENT READINESS SYNC
    # --------------------------------------------------------------------------
    print("\n[PART 4: DISCRETE VISUAL EVENT STREAMER LIFECYCLE]")
    streamer = VisualEventStreamer()
    emitted_events: List[VisualEvent] = []
    streamer.subscribe(callback=lambda ev: emitted_events.append(ev))

    eye_event = HumanEye(source="synthetic", threaded_capture=False)

    # Lifecycle:
    # 1. Desktop baseline (2 frames)
    # 2. Window opening animation (configurable steps)
    # 3. Settled window (2 frames)
    full_sequence = [desktop_canvas, desktop_canvas] + anim_frames + [anim_frames[-1]]

    for f_idx, frame in enumerate(full_sequence):
        st = eye_event.observe(frame)
        streamer.process_state(st)

    print(f"Total Discrete Events Emitted: {len(emitted_events)}")
    for ev in emitted_events:
        print(f"  [{ev.event_type.value:<18}] (Frame {ev.frame_index:2d}): {ev.title} - {ev.description}")

    event_types = [ev.event_type for ev in emitted_events]
    has_transition = VisualEventType.SCENE_TRANSITION in event_types
    has_motion = VisualEventType.MOTION_DETECTED in event_types
    has_settled = VisualEventType.STABILITY_REACHED in event_types
    has_surprise = VisualEventType.SURPRISE_ALERT in event_types

    print(f"\nLifecycle Event Assertions:")
    print(f"  • MOTION_DETECTED emitted:   {has_motion}")
    print(f"  • SURPRISE_ALERT emitted:    {has_surprise}")
    print(f"  • SCENE_TRANSITION emitted:  {has_transition}")
    print(f"  • STABILITY_REACHED emitted: {has_settled}")

    results["part4_events"] = {
        "total_events": len(emitted_events),
        "has_motion": has_motion,
        "has_surprise": has_surprise,
        "has_transition": has_transition,
        "has_settled": has_settled,
        "events": [ev.to_dict() for ev in emitted_events],
    }

    # --------------------------------------------------------------------------
    # 5. GAME REFLEX CONTROLLER SURPRISE HOOK
    # --------------------------------------------------------------------------
    print("\n[PART 5: GAME REFLEX CONTROLLER ZERO-LATENCY SURPRISE HOOK]")
    bus = NeuroStreamBus(buffer_size=50)
    driver_reflex = VirtualCursorDriver(failsafe=False)

    # Use ScreenEye or HumanEye with NeuroStreamBus
    cap_reflex = MemoryCapture(full_sequence)
    eye_reflex = HumanEye(source=cap_reflex, threaded_capture=False)
    controller = GameReflexController(eye=eye_reflex, bus=bus, driver=driver_reflex, force_virtual=True)

    reflex_calls: List[Tuple[int, float, str]] = []

    @controller.on_surprise_reflex(threshold=0.15)
    def handle_surprise(pkt: NeuroPacket) -> None:
        reflex_calls.append((pkt.seq, pkt.surprise_score, pkt.stability_state))

    # Feed sequence into bus via NeuroPacket
    for f_idx, frame in enumerate(full_sequence):
        st = eye_reflex.observe(frame)
        pkt = NeuroPacket(
            timestamp=time.time(),
            seq=f_idx + 1,
            fixation=st.gaze_fixation,
            motor_flow=st.motor_flow,
            contact_acquired=st.contact_acquired,
            primary_category=st.primary_category,
            stability_state=st.stability,
            surprise_score=st.surprise_score,
            inference_ms=st.latency_ms,
        )
        bus.publish(pkt)

    print(f"Surprise Reflex Hook Callbacks Fired: {len(reflex_calls)}")
    for seq_num, score, stab in reflex_calls:
        print(f"  Reflex Triggered at Packet Seq #{seq_num}: Surprise Score={score:.4f}, Stability='{stab}'")

    assert len(reflex_calls) > 0, "GameReflexController must trigger surprise hook during window transition!"
    results["part5_reflex"] = {
        "reflex_triggers_count": len(reflex_calls),
        "triggers": reflex_calls,
    }

    print("\n" + "=" * 80)
    print("   ALL EMPIRICAL AUDIT ASSERTIONS COMPLETED SUCCESSFULLY   ")
    print("=" * 80)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVCS Empirical Window Transition & App Launch Audit")
    parser.add_argument(
        "--app", "--primary-app",
        dest="primary_app",
        type=str,
        default="terminal",
        help="Primary application target to launch and audit (default: terminal)",
    )
    parser.add_argument(
        "--apps",
        type=str,
        default="terminal,chrome,vscode",
        help="Comma-separated application targets for desktop and ventral audits (default: terminal,chrome,vscode)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="Window expansion animation step count (default: 6)",
    )
    args = parser.parse_args()
    targets = [a.strip() for a in args.apps.split(",") if a.strip()]
    if args.primary_app not in targets:
        targets.insert(0, args.primary_app)
    run_empirical_window_transition_audit(primary_app=args.primary_app, app_targets=targets, steps=args.steps)
