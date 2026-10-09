"""AVCS Native Desktop Computer GUI & Visual Neural HUD.

A native desktop application (Tkinter + PIL + Win32/Virtual Motor) providing
real-time visual perception monitoring, natural language gaze grounding,
dorsal affordance visualization, and physical closed-loop mouse steering.
"""
from __future__ import annotations

import math
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk, ImageDraw

from avcs import ScreenEye, CortexState
from forge.vision.cortex.motor import (
    HandEyeServoingController,
    get_default_driver,
    SafetyBreakoutException,
)


class AVCSDesktopApp:
    """Native Computer Desktop GUI for AVCS."""

    def __init__(self, root: tk.Tk, fps: int = 25) -> None:
        self.root = root
        self.root.title("AVCS // Artificial Visual Cortex - Desktop Station")
        self.root.geometry("1180x760")
        self.root.minsize(960, 640)
        self.root.configure(bg="#0d1117")

        self.fps = fps
        self.eye = ScreenEye(fps=fps)
        self.driver = get_default_driver(force_virtual=False)
        self.servoing_controller = HandEyeServoingController(driver=self.driver)

        # State tracking
        self.running = True
        self.show_fovea = True
        self.show_affordances = True
        self.show_trajectory = True
        self.always_on_top = False
        self.current_target_gaze: Optional[Tuple[float, float]] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_state: Optional[CortexState] = None

        self._build_ui()
        self._start_perception_thread()
        self._schedule_gui_refresh()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        # Top Header Bar
        header = tk.Frame(self.root, bg="#161b22", height=50, padx=16, pady=10)
        header.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="PRAXIS NEURAL // AVCS DESKTOP CORTEX HUD",
            font=("Segoe UI", 12, "bold"),
            fg="#58a6ff",
            bg="#161b22",
        )
        title_lbl.pack(side=tk.LEFT)

        self.status_badge = tk.Label(
            header,
            text="● LIVE OCULAR STREAM",
            font=("Segoe UI", 10, "bold"),
            fg="#3fb950",
            bg="#161b22",
            padx=10,
        )
        self.status_badge.pack(side=tk.RIGHT)

        # Main Layout: Canvas on Left (70%), Control Panel on Right (30%)
        main_body = tk.Frame(self.root, bg="#0d1117", padx=12, pady=12)
        main_body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left Column: Desktop Visual Screen Canvas
        left_frame = tk.Frame(main_body, bg="#161b22", highlightbackground="#30363d", highlightthickness=1)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        canvas_header = tk.Frame(left_frame, bg="#21262d", padx=10, pady=6)
        canvas_header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(
            canvas_header,
            text="REAL-TIME DESKTOP PERCEPTION & FOVEAL RETICLE",
            font=("Segoe UI", 9, "bold"),
            fg="#8b949e",
            bg="#21262d",
        ).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(left_frame, bg="#000000", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Right Column: Controls & Telemetry
        right_frame = tk.Frame(main_body, bg="#161b22", width=360, highlightbackground="#30363d", highlightthickness=1, padx=14, pady=14)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        # Section 1: Natural Language Gaze Grounding
        sec1 = tk.LabelFrame(
            right_frame,
            text=" 1. Language Gaze Grounding ",
            font=("Segoe UI", 10, "bold"),
            fg="#58a6ff",
            bg="#161b22",
            padx=10,
            pady=8,
        )
        sec1.pack(fill=tk.X, pady=(0, 12))

        tk.Label(sec1, text="Target Element Query:", font=("Segoe UI", 9), fg="#c9d1d9", bg="#161b22").pack(anchor=tk.W)
        self.query_entry = tk.Entry(sec1, font=("Segoe UI", 10), bg="#0d1117", fg="#ffffff", insertbackground="#58a6ff", relief=tk.FLAT)
        self.query_entry.insert(0, "Submit Button")
        self.query_entry.pack(fill=tk.X, pady=4)

        btn_box = tk.Frame(sec1, bg="#161b22")
        btn_box.pack(fill=tk.X, pady=4)

        self.btn_look = tk.Button(
            btn_box,
            text="Look At Target",
            font=("Segoe UI", 9, "bold"),
            bg="#238636",
            fg="#ffffff",
            activebackground="#2ea043",
            relief=tk.FLAT,
            command=self.on_look_at_clicked,
            padx=8,
            pady=4,
        )
        self.btn_look.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.gaze_pos_lbl = tk.Label(sec1, text="Gaze Fixation: (0.50, 0.50)", font=("Consolas", 9), fg="#8b949e", bg="#161b22")
        self.gaze_pos_lbl.pack(anchor=tk.W, pady=(4, 0))

        # Section 2: Physical Hand-Eye Motor Servoing
        sec2 = tk.LabelFrame(
            right_frame,
            text=" 2. Physical Motor Servoing ",
            font=("Segoe UI", 10, "bold"),
            fg="#58a6ff",
            bg="#161b22",
            padx=10,
            pady=8,
        )
        sec2.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            sec2,
            text="Closed-loop Fitts' Law steering to target gaze:",
            font=("Segoe UI", 8),
            fg="#8b949e",
            bg="#161b22",
        ).pack(anchor=tk.W, pady=(0, 4))

        btn_steer_box = tk.Frame(sec2, bg="#161b22")
        btn_steer_box.pack(fill=tk.X, pady=2)

        self.btn_steer = tk.Button(
            btn_steer_box,
            text="Steer Cursor",
            font=("Segoe UI", 9, "bold"),
            bg="#1f6feb",
            fg="#ffffff",
            activebackground="#388bfd",
            relief=tk.FLAT,
            command=lambda: self.on_steer_clicked(click=False),
            pady=4,
        )
        self.btn_steer.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_steer_click = tk.Button(
            btn_steer_box,
            text="Steer & Click",
            font=("Segoe UI", 9, "bold"),
            bg="#8957e5",
            fg="#ffffff",
            activebackground="#a371f7",
            relief=tk.FLAT,
            command=lambda: self.on_steer_clicked(click=True),
            pady=4,
        )
        self.btn_steer_click.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        self.motor_status_lbl = tk.Label(
            sec2,
            text="Fail-Safe: Corner Breakout ACTIVE (Move to corner to abort)",
            font=("Segoe UI", 8),
            fg="#d29922",
            bg="#161b22",
            wraplength=310,
            justify=tk.LEFT,
        )
        self.motor_status_lbl.pack(anchor=tk.W, pady=(6, 0))

        # Section 3: Live Telemetry
        sec3 = tk.LabelFrame(
            right_frame,
            text=" 3. Cortical Telemetry ",
            font=("Segoe UI", 10, "bold"),
            fg="#58a6ff",
            bg="#161b22",
            padx=10,
            pady=8,
        )
        sec3.pack(fill=tk.X, pady=(0, 12))

        self.lbl_category = self._make_stat_row(sec3, "UI Category:", "editor_code", "#58a6ff")
        self.lbl_stability = self._make_stat_row(sec3, "Stability:", "stable", "#3fb950")
        self.lbl_surprise = self._make_stat_row(sec3, "Surprise Error:", "0.003", "#f0f6fc")
        self.lbl_hotspots = self._make_stat_row(sec3, "Affordances:", "2 Buttons, 2 Inputs", "#f0f6fc")
        self.lbl_fps = self._make_stat_row(sec3, "Perception FPS:", f"{self.fps} FPS", "#3fb950")

        # Section 4: Display Options & Toggles
        sec4 = tk.LabelFrame(
            right_frame,
            text=" 4. HUD Visual Overlays ",
            font=("Segoe UI", 10, "bold"),
            fg="#58a6ff",
            bg="#161b22",
            padx=10,
            pady=8,
        )
        sec4.pack(fill=tk.X)

        self.var_fovea = tk.BooleanVar(value=True)
        tk.Checkbutton(
            sec4,
            text="Show Foveal Reticle Crosshair",
            variable=self.var_fovea,
            fg="#c9d1d9",
            bg="#161b22",
            selectcolor="#0d1117",
            activebackground="#161b22",
            command=self._on_overlay_toggle,
        ).pack(anchor=tk.W)

        self.var_affordance = tk.BooleanVar(value=True)
        tk.Checkbutton(
            sec4,
            text="Show Action Affordance Boxes",
            variable=self.var_affordance,
            fg="#c9d1d9",
            bg="#161b22",
            selectcolor="#0d1117",
            activebackground="#161b22",
            command=self._on_overlay_toggle,
        ).pack(anchor=tk.W)

        self.var_top = tk.BooleanVar(value=False)
        tk.Checkbutton(
            sec4,
            text="Always on Top (Overlay Mode)",
            variable=self.var_top,
            fg="#c9d1d9",
            bg="#161b22",
            selectcolor="#0d1117",
            activebackground="#161b22",
            command=self._on_top_toggle,
        ).pack(anchor=tk.W)

    def _make_stat_row(self, parent: tk.Widget, label_text: str, val_text: str, val_color: str) -> tk.Label:
        row = tk.Frame(parent, bg="#161b22")
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label_text, font=("Segoe UI", 9), fg="#8b949e", bg="#161b22").pack(side=tk.LEFT)
        val_lbl = tk.Label(row, text=val_text, font=("Consolas", 9, "bold"), fg=val_color, bg="#161b22")
        val_lbl.pack(side=tk.RIGHT)
        return val_lbl

    def _on_overlay_toggle(self) -> None:
        self.show_fovea = self.var_fovea.get()
        self.show_affordances = self.var_affordance.get()

    def _on_top_toggle(self) -> None:
        self.always_on_top = self.var_top.get()
        self.root.attributes("-topmost", self.always_on_top)

    def on_look_at_clicked(self) -> None:
        query = self.query_entry.get().strip()
        if not query:
            return
        # Execute gaze grounding
        try:
            target = self.eye.look_at(query)
            self.current_target_gaze = target
            self.gaze_pos_lbl.config(text=f"Gaze Fixation: ({target[0]:.2f}, {target[1]:.2f})")
        except Exception as e:
            messagebox.showerror("Grounding Error", str(e))

    def on_steer_clicked(self, click: bool = False) -> None:
        target = self.current_target_gaze or self.eye.cortex.retina.current_fixation

        def _run_motor():
            try:
                self.motor_status_lbl.config(text="Steering mouse cursor...", fg="#58a6ff")
                res = self.servoing_controller.servo_to(
                    target_gaze=target,
                    timeout_seconds=2.0,
                    click_on_arrival=click,
                    speed_factor=1.6,
                    realtime=True,
                )
                if res.contact_acquired:
                    status_text = f"Contact Locked! ({res.elapsed_ms:.0f} ms)" + (" Clicked." if click else "")
                    self.motor_status_lbl.config(text=status_text, fg="#3fb950")
                else:
                    self.motor_status_lbl.config(text="Servoing timed out near target.", fg="#d29922")
            except SafetyBreakoutException as exc:
                self.motor_status_lbl.config(text="SAFETY ABORT: Cursor moved to corner!", fg="#f85149")

        threading.Thread(target=_run_motor, daemon=True).start()

    def _start_perception_thread(self) -> None:
        def _loop():
            tick_dt = 1.0 / self.fps
            while self.running:
                t0 = time.time()
                try:
                    frame = self.eye.source.grab()
                    state = self.eye.perceive_now(frame=frame)
                    self.latest_frame = frame
                    self.latest_state = state
                except Exception:
                    pass

                dt = time.time() - t0
                sleep_sec = tick_dt - dt
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

        threading.Thread(target=_loop, daemon=True).start()

    def _schedule_gui_refresh(self) -> None:
        if not self.running:
            return

        # Render frame onto Canvas
        if self.latest_frame is not None and self.latest_state is not None:
            self._render_hud_frame(self.latest_frame, self.latest_state)
            self._update_telemetry_labels(self.latest_state)

        # Schedule next tick (~30 FPS refresh)
        self.root.after(33, self._schedule_gui_refresh)

    def _render_hud_frame(self, frame: np.ndarray, state: CortexState) -> None:
        # Resize frame to fit canvas
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())

        img = Image.fromarray(frame).resize((cw, ch), Image.BILINEAR)
        draw = ImageDraw.Draw(img)

        # 1. Draw Dorsal Affordance Hotspots
        if self.show_affordances and state.action_hotspots:
            for nx, ny, act_type, conf in state.action_hotspots:
                px = int(nx * cw)
                py = int(ny * ch)
                radius = 16
                color = "#00ff88" if act_type == "clickable" else "#ffaa00"

                # Box outline
                draw.rectangle([px - radius, py - radius, px + radius, py + radius], outline=color, width=2)
                draw.text((px - radius, py - radius - 14), f"{act_type} {conf:.0%}", fill=color)

        # 2. Draw Foveal Gaze Reticle Crosshair
        if self.show_fovea:
            fx, fy = self.current_target_gaze or state.fixation
            cx = int(fx * cw)
            cy = int(fy * ch)
            cross_size = 20
            reticle_color = "#00ffff"  # Neon cyan

            # Outer targeting circle
            draw.ellipse([cx - 24, cy - 24, cx + 24, cy + 24], outline=reticle_color, width=2)
            # Center crosshair
            draw.line([cx - cross_size, cy, cx + cross_size, cy], fill=reticle_color, width=2)
            draw.line([cx, cy - cross_size, cx, cy + cross_size], fill=reticle_color, width=2)
            draw.text((cx + 12, cy + 12), "FOVEA", fill=reticle_color)

        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas._current_photo = photo  # prevent garbage collection

    def _update_telemetry_labels(self, state: CortexState) -> None:
        self.lbl_category.config(text=state.primary_category)
        stab_color = "#3fb950" if state.stability_state == "stable" else "#d29922"
        self.lbl_stability.config(text=state.stability_state, fg=stab_color)
        self.lbl_surprise.config(text=f"{state.surprise_score:.4f}")

        clicks = len([h for h in (state.action_hotspots or []) if h[2] == "clickable"])
        types = len([h for h in (state.action_hotspots or []) if h[2] == "typeable"])
        self.lbl_hotspots.config(text=f"{clicks} Buttons, {types} Inputs")

    def on_close(self) -> None:
        self.running = False
        try:
            self.eye.stop()
        except Exception:
            pass
        self.root.destroy()


def launch_gui() -> None:
    """Launches the AVCS Desktop Station GUI application."""
    root = tk.Tk()
    app = AVCSDesktopApp(root=root, fps=25)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
