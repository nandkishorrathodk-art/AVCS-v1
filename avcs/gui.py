"""AVCS Native Desktop Computer GUI & Visual Neural HUD.

A native desktop application (Tkinter + PIL + Win32/Virtual Motor) providing
real-time visual perception monitoring, natural language gaze grounding,
dorsal affordance visualization, and physical closed-loop mouse steering.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import threading
import time
from typing import Any, List, Optional, Tuple, Union

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError as exc:  # pragma: no cover - depends on host Tk install
    raise ImportError(
        "avcs.gui requires Tkinter, which is not available in this Python "
        "environment. On Debian/Ubuntu install the system Tk runtime with "
        "`apt-get install -y python3-tk` (provides libtk8.6.so); on Windows "
        "reinstall Python with the 'tcl/tk and IDLE' option enabled."
    ) from exc

import numpy as np
from PIL import Image, ImageTk, ImageDraw

from avcs import ScreenEye, CortexState
from avcs.human_eye import HumanEye, HumanEyeState


class AVCSDesktopApp:
    """Native Computer Desktop GUI for AVCS."""

    def __init__(
        self,
        root: tk.Tk,
        fps: int = 25,
        source: Optional[Any] = None,
        allow_capture: bool = False,
    ) -> None:
        self.root = root
        self.root.title("AVCS // Artificial Visual Cortex - Desktop Station")
        self.root.geometry("1180x760")
        self.root.minsize(960, 640)
        self.root.configure(bg="#0d1117")

        self.fps = fps
        self.allow_capture = bool(allow_capture)
        try:
            self.eye = ScreenEye(fps=fps, source=source)
        except Exception:
            if source is None:
                from forge.vision.capture import SyntheticScreen

                source = SyntheticScreen()
                self.eye = ScreenEye(fps=fps, source=source)
            else:
                raise
        self.human_eye = HumanEye(source=self.eye.source, threaded_capture=True if source is None else False)

        # State tracking
        self.running = True
        self.show_fovea = True
        self.show_affordances = True
        self.show_motion = True
        self.show_pursuit = True
        self.show_trajectory = True
        self.always_on_top = False
        self.current_target_gaze: Optional[Tuple[float, float]] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_state: Optional[CortexState] = None
        self.latest_human_state: Optional[HumanEyeState] = None
        self._canvas_img_id: Optional[int] = None
        self._last_drawn_image: Optional[Image.Image] = None

        self._build_ui()
        self._setup_hotkeys()
        self._start_perception_thread()
        self._schedule_gui_refresh()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Delay capture exclusion until DWM has fully registered the window.
        # Calling SetWindowDisplayAffinity before the compositor owns the HWND
        # silently succeeds but has no effect, leaving the HUD visible to
        # BitBlt/mss and causing the recursive mirror feedback loop.
        self.root.after(300, self._apply_capture_exclusion)
        self._wda_retry_count = 0

    def set_capture_affinity(self, allow_capture: bool) -> bool:
        """Sets the Win32 window display capture affinity.

        When allow_capture is False (default), the window is excluded from OS screen
        captures via WDA_EXCLUDEFROMCAPTURE (0x11) to eliminate recursive mirror feedback.
        When allow_capture is True, exclusion is removed via WDA_NONE (0x00), allowing
        Windows screenshot tools (Win+Shift+S, Snipping Tool, OBS, ShareX) to capture the window.
        """
        self.allow_capture = bool(allow_capture)
        if hasattr(self, "var_capture"):
            self.var_capture.set(self.allow_capture)
        if hasattr(self, "status_capture_badge"):
            if self.allow_capture:
                self.status_capture_badge.config(text="CAPTURE: ALLOWED [T]", fg="#3fb950")
            else:
                self.status_capture_badge.config(text="CAPTURE: EXCLUDED [T]", fg="#8b949e")

        if sys.platform == "win32":
            try:
                import ctypes
                self.root.update_idletasks()
                hwnd = self._resolve_top_hwnd()

                u32 = ctypes.windll.user32
                if hasattr(u32, "SetWindowDisplayAffinity"):
                    u32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                    u32.SetWindowDisplayAffinity.restype = ctypes.c_bool
                    # WDA_NONE = 0x00000000; WDA_EXCLUDEFROMCAPTURE = 0x00000011 (Windows 10 2004+)
                    desired = 0x00000000 if self.allow_capture else 0x00000011
                    res = u32.SetWindowDisplayAffinity(hwnd, desired)
                    if res:
                        # Verify the affinity was actually applied by reading it back.
                        u32.GetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
                        u32.GetWindowDisplayAffinity.restype = ctypes.c_bool
                        actual = ctypes.c_uint32(0)
                        if u32.GetWindowDisplayAffinity(hwnd, ctypes.byref(actual)):
                            self._wda_active = (actual.value == 0x00000011)
                            return actual.value == desired
                        self._wda_active = not self.allow_capture
                    return bool(res)
            except Exception:
                pass
        self._wda_active = False
        return False

    def _resolve_top_hwnd(self) -> int:
        """Resolves the top-level Win32 HWND for this Tk root window."""
        import ctypes
        hwnd = 0
        try:
            raw_frame = self.root.wm_frame()
            hwnd = int(str(raw_frame), 0) if str(raw_frame).startswith(("0x", "0X")) else int(str(raw_frame))
        except Exception:
            pass
        if not hwnd:
            hwnd = int(self.root.winfo_id())
            u32 = ctypes.windll.user32
            u32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            u32.GetAncestor.restype = ctypes.c_void_p
            top_hwnd = u32.GetAncestor(hwnd, 2)
            if top_hwnd:
                hwnd = top_hwnd
        return hwnd

    def toggle_capture_affinity(self, event: Optional[tk.Event] = None) -> bool:
        """Toggles between capture-allowed and capture-excluded."""
        return self.set_capture_affinity(not self.allow_capture)

    def _apply_capture_exclusion(self) -> None:
        """Initializes OS screen capture affinity for this HUD window.

        On first call after the DWM delay, retries up to 3 times at 500ms
        intervals if SetWindowDisplayAffinity does not stick.  This handles
        the race where the compositor has not yet fully registered the HWND.
        """
        success = self.set_capture_affinity(self.allow_capture)
        retry = getattr(self, "_wda_retry_count", 0)
        if not success and not self.allow_capture and retry < 3:
            self._wda_retry_count = retry + 1
            self.root.after(500, self._apply_capture_exclusion)

    def _build_ui(self) -> None:
        # Top Header Bar
        header = tk.Frame(self.root, bg="#161b22", height=50, padx=16, pady=10)
        header.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="PARTH LABS // AVCS DESKTOP CORTEX HUD",
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

        self.status_capture_badge = tk.Label(
            header,
            text="CAPTURE: ALLOWED [T]" if self.allow_capture else "CAPTURE: EXCLUDED [T]",
            font=("Segoe UI", 9, "bold"),
            fg="#3fb950" if self.allow_capture else "#8b949e",
            bg="#161b22",
            padx=8,
        )
        self.status_capture_badge.pack(side=tk.RIGHT)

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
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Double-Button-1>", self.on_canvas_double_click)

        # Right Column: Controls & Telemetry
        right_frame = tk.Frame(main_body, bg="#161b22", width=360, highlightbackground="#30363d", highlightthickness=1, padx=14, pady=14)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        # Section 1: Natural Language Gaze Grounding
        sec1 = tk.LabelFrame(
            right_frame,
            text=" 1. Language & Gaze Grounding ",
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

        self.btn_snap_affordance = tk.Button(
            btn_box,
            text="Snap Affordance",
            font=("Segoe UI", 9, "bold"),
            bg="#1f6feb",
            fg="#ffffff",
            activebackground="#388bfd",
            relief=tk.FLAT,
            command=self.on_snap_affordance_clicked,
            padx=8,
            pady=4,
        )
        self.btn_snap_affordance.pack(side=tk.RIGHT, fill=tk.X, expand=True)

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
            state=tk.NORMAL,
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
            state=tk.NORMAL,
            command=lambda: self.on_steer_clicked(click=True),
            pady=4,
        )
        self.btn_steer_click.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        self.motor_status_lbl = tk.Label(
            sec2,
            text="Biological motor servoing ready. Click stream or target to steer.",
            font=("Segoe UI", 8),
            fg="#3fb950",
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
        self.lbl_latency = self._make_stat_row(sec3, "Cortical Latency:", "7.0 ms", "#3fb950")
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

        self.var_motion = tk.BooleanVar(value=True)
        tk.Checkbutton(
            sec4,
            text="Show Optical Motion Vectors",
            variable=self.var_motion,
            fg="#c9d1d9",
            bg="#161b22",
            selectcolor="#0d1117",
            activebackground="#161b22",
            command=self._on_overlay_toggle,
        ).pack(anchor=tk.W)

        self.var_pursuit = tk.BooleanVar(value=True)
        tk.Checkbutton(
            sec4,
            text="Show Smooth Pursuit Lock",
            variable=self.var_pursuit,
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

        self.var_capture = tk.BooleanVar(value=self.allow_capture)
        self.chk_capture = tk.Checkbutton(
            sec4,
            text="Allow Screen Capture (Snipping Tool / OBS) [T / F11]",
            variable=self.var_capture,
            fg="#c9d1d9",
            bg="#161b22",
            selectcolor="#0d1117",
            activebackground="#161b22",
            command=self._on_capture_toggle,
        )
        self.chk_capture.pack(anchor=tk.W)

        btn_cap_box = tk.Frame(sec4, bg="#161b22")
        btn_cap_box.pack(fill=tk.X, pady=(6, 2))

        self.btn_toggle_capture = tk.Button(
            btn_cap_box,
            text="Toggle Capture (T / F11)",
            font=("Segoe UI", 8, "bold"),
            bg="#21262d",
            fg="#c9d1d9",
            activebackground="#30363d",
            relief=tk.FLAT,
            command=self.toggle_capture_affinity,
            padx=4,
            pady=3,
        )
        self.btn_toggle_capture.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_save_snapshot = tk.Button(
            btn_cap_box,
            text="Save Screenshot (Ctrl+S / P)",
            font=("Segoe UI", 8, "bold"),
            bg="#238636",
            fg="#ffffff",
            activebackground="#2ea043",
            relief=tk.FLAT,
            command=self.save_snapshot,
            padx=4,
            pady=3,
        )
        self.btn_save_snapshot.pack(side=tk.RIGHT, fill=tk.X, expand=True)

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
        self.show_motion = self.var_motion.get()
        self.show_pursuit = self.var_pursuit.get()

    def _on_top_toggle(self) -> None:
        self.always_on_top = self.var_top.get()
        self.root.attributes("-topmost", self.always_on_top)

    def _on_capture_toggle(self) -> None:
        self.set_capture_affinity(self.var_capture.get())

    def _setup_hotkeys(self) -> None:
        """Registers hotkeys for capture toggle and saving snapshots."""
        # Capture display affinity toggle hotkeys (T, F11)
        self.root.bind("<F11>", lambda e: self.toggle_capture_affinity())
        self.root.bind("<Key-t>", self._on_key_toggle_capture)
        self.root.bind("<Key-T>", self._on_key_toggle_capture)

        # Snapshot / screenshot hotkeys (Ctrl+S, F12, PrintScreen, P)
        self.root.bind("<Control-s>", lambda e: self.save_snapshot())
        self.root.bind("<Control-S>", lambda e: self.save_snapshot())
        self.root.bind("<F12>", lambda e: self.save_snapshot())
        self.root.bind("<Print>", lambda e: self.save_snapshot())
        self.root.bind("<Key-p>", self._on_key_save_snapshot)
        self.root.bind("<Key-P>", self._on_key_save_snapshot)

    def _on_key_toggle_capture(self, event: tk.Event) -> None:
        """Key handler to avoid triggering when typing in input boxes."""
        if isinstance(event.widget, (tk.Entry, tk.Text)):
            return
        self.toggle_capture_affinity()

    def _on_key_save_snapshot(self, event: tk.Event) -> None:
        """Key handler to avoid triggering when typing in input boxes."""
        if isinstance(event.widget, (tk.Entry, tk.Text)):
            return
        self.save_snapshot()

    def save_snapshot(
        self,
        path: Optional[Union[str, Path, tk.Event]] = None,
        event: Optional[tk.Event] = None,
    ) -> Optional[str]:
        """Saves a snapshot of the current perception visual frame to disk."""
        if isinstance(path, tk.Event):
            event = path
            path = None

        target_img = getattr(self, "_last_drawn_image", None)
        if target_img is None and self.latest_frame is not None:
            target_img = Image.fromarray(self.latest_frame)

        if target_img is None:
            if hasattr(self, "motor_status_lbl"):
                self.motor_status_lbl.config(
                    text="No visual frame available yet to save snapshot.",
                    fg="#d29922",
                )
            return None

        if path is not None:
            filepath = Path(path)
            filepath.parent.mkdir(parents=True, exist_ok=True)
        else:
            snap_dir = Path("runs/screenshots")
            snap_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"avcs_hud_{timestamp}.png"
            filepath = snap_dir / filename

        try:
            target_img.save(filepath)
            msg = f"Screenshot saved: {filepath.name}"
            if hasattr(self, "motor_status_lbl"):
                self.motor_status_lbl.config(text=msg, fg="#3fb950")
            return str(filepath)
        except Exception as e:
            if hasattr(self, "motor_status_lbl"):
                self.motor_status_lbl.config(text=f"Failed to save screenshot: {e}", fg="#f85149")
        return None

    def on_canvas_click(self, event: tk.Event) -> None:
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        nx = float(np.clip(event.x / cw, 0.0, 1.0))
        ny = float(np.clip(event.y / ch, 0.0, 1.0))
        self.current_target_gaze = (nx, ny)
        try:
            self.human_eye.inspect_fovea(centroid=(nx, ny))
        except Exception:
            pass
        self.btn_steer.config(state=tk.NORMAL)
        self.btn_steer_click.config(state=tk.NORMAL)
        self.gaze_pos_lbl.config(text=f"Gaze Fixation: ({nx:.2f}, {ny:.2f}) [TARGET]")
        self.motor_status_lbl.config(
            text=f"Target locked at ({nx:.3f}, {ny:.3f}). Hand ready.",
            fg="#3fb950",
        )

    def on_canvas_double_click(self, event: tk.Event) -> None:
        self.on_canvas_click(event)
        self.on_steer_clicked(click=False)

    def on_snap_affordance_clicked(self) -> None:
        if self.latest_state and self.latest_state.action_hotspots:
            best = max(self.latest_state.action_hotspots, key=lambda a: a[3])
            target = (best[0], best[1])
            self.current_target_gaze = target
            try:
                self.human_eye.inspect_fovea(centroid=target)
            except Exception:
                pass
            self.gaze_pos_lbl.config(
                text=f"Gaze Fixation: ({target[0]:.2f}, {target[1]:.2f}) [{best[2]}]"
            )
            self.btn_steer.config(state=tk.NORMAL)
            self.btn_steer_click.config(state=tk.NORMAL)
            self.motor_status_lbl.config(
                text=f"Snapped to affordance '{best[2]}' at ({target[0]:.2f}, {target[1]:.2f}). Ready.",
                fg="#3fb950",
            )
        else:
            self.motor_status_lbl.config(
                text="No dorsal affordances currently detected on screen.",
                fg="#d29922",
            )

    def on_look_at_clicked(self) -> None:
        query = self.query_entry.get().strip()
        if not query:
            return
        # Execute language grounding via HumanEye
        try:
            res = self.human_eye.look_at(query)
            target = res.target_coord
            self.current_target_gaze = target
            insp = self.human_eye._last_foveal_inspection
            foveal_str = f" [{insp.element_kind}]" if insp else ""
            self.gaze_pos_lbl.config(
                text=f"Gaze Fixation: ({target[0]:.2f}, {target[1]:.2f}){foveal_str}"
            )
            self.btn_steer.config(state=tk.NORMAL)
            self.btn_steer_click.config(state=tk.NORMAL)
            self.motor_status_lbl.config(
                text=f"Target locked at ({target[0]:.2f}, {target[1]:.2f}). Ready to steer.",
                fg="#3fb950",
            )
        except Exception as e:
            try:
                target = self.eye.look_at(query)
                self.current_target_gaze = target
                self.gaze_pos_lbl.config(text=f"Gaze Fixation: ({target[0]:.2f}, {target[1]:.2f})")
                self.btn_steer.config(state=tk.NORMAL)
                self.btn_steer_click.config(state=tk.NORMAL)
                self.motor_status_lbl.config(
                    text=f"Target locked at ({target[0]:.2f}, {target[1]:.2f}). Ready to steer.",
                    fg="#3fb950",
                )
            except Exception:
                messagebox.showerror("Grounding Error", str(e))

    def on_steer_clicked(self, click: bool = False) -> None:
        target = self.current_target_gaze
        if target is None:
            if self.latest_state and self.latest_state.action_hotspots:
                best = max(self.latest_state.action_hotspots, key=lambda a: a[3])
                target = (best[0], best[1])
            else:
                target = (0.5, 0.5)
            self.current_target_gaze = target

        action_name = "Steer & Click" if click else "Steer Cursor"
        if getattr(self, "_servo_active", False):
            self.motor_status_lbl.config(
                text="Hand movement already in progress. Please wait...",
                fg="#d29922",
            )
            return
        self._servo_active = True
        self.motor_status_lbl.config(
            text=f"Biological hand servoing to ({target[0]:.2f}, {target[1]:.2f})...",
            fg="#58a6ff",
        )

        def _worker():
            try:
                from forge.vision.cortex.motor import get_default_driver
                from forge.vision.capture import SyntheticScreen
                from avcs.servoing import NeuromuscularServoingController
                force_virt = False
                if self.eye.source is not None and isinstance(self.eye.source, SyntheticScreen):
                    force_virt = True
                driver = get_default_driver(force_virtual=force_virt)
                controller = NeuromuscularServoingController(driver=driver)
                res = controller.servo_to(target=target, realtime=True)
                if click:
                    pre_click = self.human_eye.observe()
                    driver.click(button="left")
                    try:
                        from avcs.verifier import VisualRippleVerifier
                        verifier = VisualRippleVerifier(eye=self.human_eye, driver=driver)
                        v_res = verifier.verify_action(
                            action_name="click",
                            target=target,
                            pre_state=pre_click,
                        )
                        status_str = f"Clicked ({v_res.status.name}, ripple={v_res.optical_ripple_energy:.3f})"
                    except Exception as exc:
                        status_str = f"Clicked target (verify notice: {exc})."
                else:
                    status_str = f"Docked at ({res.final_cursor[0]:.2f}, {res.final_cursor[1]:.2f}) in {res.duration_ms:.0f}ms."

                self.root.after(0, lambda: self.motor_status_lbl.config(
                    text=f"Hand complete: {status_str}",
                    fg="#3fb950",
                ))
            except Exception as exc:
                self.root.after(0, lambda: self.motor_status_lbl.config(
                    text=f"Motor alert: {exc}",
                    fg="#f85149",
                ))
            finally:
                self._servo_active = False

        threading.Thread(target=_worker, daemon=True).start()

    def _start_perception_thread(self) -> None:
        def _loop():
            tick_dt = 1.0 / self.fps
            while self.running:
                t0 = time.time()
                try:
                    frame = self.human_eye.source.grab() if hasattr(self.human_eye, "source") and hasattr(self.human_eye.source, "grab") else self.eye.source.grab()
                    human_state = self.human_eye.observe(frame=frame)
                    state = CortexState(
                        visual_latent=np.zeros(64, dtype=np.float32),
                        primary_category=human_state.primary_category,
                        category_confidence=human_state.category_confidence,
                        text_density=0.0,
                        stability_state=human_state.stability,
                        surprise_score=human_state.surprise_score,
                        fixation=human_state.gaze_fixation,
                        motor_flow=human_state.motor_flow,
                        contact_acquired=human_state.contact_acquired,
                        action_hotspots=human_state.affordance_hotspots,
                        inference_ms=human_state.latency_ms,
                    )

                    self.latest_frame = frame
                    self.latest_state = state
                    self.latest_human_state = human_state
                except Exception:
                    pass

                dt = time.time() - t0
                sleep_sec = tick_dt - dt
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

        self._perception_thread = threading.Thread(target=_loop, daemon=True)
        self._perception_thread.start()

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

        # Anti-recursion shield: When allow_capture is enabled (or before WDA kicks in),
        # mask out the HUD window's footprint on the screen capture to eliminate the
        # infinite mirror / Droste recursion feedback loop.
        source_frame = frame
        hud_box_screen: Optional[Tuple[int, int, int, int]] = None
        if self.allow_capture or not getattr(self, "_wda_active", False):
            try:
                fh, fw = frame.shape[:2]
                sw = max(1, self.root.winfo_screenwidth())
                sh = max(1, self.root.winfo_screenheight())
                scale_x = fw / sw
                scale_y = fh / sh

                rx = int(self.root.winfo_rootx() * scale_x)
                ry = int(self.root.winfo_rooty() * scale_y)
                rw = int(self.root.winfo_width() * scale_x)
                rh = int(self.root.winfo_height() * scale_y)

                x1 = max(0, min(fw, rx))
                y1 = max(0, min(fh, ry))
                x2 = max(0, min(fw, rx + rw))
                y2 = max(0, min(fh, ry + rh))

                if x2 > x1 and y2 > y1:
                    source_frame = frame.copy()
                    source_frame[y1:y2, x1:x2] = (13, 17, 23)
                    hud_box_screen = (x1, y1, x2, y2)
            except Exception:
                source_frame = frame

        img = Image.fromarray(source_frame).resize((cw, ch), Image.BILINEAR)
        draw = ImageDraw.Draw(img)

        # Draw anti-recursion shield badge on HUD's own footprint in canvas
        if hud_box_screen is not None and (self.allow_capture or not getattr(self, "_wda_active", False)):
            try:
                fh, fw = frame.shape[:2]
                hx1, hy1, hx2, hy2 = hud_box_screen
                bx1 = int((hx1 / fw) * cw)
                by1 = int((hy1 / fh) * ch)
                bx2 = int((hx2 / fw) * cw)
                by2 = int((hy2 / fh) * ch)
                draw.rectangle([bx1, by1, bx2, by2], fill="#161b22", outline="#58a6ff", width=2)
                draw.text((bx1 + 10, by1 + 10), "AVCS CORTEX HUD // ACTIVE DESKTOP STATION", fill="#58a6ff")
                draw.text((bx1 + 10, by1 + 26), "Anti-Recursion Shield Active", fill="#8b949e")
            except Exception:
                pass

        # 1. Draw Dorsal Affordance Hotspots (Dual-Stroke for high contrast on light/dark themes)
        if self.show_affordances and state.action_hotspots:
            for nx, ny, act_type, conf in state.action_hotspots:
                px = int(nx * cw)
                py = int(ny * ch)
                radius = 16
                color = "#00ff88" if act_type == "clickable" else "#ffaa00"

                # Outer black shadow halo for contrast against white backgrounds
                draw.rectangle([px - radius - 1, py - radius - 1, px + radius + 1, py + radius + 1], outline="#000000", width=3)
                draw.rectangle([px - radius, py - radius, px + radius, py + radius], outline=color, width=2)
                # Label background shadow
                draw.text((px - radius + 1, py - radius - 13), f"{act_type} {conf:.0%}", fill="#000000")
                draw.text((px - radius, py - radius - 14), f"{act_type} {conf:.0%}", fill=color)

        # 2. Draw Foveal Gaze Reticle Crosshair (Dual-Stroke Neon + Black Halo)
        if self.show_fovea:
            fx, fy = self.current_target_gaze or state.fixation
            cx = int(fx * cw)
            cy = int(fy * ch)
            cross_size = 20
            reticle_color = "#00ffff"  # Neon cyan

            # Outer targeting circle with shadow
            draw.ellipse([cx - 25, cy - 25, cx + 25, cy + 25], outline="#000000", width=4)
            draw.ellipse([cx - 24, cy - 24, cx + 24, cy + 24], outline=reticle_color, width=2)
            # Center crosshair with shadow
            draw.line([cx - cross_size, cy, cx + cross_size, cy], fill="#000000", width=3)
            draw.line([cx, cy - cross_size, cx, cy + cross_size], fill="#000000", width=3)
            draw.line([cx - cross_size, cy, cx + cross_size, cy], fill=reticle_color, width=1)
            draw.line([cx, cy - cross_size, cx, cy + cross_size], fill=reticle_color, width=1)
            draw.text((cx + 13, cy + 13), "FOVEA", fill="#000000")
            draw.text((cx + 12, cy + 12), "FOVEA", fill=reticle_color)

        # 3. Draw Magnocellular Optical Motion Vectors & Moving Entities
        if self.show_motion and self.latest_human_state and self.latest_human_state.moving_objects:
            for obj in self.latest_human_state.moving_objects:
                ox, oy = int(obj.centroid[0] * cw), int(obj.centroid[1] * ch)
                r = int(max(12, min(50, obj.area * cw * 0.4)))
                # Motion bounding box
                draw.rectangle([ox - r, oy - r, ox + r, oy + r], outline="#00e5ff", width=2)
                # Motion heading arrow
                vx, vy = int(obj.velocity[0] * cw * 0.3), int(obj.velocity[1] * ch * 0.3)
                draw.line([ox, oy, ox + vx, oy + vy], fill="#00ff66", width=3)
                draw.text((ox - r, oy - r - 12), f"ENTITY | v={obj.speed:.2f}", fill="#00ff66")

        # 4. Draw Smooth Pursuit Target Lock
        if self.show_pursuit and self.latest_human_state and self.latest_human_state.pursuit_target:
            tx = int(self.latest_human_state.pursuit_target[0] * cw)
            ty = int(self.latest_human_state.pursuit_target[1] * ch)
            lock_color = "#ff007f"  # Neon magenta
            draw.ellipse([tx - 18, ty - 18, tx + 18, ty + 18], outline=lock_color, width=3)
            draw.line([tx - 10, ty, tx + 10, ty], fill=lock_color, width=2)
            draw.line([tx, ty - 10, tx, ty + 10], fill=lock_color, width=2)
            draw.text((tx + 14, ty - 8), "PURSUIT LOCK", fill=lock_color)

        # 5. Draw Foveal Acuity High-Resolution Inspection Box
        if self.show_fovea and self.human_eye._last_foveal_inspection:
            insp = self.human_eye._last_foveal_inspection
            x1, y1, x2, y2 = insp.bbox
            px1, py1 = int(x1 * cw), int(y1 * ch)
            px2, py2 = int(x2 * cw), int(y2 * ch)
            draw.rectangle([px1, py1, px2, py2], outline="#ffd700", width=2)
            draw.text((px1, max(0, py1 - 13)), f"FOVEA: {insp.element_kind.upper()} ({insp.confidence:.0%})", fill="#ffd700")

        # 6. Warning banner when capture exclusion is off (recursive mirror expected)
        if self.allow_capture:
            banner_y = ch - 28
            draw.rectangle([0, banner_y, cw, ch], fill="#b3200080")
            draw.text(
                (8, banner_y + 6),
                "MIRROR MODE: Press T to exclude HUD from capture",
                fill="#ffffff",
            )

        self._last_drawn_image = img.copy()

        # Convert to PhotoImage and update canvas item without memory leak
        photo = ImageTk.PhotoImage(image=img)
        if self._canvas_img_id is None:
            self._canvas_img_id = self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        else:
            self.canvas.itemconfig(self._canvas_img_id, image=photo)
        self.canvas._current_photo = photo  # prevent Python GC

    def _update_telemetry_labels(self, state: CortexState) -> None:
        self.lbl_category.config(text=state.primary_category)
        stab_color = "#3fb950" if state.stability_state == "stable" else "#d29922"
        self.lbl_stability.config(text=state.stability_state, fg=stab_color)
        self.lbl_surprise.config(text=f"{state.surprise_score:.4f}")

        clicks = len([h for h in (state.action_hotspots or []) if h[2] == "clickable"])
        types = len([h for h in (state.action_hotspots or []) if h[2] == "typeable"])
        self.lbl_hotspots.config(text=f"{clicks} Buttons, {types} Inputs")

        if self.latest_human_state is not None:
            self.lbl_latency.config(text=f"{self.latest_human_state.latency_ms:.1f} ms")
            self.lbl_fps.config(text=f"{self.latest_human_state.fps:.1f} FPS")
        else:
            self.lbl_fps.config(text=f"{self.fps} FPS")

    def on_close(self) -> None:
        self.running = False
        if hasattr(self, "_perception_thread") and self._perception_thread is not None:
            try:
                self._perception_thread.join(timeout=0.5)
            except Exception:
                pass
        try:
            self.eye.close()
        except Exception:
            pass
        try:
            self.human_eye.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


def launch_gui(
    root: Optional[Any] = None,
    fps: int = 25,
    source: Optional[Any] = None,
    mainloop: Optional[bool] = None,
    allow_capture: bool = False,
    **kwargs: Any,
) -> AVCSDesktopApp:
    """Launches the AVCS Desktop Station GUI application."""
    if isinstance(root, bool):
        allow_capture = root
        root = None
    elif isinstance(root, (int, float)):
        fps = int(root)
        root = None
    should_loop = mainloop if mainloop is not None else (root is None)
    if root is None:
        root = tk.Tk()
    app = AVCSDesktopApp(root=root, fps=fps, source=source, allow_capture=allow_capture)
    if should_loop:
        root.mainloop()
    return app


if __name__ == "__main__":
    launch_gui()
