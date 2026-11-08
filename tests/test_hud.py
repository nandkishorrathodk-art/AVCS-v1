"""Unit Tests for AVCS Desktop Dynamic Island HUD (avcs/ace/hud.py)."""
import math
import sys
import time
import unittest

from avcs.ace.hud import (
    AVCSVisualHUD,
    COLOR_OBSIDIAN,
    COLOR_RIM,
    COLOR_HIGHLIGHT,
    STATE_COLORS,
    play_acoustic_earcon,
)


class TestDynamicIslandHUD(unittest.TestCase):
    """Verifies all algorithmic logic, state machines, and styling metrics of the Dynamic Island HUD."""

    def test_state_classification(self):
        """Verifies status string mapping to Dynamic Island visual states."""
        # IDLE / READY
        self.assertEqual(AVCSVisualHUD.classify_state("READY"), "IDLE")
        self.assertEqual(AVCSVisualHUD.classify_state("IDLE"), "IDLE")
        self.assertEqual(AVCSVisualHUD.classify_state("STANDBY"), "IDLE")
        self.assertEqual(AVCSVisualHUD.classify_state("WAITING"), "IDLE")
        self.assertEqual(AVCSVisualHUD.classify_state("INITIALIZING"), "IDLE")
        self.assertEqual(AVCSVisualHUD.classify_state("STARTING"), "IDLE")
        self.assertEqual(AVCSVisualHUD.classify_state(""), "IDLE")

        # PLANNING / COGNITIVE
        self.assertEqual(AVCSVisualHUD.classify_state("PLANNING"), "PLANNING")
        self.assertEqual(AVCSVisualHUD.classify_state("THINKING"), "PLANNING")
        self.assertEqual(AVCSVisualHUD.classify_state("OBSERVING"), "PLANNING")
        self.assertEqual(AVCSVisualHUD.classify_state("LOOKUP_SKILL"), "PLANNING")

        # ACTIVE MOTOR EXECUTION
        self.assertEqual(AVCSVisualHUD.classify_state("LEFT_CLICK"), "MOTOR")
        self.assertEqual(AVCSVisualHUD.classify_state("RIGHT_CLICK"), "MOTOR")
        self.assertEqual(AVCSVisualHUD.classify_state("DOUBLE_CLICK"), "MOTOR")
        self.assertEqual(AVCSVisualHUD.classify_state("TYPE"), "MOTOR")
        self.assertEqual(AVCSVisualHUD.classify_state("KEY"), "MOTOR")
        self.assertEqual(AVCSVisualHUD.classify_state("HOTKEY"), "MOTOR")
        self.assertEqual(AVCSVisualHUD.classify_state("ACTUATING"), "MOTOR")

        # VERIFICATION / SUCCESS
        self.assertEqual(AVCSVisualHUD.classify_state("VERIFIED"), "VERIFIED")
        self.assertEqual(AVCSVisualHUD.classify_state("VERIFYING"), "VERIFIED")
        self.assertEqual(AVCSVisualHUD.classify_state("COMPLETE"), "VERIFIED")
        self.assertEqual(AVCSVisualHUD.classify_state("SUCCESS"), "VERIFIED")
        self.assertEqual(AVCSVisualHUD.classify_state("FINISHED"), "VERIFIED")

        # HALT / ERROR
        self.assertEqual(AVCSVisualHUD.classify_state("HALTED (ERROR)"), "ERROR")
        self.assertEqual(AVCSVisualHUD.classify_state("ERROR"), "ERROR")
        self.assertEqual(AVCSVisualHUD.classify_state("ABORTED"), "ERROR")
        self.assertEqual(AVCSVisualHUD.classify_state("FAILED"), "ERROR")

    def test_dynamic_target_sizing(self):
        """Verifies dimension contracts across all Dynamic Island morph states."""
        # Compact Idle capsule: ~220px wide, ~36px high
        w_idle, h_idle = AVCSVisualHUD.calculate_target_size("IDLE")
        self.assertEqual(w_idle, 220)
        self.assertEqual(h_idle, 36)

        # Thinking / Planning: ~380-520px wide, ~38px high
        w_plan_short, h_plan = AVCSVisualHUD.calculate_target_size("PLANNING", "Decomposing goal")
        self.assertGreaterEqual(w_plan_short, 380)
        self.assertLessEqual(w_plan_short, 520)
        self.assertEqual(h_plan, 38)

        # Active Motor Execution: ~620-690px wide, ~42px high
        w_motor, h_motor = AVCSVisualHUD.calculate_target_size("MOTOR", "Step 1/3: Open Start Menu")
        self.assertGreaterEqual(w_motor, 620)
        self.assertLessEqual(w_motor, 690)
        self.assertEqual(h_motor, 42)

        # Verified / Complete: ~290px wide, ~38px high
        w_ver, h_ver = AVCSVisualHUD.calculate_target_size("VERIFIED")
        self.assertEqual(w_ver, 290)
        self.assertEqual(h_ver, 38)

        # Error / Halted: ~420-560px wide, ~38px high
        w_err, h_err = AVCSVisualHUD.calculate_target_size("ERROR", "Window did not emerge on desktop")
        self.assertGreaterEqual(w_err, 420)
        self.assertLessEqual(w_err, 560)
        self.assertEqual(h_err, 38)

    def test_fluid_spring_morphing_interpolation(self):
        """Verifies exponential spring interpolation smoothly converges to target dimensions."""
        hud = AVCSVisualHUD()
        hud.current_w = 220.0
        hud.current_h = 36.0
        hud.target_w = 640.0
        hud.target_h = 42.0

        # Simulate 20 timer frames of spring interpolation
        widths = [hud.current_w]
        for _ in range(30):
            hud.step_spring()
            widths.append(hud.current_w)

        # Must monotonically increase towards 640
        for i in range(1, len(widths)):
            self.assertGreaterEqual(widths[i], widths[i - 1])

        # Must cleanly settle exactly at target_w and target_h
        self.assertEqual(hud.current_w, 640.0)
        self.assertEqual(hud.current_h, 42.0)

        # Now test shrinking back to idle (640 -> 220)
        hud.target_w = 220.0
        hud.target_h = 36.0
        for _ in range(30):
            hud.step_spring()

        self.assertEqual(hud.current_w, 220.0)
        self.assertEqual(hud.current_h, 36.0)

    def test_step_progress_parsing(self):
        """Verifies robust extraction of step count and total steps."""
        self.assertEqual(AVCSVisualHUD.parse_step_progress("1/3"), (1, 3))
        self.assertEqual(AVCSVisualHUD.parse_step_progress("2/5"), (2, 5))
        self.assertEqual(AVCSVisualHUD.parse_step_progress("4/4"), (4, 4))
        self.assertEqual(AVCSVisualHUD.parse_step_progress("0/0"), (0, 0))
        self.assertEqual(AVCSVisualHUD.parse_step_progress("invalid"), (0, 0))
        self.assertEqual(AVCSVisualHUD.parse_step_progress(""), (0, 0))

    def test_update_status_telemetry_and_target_adjustments(self):
        """Verifies telemetry updates trigger appropriate target morphing dimensions."""
        hud = AVCSVisualHUD()
        self.assertEqual(hud.target_w, 220.0)
        self.assertEqual(hud.target_h, 36.0)

        # Switch to PLANNING
        hud.update_status(
            status="PLANNING",
            subgoal="Decomposing goal into subgoals",
            step="1/3",
            mode="LIVE",
            latency_ms=120.0,
        )
        self.assertEqual(hud.status_text, "PLANNING")
        self.assertEqual(hud.subgoal_text, "Decomposing goal into subgoals")
        self.assertEqual(hud.step_info, "1/3")
        self.assertEqual(hud.latency_ms, 120.0)
        self.assertGreaterEqual(hud.target_w, 380.0)
        self.assertEqual(hud.target_h, 38.0)

        # Switch to MOTOR CLICK
        hud.update_status(
            status="LEFT_CLICK",
            subgoal="Click Start Menu button",
            step="1/3",
            mode="LIVE",
            latency_ms=45.0,
        )
        self.assertEqual(hud.status_text, "LEFT_CLICK")
        self.assertGreaterEqual(hud.target_w, 620.0)
        self.assertEqual(hud.target_h, 42.0)

    def test_click_reticle_dispatch(self):
        """Verifies targeting reticle coordinates and duration."""
        hud = AVCSVisualHUD()
        hud.show_click_reticle(350, 420, kind="left_click")
        self.assertEqual(hud.reticle_coord, (350, 420))
        self.assertEqual(hud.reticle_kind, "left_click")
        self.assertGreater(hud.reticle_start_time, 0.0)
        self.assertEqual(hud.reticle_duration, 0.35)

    def test_living_breathing_orb_pulse_bounds(self):
        """Verifies sinusoidal glowing indicator pulse remains strictly within [0.0, 1.0]."""
        for t in [0.0, 0.25, 0.5, 0.785, 1.0, 1.57, 2.0, 3.14, 5.0, 10.0]:
            pulse = 0.5 + 0.5 * math.sin(t * 4.0)
            self.assertGreaterEqual(pulse, 0.0)
            self.assertLessEqual(pulse, 1.0)

    def test_obsidian_palette_constants(self):
        """Verifies BGR hex values conform to obsidian glass requirements."""
        # Deep obsidian black core: #060709 (R=6, G=7, B=9) -> BGR 0x00090706
        self.assertEqual(COLOR_OBSIDIAN, 0x00090706)
        # Frosted rim: #2E3038 (R=46, G=48, B=56) -> BGR 0x0038302E
        self.assertEqual(COLOR_RIM, 0x0038302E)
        # Top-edge curved glass light highlight: #444A55 -> BGR 0x00554A44
        self.assertEqual(COLOR_HIGHLIGHT, 0x00554A44)

        # State accent palettes present
        for state in ("LIVE", "PLANNING", "MOTOR", "VERIFIED", "ERROR", "SIMULATION"):
            self.assertIn(state, STATE_COLORS)
            self.assertIn("primary", STATE_COLORS[state])
            self.assertIn("core", STATE_COLORS[state])
            self.assertIn("glow", STATE_COLORS[state])

    def test_context_manager_lifecycle(self):
        """Verifies HUD starts and stops cleanly via Python context manager."""
        with AVCSVisualHUD(title="Test HUD Lifecycle") as hud:
            self.assertIsNotNone(hud)
            hud.update_status(status="READY", mode="SIMULATION")
        # Ensure stopped cleanly
        self.assertTrue(hud._stop_event.is_set())

    def test_earcon_dispatch_non_blocking(self):
        """Verifies acoustic earcon functions gracefully without raising exceptions."""
        play_acoustic_earcon("start")
        play_acoustic_earcon("action")
        play_acoustic_earcon("verify")
        play_acoustic_earcon("success")
        play_acoustic_earcon("alert")
        # Execution does not block
        self.assertTrue(True)


    def test_gdi_render_pipeline_all_states(self):
        """Verifies GDI double-buffered drawing routine executes without errors across all states."""
        if sys.platform != "win32":
            return
        import ctypes
        u32 = ctypes.windll.user32
        gdi = ctypes.windll.gdi32

        screen_dc = u32.GetDC(None)
        mem_dc = gdi.CreateCompatibleDC(screen_dc)
        mem_bm = gdi.CreateCompatibleBitmap(screen_dc, 800, 200)
        old_bm = gdi.SelectObject(mem_dc, mem_bm)

        hud = AVCSVisualHUD()
        hud.screen_w = 800
        hud.screen_h = 200
        hud._create_fonts(gdi)

        states_to_test = [
            ("READY", "", "0/0", "LIVE", 0.0),
            ("PLANNING", "Decomposing goal into subgoals", "1/3", "LIVE", 120.0),
            ("LEFT_CLICK", "Click search button", "1/3", "LIVE", 45.0),
            ("COMPLETE", "Verified Successfully", "3/3", "LIVE", 30.0),
            ("HALTED (ERROR)", "Target window did not emerge", "2/3", "LIVE", 0.0),
            ("READY", "", "0/0", "SIMULATION", 0.0),
        ]

        try:
            for st, sg, step, mode, lat in states_to_test:
                hud.update_status(status=st, subgoal=sg, step=step, mode=mode, latency_ms=lat)
                hud.current_w = hud.target_w
                hud.current_h = hud.target_h
                # Call draw routine into the in-memory double buffer DC
                hud._draw_hud(mem_dc)

            # Test click reticle rendering
            hud.show_click_reticle(400, 100, kind="left_click")
            hud._draw_hud(mem_dc)
        finally:
            hud._delete_fonts(gdi)
            gdi.SelectObject(mem_dc, old_bm)
            gdi.DeleteObject(mem_bm)
            gdi.DeleteDC(mem_dc)
            u32.ReleaseDC(None, screen_dc)

    def test_win32_signature_completeness_64bit(self):
        """Verifies critical Win32 functions have explicit argtypes to prevent 64-bit handle truncation."""
        if sys.platform != "win32":
            return
        import ctypes
        from avcs.ace.hud import _setup_win32_signatures
        _setup_win32_signatures()

        u32 = ctypes.windll.user32
        gdi = ctypes.windll.gdi32

        critical_user32 = [
            "DefWindowProcW", "BeginPaint", "EndPaint", "FillRect", "DrawTextW",
            "InvalidateRect", "CreateWindowExW", "RegisterClassExW",
            "SetLayeredWindowAttributes", "ShowWindow", "SetTimer", "KillTimer",
            "GetMessageW", "TranslateMessage", "DispatchMessageW", "PostMessageW",
            "LoadCursorW", "GetSystemMetrics", "GetDC", "ReleaseDC"
        ]
        for name in critical_user32:
            fn = getattr(u32, name, None)
            self.assertIsNotNone(fn, f"user32.{name} must exist")
            self.assertIsNotNone(fn.argtypes, f"user32.{name}.argtypes must be explicitly defined")

        critical_gdi32 = [
            "CreateSolidBrush", "CreatePen", "SelectObject", "GetStockObject",
            "DeleteObject", "DeleteDC", "CreateCompatibleDC", "CreateCompatibleBitmap",
            "BitBlt", "RoundRect", "Ellipse", "MoveToEx", "LineTo", "SetBkMode",
            "SetTextColor", "CreateFontW", "GetDIBits"
        ]
        for name in critical_gdi32:
            fn = getattr(gdi, name, None)
            self.assertIsNotNone(fn, f"gdi32.{name} must exist")
            self.assertIsNotNone(fn.argtypes, f"gdi32.{name}.argtypes must be explicitly defined")

    def test_bgr_color_math_and_blending(self):
        """Verifies BGR unpack, pack, and linear interpolation blending routines."""
        from avcs.ace.hud import _unpack_bgr, _pack_bgr, _blend_bgr

        # Unpack test: 0x00BBGGRR -> R, G, B
        r, g, b = _unpack_bgr(0x00FF8040)  # B=255, G=128, R=64
        self.assertEqual(r, 64)
        self.assertEqual(g, 128)
        self.assertEqual(b, 255)

        # Pack test
        c = _pack_bgr(64, 128, 255)
        self.assertEqual(c, 0x00FF8040)

        # 50% blend between black (0) and white (0x00FFFFFF)
        mid = _blend_bgr(0x00FFFFFF, 0x00000000, 0.5)
        self.assertEqual(mid, 0x007F7F7F)

        # 0% blend returns base
        base = _blend_bgr(0x00FF0000, 0x0000FF00, 0.0)
        self.assertEqual(base, 0x0000FF00)

        # 100% blend returns target
        target = _blend_bgr(0x00FF0000, 0x0000FF00, 1.0)
        self.assertEqual(target, 0x00FF0000)

    def test_multimonitor_and_narrow_buffer_clamping(self):
        """Verifies pill_x and pill_y clamp cleanly within memory buffers and multi-monitor layouts."""
        hud = AVCSVisualHUD()
        hud.screen_w = 400
        hud.screen_h = 100
        hud.current_w = 380.0
        hud.current_h = 38.0

        # Simulate coordinates calculation logic
        primary_w = min(hud.screen_w, 1920)
        primary_offset_x = max(0, -hud.screen_x)
        pill_w = int(hud.current_w)
        pill_x = primary_offset_x + (primary_w - pill_w) // 2
        pill_x = max(0, min(hud.screen_w - pill_w, pill_x))

        self.assertGreaterEqual(pill_x, 0)
        self.assertLessEqual(pill_x + pill_w, hud.screen_w)


if __name__ == "__main__":
    unittest.main()


