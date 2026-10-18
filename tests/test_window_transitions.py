"""Test suite for Multi-Step Application Launch and Window Transition Tracking in AVCS."""
from __future__ import annotations

import numpy as np
import pytest

from avcs.events import VisualEventStreamer, VisualEventType
from avcs.human_eye import HumanEye, MagnocellularMotionEngine
from avcs.testing.audit_window_transition import DesktopSceneBuilder
from avcs.verifier import VerificationStatus, VisualRippleVerifier
from forge.vision.cortex.motor import VirtualCursorDriver


def test_icon_click_and_local_ripple_verification():
    builder = DesktopSceneBuilder(width=320, height=180)
    desktop, icons = builder.render_desktop_canvas()
    terminal_icon = icons["terminal"]

    clicked = builder.render_icon_click_ripple(desktop, terminal_icon)

    class SequenceCapture:
        def __init__(self, frames):
            self.frames = frames
            self.idx = 0
        def grab(self):
            f = self.frames[min(self.idx, len(self.frames) - 1)]
            self.idx += 1
            return f

    eye = HumanEye(source=SequenceCapture([desktop, clicked, clicked]), threaded_capture=False)
    driver = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye, driver=driver, ripple_radius=0.08)

    pre_state = eye.observe()
    driver.move_to_normalized(terminal_icon[0], terminal_icon[1])
    driver.click(button="left")
    res = verifier.verify_action("click_terminal", terminal_icon, pre_state)

    assert res.is_successful
    assert res.status == VerificationStatus.VERIFIED
    assert res.optical_ripple_energy > 0.015


def test_magnocellular_motion_detects_window_expansion():
    builder = DesktopSceneBuilder(width=320, height=180)
    desktop, icons = builder.render_desktop_canvas()
    terminal_icon = icons["terminal"]

    anim_frames = builder.render_window_launch_animation(
        desktop, terminal_icon, app_type="terminal", steps=4
    )
    engine = MagnocellularMotionEngine(grid_rows=16, grid_cols=24)

    # Base frame
    engine.process(desktop)

    motions = []
    objects_detected = []
    for f in anim_frames:
        mf, mm, objs, surprise = engine.process(f)
        motions.append(mm)
        if objs:
            objects_detected.append(objs[0])

    peak_motion = max(motions)
    assert peak_motion > 0.02
    assert len(objects_detected) > 0
    # Centroid should be within valid bounds
    top_obj = objects_detected[0]
    assert 0.0 <= top_obj.centroid[0] <= 1.0
    assert 0.0 <= top_obj.centroid[1] <= 1.0
    assert top_obj.speed > 0.0


def test_ventral_semantic_category_shift():
    builder = DesktopSceneBuilder(width=320, height=180)
    desktop, icons = builder.render_desktop_canvas()

    eye = HumanEye(source="synthetic", threaded_capture=False)
    st_desktop = eye.observe(desktop)
    assert st_desktop.primary_category == "desktop_canvas"

    terminal_anim = builder.render_window_launch_animation(
        desktop, icons["terminal"], app_type="terminal", steps=5
    )

    categories = [st_desktop.primary_category]
    for f in terminal_anim:
        st = eye.observe(f)
        categories.append(st.primary_category)

    final_cat = categories[-1]
    assert final_cat == "terminal_shell"
    # Transition must have shifted from desktop_canvas to terminal_shell
    assert "desktop_canvas" in categories
    assert "terminal_shell" in categories


def test_visual_event_streamer_lifecycle():
    builder = DesktopSceneBuilder(width=320, height=180)
    desktop, icons = builder.render_desktop_canvas()

    streamer = VisualEventStreamer()
    events = []
    streamer.subscribe(callback=lambda ev: events.append(ev))

    eye = HumanEye(source="synthetic", threaded_capture=False)

    terminal_anim = builder.render_window_launch_animation(
        desktop, icons["terminal"], app_type="terminal", steps=4
    )
    full_seq = [desktop, desktop] + terminal_anim + [terminal_anim[-1]]

    for f in full_seq:
        st = eye.observe(f)
        streamer.process_state(st)

    event_types = {ev.event_type for ev in events}
    assert VisualEventType.SCENE_TRANSITION in event_types
    assert VisualEventType.MOTION_DETECTED in event_types
