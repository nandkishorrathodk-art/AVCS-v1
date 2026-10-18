import ctypes
import json
from ctypes import wintypes

import numpy as np
import pytest

from avcs.training.demo_recorder import (
    WindowInfo,
    WindowsDemoRecorder,
    build_absence_record,
    build_click_record,
    crop_window_frame,
    find_exact_window_title,
    main,
    normalized_click,
    physical_window_bounds,
    render_click_audit,
)
from forge.vision.cortex.motor.win32 import HARDWAREINPUT


def test_find_exact_window_title_rejects_partial_and_duplicate_matches():
    windows = [
        WindowInfo(1, "AVCS_DEMO_WINDOW", (10, 20, 110, 120)),
        WindowInfo(2, "AVCS_DEMO_WINDOW - Settings", (0, 0, 50, 50)),
    ]

    assert find_exact_window_title("AVCS_DEMO_WINDOW", windows).hwnd == 1
    with pytest.raises(RuntimeError, match="found 0"):
        find_exact_window_title("DEMO_WINDOW", windows)
    with pytest.raises(RuntimeError, match="found 2"):
        find_exact_window_title(
            "AVCS_DEMO_WINDOW",
            [windows[0], WindowInfo(3, "AVCS_DEMO_WINDOW", (0, 0, 20, 20))],
        )


def test_crop_window_frame_respects_virtual_screen_origin():
    frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    crop, screen_rect = crop_window_frame(frame, (-2, 11, 2, 14), (-3, 10))

    assert crop.shape == (3, 4, 3)
    np.testing.assert_array_equal(crop, frame[1:4, 1:5])
    assert screen_rect == (-2, 11, 2, 14)


def test_crop_window_frame_rejects_invalid_or_offscreen_frames():
    with pytest.raises(ValueError, match="RGB frame"):
        crop_window_frame(np.zeros((10, 10), dtype=np.uint8), (0, 0, 5, 5))
    with pytest.raises(ValueError, match="outside"):
        crop_window_frame(np.zeros((10, 10, 3), dtype=np.uint8), (20, 20, 30, 30))


def test_recorder_captures_clipped_window_region_directly():
    class Capture:
        virtual_left = 0
        virtual_top = 0
        virtual_width = 100
        virtual_height = 80
        requested = None

        def get_geometry(self):
            return self

        def grab_region(self, rect):
            self.requested = rect
            return np.zeros((rect[3] - rect[1], rect[2] - rect[0], 3), dtype=np.uint8)

        def grab(self):
            raise AssertionError("region-capable capture must not grab the full desktop")

    recorder = object.__new__(WindowsDemoRecorder)
    recorder.capture = Capture()

    frame, origin = recorder._capture_rgb((-10, 5, 30, 45))

    assert recorder.capture.requested == (0, 5, 30, 45)
    assert frame.shape == (40, 30, 3)
    assert origin == (0, 5)


def test_normalized_click_uses_exact_window_bounds_and_rejects_outside_points():
    rect = (100, 200, 300, 400)

    assert normalized_click((100, 200), rect) == (0.0, 0.0)
    assert normalized_click((200, 300), rect) == (0.5, 0.5)
    assert normalized_click((300, 300), rect) is None
    assert normalized_click((99, 300), rect) is None


def test_physical_window_bounds_use_dwm_pixel_coordinates():
    def get_bounds(hwnd, attribute, output, size):
        assert hwnd == 42
        assert attribute == 9
        assert size == ctypes.sizeof(wintypes.RECT)
        rect = ctypes.cast(output, ctypes.POINTER(wintypes.RECT)).contents
        rect.left, rect.top, rect.right, rect.bottom = 152, 152, 1592, 970
        return 0

    assert physical_window_bounds(42, get_bounds) == (152, 152, 1592, 970)


def test_physical_window_bounds_fails_closed_on_dwm_errors():
    with pytest.raises(OSError, match="DwmGetWindowAttribute failed"):
        physical_window_bounds(42, lambda *_args: -1)


def test_click_audit_marks_a_copy_and_preserves_source_pixels():
    frame = np.zeros((60, 100, 3), dtype=np.uint8)
    audit = render_click_audit(frame, (0.25, 0.5))

    assert frame.max() == 0
    assert audit.size == (100, 60)
    assert audit.getpixel((25, 30)) == (255, 40, 40)
    assert audit.getpixel((0, 0)) == (0, 0, 0)


def test_click_audit_rejects_invalid_point():
    with pytest.raises(ValueError, match="point_norm"):
        render_click_audit(np.zeros((20, 20, 3), dtype=np.uint8), (1.0, 0.5))


def test_click_record_contains_no_captured_keystrokes_or_text():
    record = build_click_record(
        task="Open the test menu",
        target_label="File menu",
        frame_path="frames/sample.png",
        point_norm=(0.25, 0.75),
        recorded_at_utc="2026-10-11T00:00:00+00:00",
    )

    assert record["action"] == {
        "type": "mouse_click",
        "button": "left",
        "point_norm": [0.25, 0.75],
    }
    assert "typed_text" not in record
    assert "key" not in record["action"]


def test_absence_record_has_no_fabricated_click_coordinate():
    record = build_absence_record(
        task="Find a missing Save button",
        target_label="Save button",
        frame_path="frames/sample.png",
        recorded_at_utc="2026-10-11T00:00:00+00:00",
    )

    assert record["action"] == {"type": "target_absent"}
    assert record["target_label"] == "Save button"
    assert "point_norm" not in record["action"]


def test_calibration_stream_keeps_pixels_in_memory_and_stops_if_window_closes(tmp_path):
    class Capture:
        virtual_left = 0
        virtual_top = 0
        virtual_width = 100
        virtual_height = 100

        def __init__(self):
            self.closed = False
            self.captured_rects = []

        def open(self):
            pass

        def get_geometry(self):
            return self

        def grab_region(self, rect):
            self.captured_rects.append(rect)
            return np.zeros((rect[3] - rect[1], rect[2] - rect[0], 3), dtype=np.uint8)

        def close(self):
            self.closed = True

    class Recorder(WindowsDemoRecorder):
        def _selected_window(self):
            self.window_checks = getattr(self, "window_checks", 0) + 1
            if self.window_checks > 3:
                raise RuntimeError(
                    "Expected exactly one visible window titled 'Untitled - Notepad'; found 0."
                )
            return WindowInfo(1, "Untitled - Notepad", (0, 0, 100, 100))

        def _owns_foreground(self, _target_hwnd):
            return True

        def _cursor_position(self):
            return 20, 30

        def _left_button_down(self):
            return False

    class Eye:
        def __init__(self):
            self.frames = []

        def perceive(self, *, frame):
            self.frames.append(frame)
            return {
                "latency_ms": 1.0,
                "weights_loaded": False,
                "is_static": False,
                "motion_magnitude": 0.0,
            }

    events = []
    capture = Capture()
    eye = Eye()
    output_dir = tmp_path / "must-not-be-created"
    recorder = Recorder(
        window_title="Untitled - Notepad",
        task="Calibrate cursor-to-frame mapping",
        output_dir=output_dir,
        capture=capture,
        eye_processor=eye,
        user32=object(),
        calibration_only=True,
        capture_interval=0.0001,
        poll_interval=0.0001,
        event_sink=events.append,
    )

    assert recorder.run() == 1
    assert not output_dir.exists()
    assert capture.closed
    assert capture.captured_rects == [(0, 0, 100, 100)]
    assert len(eye.frames) == 1
    assert eye.frames[0].shape == (100, 100, 3)
    assert any("Live pixels #1" in event for event in events)
    assert any("capture=" in event and "eye=" in event for event in events)
    assert any(
        "recent " in event and ("static-reuse" in event or "dynamic-pass" in event)
        for event in events
    )
    assert any("no screenshots or samples are saved" in event for event in events)
    assert events[-1] == "Selected window closed; live calibration stopped."


def test_calibration_mode_rejects_training_target_labels():
    with pytest.raises(ValueError, match="does not accept"):
        WindowsDemoRecorder(
            window_title="Untitled - Notepad",
            task="Calibrate cursor-to-frame mapping",
            output_dir=None,
            user32=object(),
            target_labels=["File menu"],
            calibration_only=True,
        )

    recorder = WindowsDemoRecorder(
        window_title="Untitled - Notepad",
        task="Calibrate cursor-to-frame mapping",
        output_dir=None,
        user32=object(),
        calibration_only=True,
    )
    assert recorder.output_dir is None


def test_calibration_cli_rejects_output_directory_without_creating_it(tmp_path):
    output_dir = tmp_path / "must-not-be-created"
    with pytest.raises(SystemExit):
        main(
            [
                "--window-title",
                "Untitled - Notepad",
                "--task",
                "Calibrate cursor-to-frame mapping",
                "--output-dir",
                str(output_dir),
                "--calibrate-only",
                "--confirm-isolated-window",
            ]
        )
    assert not output_dir.exists()


def test_recorder_supports_predeclared_targets_for_console_free_capture(tmp_path):
    events = []
    recorder = WindowsDemoRecorder(
        window_title="Untitled - Notepad",
        task="Ground visible controls",
        output_dir=tmp_path / "session",
        user32=object(),
        target_labels=["File menu", "Edit menu"],
        event_sink=events.append,
    )

    recorder._announce_next_label(0)

    assert events == ["Next target (1/2): File menu"]
    assert not recorder._all_predefined_targets_recorded(0)
    assert recorder._next_label() == "File menu"
    assert recorder._expected_click_count == 2
    assert recorder._next_label() == "Edit menu"
    assert recorder._expected_click_count == 2
    assert not recorder._all_predefined_targets_recorded(1)
    assert recorder._all_predefined_targets_recorded(2)
    assert recorder._next_label() is None


def test_recorder_rejects_invalid_predeclared_target_labels(tmp_path):
    for label in ("", "é" * 49):
        with pytest.raises(ValueError, match="Predefined target labels"):
            WindowsDemoRecorder(
                window_title="Untitled - Notepad",
                task="Ground visible controls",
                output_dir=tmp_path / "session",
                user32=object(),
                target_labels=[label],
            )


def test_recorder_supports_predeclared_absence_labels(tmp_path):
    events = []
    recorder = WindowsDemoRecorder(
        window_title="Untitled - Notepad",
        task="Calibrate target absence",
        output_dir=tmp_path / "session",
        user32=object(),
        absent_target_labels=["Save button", "Export menu"],
        event_sink=events.append,
    )

    recorder._announce_next_absent_label(0)

    assert events == ["Next absent target (1/2): Save button"]
    assert recorder._next_absent_label() == "Save button"
    assert not recorder._all_predefined_targets_recorded(0, 1)
    assert recorder._next_absent_label() == "Export menu"
    assert recorder._all_predefined_targets_recorded(0, 2)


def test_recorder_writes_target_absence_sample_without_audit_click(tmp_path):
    class Capture:
        virtual_left = 0
        virtual_top = 0
        virtual_width = 100
        virtual_height = 80

        def open(self):
            pass

        def get_geometry(self):
            return self

        def grab_region(self, rect):
            return np.zeros((rect[3] - rect[1], rect[2] - rect[0], 3), dtype=np.uint8)

        def close(self):
            pass

    class Recorder(WindowsDemoRecorder):
        def _selected_window(self):
            return WindowInfo(1, "AVCS_DEMO_WINDOW", (0, 0, 100, 80))

        def _owns_foreground(self, _target_hwnd):
            return True

        def _left_button_down(self):
            return False

        def _right_button_down(self):
            self.right_checks = getattr(self, "right_checks", 0) + 1
            return self.right_checks == 2

    output = tmp_path / "absence-session"
    recorder = Recorder(
        window_title="AVCS_DEMO_WINDOW",
        task="Find an absent Save button",
        output_dir=output,
        capture=Capture(),
        user32=object(),
        absent_target_labels=["Save button"],
        capture_interval=0.0001,
        poll_interval=0.0001,
    )

    assert recorder.run() == 1
    sample = json.loads((output / "samples.jsonl").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert sample["action"] == {"type": "target_absent"}
    assert sample["target_label"] == "Save button"
    assert manifest["sample_count"] == 1
    assert list((output / "audit").iterdir()) == []


def test_win32_hardware_input_uses_documented_field_names():
    assert [name for name, _ in HARDWAREINPUT._fields_] == [
        "uMsg",
        "wParamL",
        "wParamH",
    ]
