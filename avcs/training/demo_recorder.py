"""Opt-in local recorder for labeled clicks in one isolated Windows window."""
from __future__ import annotations

import argparse
import ctypes
import json
import queue
import sys
import threading
import time
import uuid
from collections import deque
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from avcs.query_limits import MAX_QUERY_BYTES


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    rect: Tuple[int, int, int, int]


def find_exact_window_title(target: str, windows: Iterable[WindowInfo]) -> WindowInfo:
    """Require one visible window with an exact title match."""
    title = target.strip()
    if not title:
        raise ValueError("A non-empty exact window title is required.")
    matches = [window for window in windows if window.title == title]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one visible window titled {title!r}; found {len(matches)}."
        )
    return matches[0]


def physical_window_bounds(hwnd: int, dwm_get_window_attribute: Any) -> Tuple[int, int, int, int]:
    """Read window bounds in physical screen pixels, matching GDI capture coordinates."""
    rect = wintypes.RECT()
    result = dwm_get_window_attribute(
        hwnd,
        9,
        ctypes.byref(rect),
        ctypes.sizeof(rect),
    )
    if result != 0:
        raise OSError(f"DwmGetWindowAttribute failed with HRESULT {result}.")
    bounds = (rect.left, rect.top, rect.right, rect.bottom)
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise OSError(f"Window {hwnd} has invalid physical bounds: {bounds}.")
    return bounds


def crop_window_frame(
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    virtual_origin: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop a screen frame to a window; return crop pixels and screen-space bounds."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected an RGB frame with shape (height, width, 3).")

    left, top, right, bottom = (int(value) for value in rect)
    origin_x, origin_y = virtual_origin
    x0 = max(0, left - origin_x)
    y0 = max(0, top - origin_y)
    x1 = min(frame.shape[1], right - origin_x)
    y1 = min(frame.shape[0], bottom - origin_y)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("The target window is outside the captured screen frame.")

    screen_bounds = (x0 + origin_x, y0 + origin_y, x1 + origin_x, y1 + origin_y)
    return frame[y0:y1, x0:x1].copy(), screen_bounds


def normalized_click(
    point: Tuple[int, int],
    rect: Tuple[int, int, int, int],
) -> Optional[Tuple[float, float]]:
    """Map an in-window screen-pixel point to the cropped window's [0, 1] space."""
    left, top, right, bottom = rect
    x, y = point
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0 or not (left <= x < right and top <= y < bottom):
        return None
    return (x - left) / width, (y - top) / height


def build_click_record(
    *,
    task: str,
    target_label: str,
    frame_path: str,
    point_norm: Tuple[float, float],
    recorded_at_utc: str,
) -> dict[str, Any]:
    """Create a versioned, text-free action label record."""
    return {
        "schema_version": 1,
        "task": task,
        "target_label": target_label,
        "frame": frame_path,
        "action": {
            "type": "mouse_click",
            "button": "left",
            "point_norm": [round(point_norm[0], 6), round(point_norm[1], 6)],
        },
        "recorded_at_utc": recorded_at_utc,
    }


def render_click_audit(frame_rgb: np.ndarray, point_norm: Tuple[float, float]) -> Image.Image:
    """Render a review-only marker without altering the source training frame."""
    if (
        frame_rgb.ndim != 3
        or frame_rgb.shape[2] != 3
        or frame_rgb.dtype != np.uint8
    ):
        raise ValueError("frame_rgb must be a uint8 RGB image.")
    x_norm, y_norm = point_norm
    if not (
        np.isfinite(x_norm)
        and np.isfinite(y_norm)
        and 0 <= x_norm < 1
        and 0 <= y_norm < 1
    ):
        raise ValueError("point_norm coordinates must be finite and in [0, 1).")
    image = Image.fromarray(frame_rgb.copy())
    x = round(x_norm * (image.width - 1))
    y = round(y_norm * (image.height - 1))
    radius = max(5, min(12, min(image.width, image.height) // 100))
    draw = ImageDraw.Draw(image)
    color = (255, 40, 40)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    draw.line((x - radius * 2, y, x + radius * 2, y), fill=color, width=2)
    draw.line((x, y - radius * 2, x, y + radius * 2), fill=color, width=2)
    return image


class WindowsDemoRecorder:
    """Capture target-window demonstrations or cursor calibration while foregrounded."""

    def __init__(
        self,
        *,
        window_title: str,
        task: str,
        output_dir: Optional[Path],
        capture: Optional[Any] = None,
        eye_processor: Optional[Any] = None,
        user32: Optional[Any] = None,
        label_queue: Optional[queue.Queue[str]] = None,
        target_labels: Optional[Sequence[str]] = None,
        calibration_only: bool = False,
        event_sink: Optional[Any] = None,
        poll_interval: float = 0.004,
        capture_interval: float = 0.05,
    ) -> None:
        if sys.platform != "win32" and user32 is None:
            raise OSError("The demonstration recorder requires Windows.")
        if not task.strip():
            raise ValueError("A task description is required.")
        self.window_title = window_title.strip()
        self.task = task.strip()
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.capture = capture
        self.eye_processor = eye_processor
        self.user32 = user32
        self.label_queue = label_queue or queue.Queue()
        self._target_labels = deque(target_labels or ())
        self._expected_click_count = len(self._target_labels)
        self.calibration_only = calibration_only
        if any(
            not label.strip()
            or len(label.strip().encode("utf-8")) > MAX_QUERY_BYTES
            for label in self._target_labels
        ):
            raise ValueError(
                f"Predefined target labels must contain 1-{MAX_QUERY_BYTES} UTF-8 bytes."
            )
        if calibration_only and self._expected_click_count:
            raise ValueError("Calibration mode does not accept demonstration target labels.")
        self.event_sink = event_sink
        self.poll_interval = poll_interval
        self.capture_interval = capture_interval
        self._stop = threading.Event()
        self._native_user32 = user32 is None
        self._user32 = user32 if user32 is not None else ctypes.windll.user32
        self._dwmapi = ctypes.windll.dwmapi if self._native_user32 else None
        if poll_interval <= 0 or capture_interval <= 0:
            raise ValueError("Polling and capture intervals must be positive.")
        if self._native_user32:
            self._configure_user32()

    def _configure_user32(self) -> None:
        self._user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self._user32.GetWindowTextLengthW.restype = ctypes.c_int
        self._user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self._user32.GetWindowTextW.restype = ctypes.c_int
        self._user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self._user32.IsWindowVisible.restype = wintypes.BOOL
        self._user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self._user32.GetWindowRect.restype = wintypes.BOOL
        self._user32.GetPhysicalCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        self._user32.GetPhysicalCursorPos.restype = wintypes.BOOL
        self._dwmapi.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self._dwmapi.DwmGetWindowAttribute.restype = wintypes.HRESULT
        self._user32.EnumWindows.argtypes = [
            ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM),
            wintypes.LPARAM,
        ]
        self._user32.EnumWindows.restype = wintypes.BOOL
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        self._user32.GetAncestor.restype = wintypes.HWND
        self._user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        self._user32.GetCursorPos.restype = wintypes.BOOL
        self._user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self._user32.GetAsyncKeyState.restype = ctypes.c_short

    def _window_text(self, hwnd: int) -> str:
        length = int(self._user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        self._user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value

    def _visible_windows(self) -> list[WindowInfo]:
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        windows: list[WindowInfo] = []
        errors: list[OSError] = []

        def visit(hwnd: int, _lparam: int) -> bool:
            if not self._user32.IsWindowVisible(hwnd):
                return True
            title = self._window_text(hwnd)
            if self._native_user32:
                try:
                    rect = physical_window_bounds(hwnd, self._dwmapi.DwmGetWindowAttribute)
                except OSError as exc:
                    if title == self.window_title:
                        errors.append(exc)
                    return True
            else:
                rect_type = wintypes.RECT()
                if not self._user32.GetWindowRect(hwnd, ctypes.byref(rect_type)):
                    return True
                rect = (rect_type.left, rect_type.top, rect_type.right, rect_type.bottom)
            windows.append(WindowInfo(int(hwnd), title, rect))
            return True

        callback = callback_type(visit)
        self._user32.EnumWindows(callback, 0)
        if errors:
            raise errors[0]
        return windows

    def _selected_window(self) -> WindowInfo:
        return find_exact_window_title(self.window_title, self._visible_windows())

    def _owns_foreground(self, target_hwnd: int) -> bool:
        foreground = int(self._user32.GetForegroundWindow() or 0)
        if not foreground:
            return False
        root_owner = int(self._user32.GetAncestor(foreground, 3) or foreground)
        return root_owner == target_hwnd

    def _cursor_position(self) -> Tuple[int, int]:
        point = wintypes.POINT()
        if not self._user32.GetPhysicalCursorPos(ctypes.byref(point)):
            raise OSError("GetPhysicalCursorPos failed while recording a target-window click.")
        return int(point.x), int(point.y)

    def _left_button_down(self) -> bool:
        return bool(int(self._user32.GetAsyncKeyState(0x01)) & 0x8000)

    def _capture_rgb(
        self, rect: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        grab_region = getattr(self.capture, "grab_region", None)
        if rect is not None and callable(grab_region):
            geo = self.capture.get_geometry()
            capture_rect = (
                max(rect[0], geo.virtual_left),
                max(rect[1], geo.virtual_top),
                min(rect[2], geo.virtual_left + geo.virtual_width),
                min(rect[3], geo.virtual_top + geo.virtual_height),
            )
            if capture_rect[2] <= capture_rect[0] or capture_rect[3] <= capture_rect[1]:
                raise ValueError("The target window does not intersect the captured screen.")
            return grab_region(capture_rect), (capture_rect[0], capture_rect[1])

        frame = self.capture.grab()
        geometry = self.capture.get_geometry()
        return frame, (geometry.virtual_left, geometry.virtual_top)

    def _write_manifest(self, *, status: str, sample_count: int) -> None:
        if self.output_dir is None:
            raise ValueError("An output directory is required to write a recording manifest.")
        manifest = {
            "schema_version": 1,
            "status": status,
            "recording_mode": "demonstration",
            "task": self.task,
            "target_window_title": self.window_title,
            "started_at_utc": self._started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample_count": sample_count,
            "capture_scope": "cropped target window only",
            "keyboard_recording": False,
            "network_upload": False,
        }
        path = self.output_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def run(self) -> int:
        if self.calibration_only:
            return self._run_live_calibration()
        if self.output_dir is None:
            raise ValueError("An output directory is required for demonstration recording.")
        if self.output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {self.output_dir}")

        window = self._selected_window()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir()
        audit_dir = self.output_dir / "audit"
        audit_dir.mkdir()
        self._started_at_utc = datetime.now(timezone.utc).isoformat()
        records_path = self.output_dir / "samples.jsonl"
        sample_count = 0
        previous_left_down = False
        latest_crop: Optional[np.ndarray] = None
        latest_rect: Optional[Tuple[int, int, int, int]] = None
        latest_window_rect: Optional[Tuple[int, int, int, int]] = None
        latest_capture_time = 0.0
        last_capture_time = 0.0

        if self.capture is None:
            from avcs.capture_adapter import Win32GDICaptureAdapter

            self.capture = Win32GDICaptureAdapter()
        self._started_at_utc = datetime.now(timezone.utc).isoformat()
        status = "failed"
        capture_open = False
        records_file = records_path.open("x", encoding="utf-8", newline="\n")
        try:
            self.capture.open()
            capture_open = True
            if self.event_sink:
                self.event_sink(
                    "Recorder ready. Focus the selected window; only labeled left clicks are recorded."
                )
                self._announce_next_label(sample_count)

            with records_file:
                while not self._stop.is_set():
                    if not self._owns_foreground(window.hwnd):
                        previous_left_down = False
                        latest_crop = None
                        latest_rect = None
                        latest_window_rect = None
                        time.sleep(self.poll_interval)
                        continue

                    now = time.monotonic()
                    if now - last_capture_time >= self.capture_interval:
                        capture_window = self._selected_window()
                        frame, origin = self._capture_rgb(capture_window.rect)
                        current_window = self._selected_window()
                        last_capture_time = now
                        if current_window.rect == capture_window.rect:
                            crop, crop_rect = crop_window_frame(
                                frame, capture_window.rect, origin
                            )
                            latest_crop = crop
                            latest_rect = crop_rect
                            latest_capture_time = now
                            latest_window_rect = capture_window.rect
                        else:
                            latest_crop = None
                            latest_rect = None
                            latest_window_rect = None

                    left_down = self._left_button_down()
                    if left_down and not previous_left_down:
                        point = self._cursor_position()
                        point_norm = (
                            normalized_click(point, latest_rect) if latest_rect is not None else None
                        )
                        age = time.monotonic() - latest_capture_time
                        current_window = self._selected_window()
                        if (
                            current_window.rect == latest_window_rect
                            and latest_crop is not None
                            and point_norm is not None
                            and age <= self.capture_interval * 3
                        ):
                            label = self._next_label()
                        else:
                            label = None
                        if label and latest_crop is not None and point_norm is not None:
                            sample_id = uuid.uuid4().hex
                            relative_frame = f"frames/{sample_id}.png"
                            Image.fromarray(latest_crop).save(self.output_dir / relative_frame)
                            render_click_audit(latest_crop, point_norm).save(
                                audit_dir / f"{sample_id}.png"
                            )
                            record = build_click_record(
                                task=self.task,
                                target_label=label,
                                frame_path=relative_frame,
                                point_norm=point_norm,
                                recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                            )
                            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            records_file.flush()
                            sample_count += 1
                            if self.event_sink:
                                self.event_sink(f"Recorded labeled click {sample_count}: {label}")
                                self._announce_next_label(sample_count)
                            if self._all_predefined_targets_recorded(sample_count):
                                self._stop.set()
                        elif self.event_sink:
                            self.event_sink(
                                "Click ignored: provide a target label and keep the click inside "
                                "the selected window."
                            )
                    previous_left_down = left_down
                    time.sleep(self.poll_interval)
            status = "stopped"
        except KeyboardInterrupt:
            status = "interrupted"
        finally:
            self._stop.set()
            try:
                records_file.close()
            finally:
                try:
                    if capture_open:
                        self.capture.close()
                finally:
                    self._write_manifest(status=status, sample_count=sample_count)
        return sample_count

    def _run_live_calibration(self) -> int:
        """Check live pixel/cursor alignment in memory without retaining frames."""
        window = self._selected_window()
        if self.capture is None:
            from avcs.capture_adapter import Win32GDICaptureAdapter

            self.capture = Win32GDICaptureAdapter()

        frame_count = 0
        last_report_at = 0.0
        last_focus_message = False
        capture_open = False
        eye_created_here = self.eye_processor is None
        eye = self.eye_processor
        frame_times: deque[float] = deque(maxlen=61)
        if self.event_sink:
            self.event_sink(
                "Live calibration running: pixels stay in memory, no screenshots or samples "
                "are saved. HumanEye2M processes each fresh frame; move the pointer over "
                "visible locations without clicking; Ctrl+C stops."
            )
        try:
            self.capture.open()
            capture_open = True
            if eye is None:
                from avcs.cortex_2m import HumanEye2M

                eye = HumanEye2M(source="synthetic", threaded_capture=False)
            while not self._stop.is_set():
                tick_started = time.monotonic()
                capture_window = self._calibration_window_if_present()
                if capture_window is None:
                    break
                if not self._owns_foreground(window.hwnd):
                    if not last_focus_message and self.event_sink:
                        self.event_sink(
                            "Paused: focus the selected window to resume the live pixel stream."
                        )
                    last_focus_message = True
                    time.sleep(self.poll_interval)
                    continue

                capture_started = time.perf_counter()
                frame, origin = self._capture_rgb(window.rect)
                capture_latency_ms = (time.perf_counter() - capture_started) * 1000.0
                current_window = self._calibration_window_if_present()
                if current_window is None:
                    del frame
                    break
                cursor = self._cursor_position()
                if current_window.rect != capture_window.rect:
                    del frame
                    time.sleep(self.poll_interval)
                    continue
                crop, crop_rect = crop_window_frame(frame, capture_window.rect, origin)
                point_norm = normalized_click(cursor, crop_rect)
                if point_norm is None:
                    del frame, crop
                    time.sleep(self.poll_interval)
                    continue

                perception = eye.perceive(frame=crop)
                frame_count += 1
                now = time.monotonic()
                frame_times.append(now)
                if self.event_sink and now - last_report_at >= 0.5:
                    local_x = cursor[0] - crop_rect[0]
                    local_y = cursor[1] - crop_rect[1]
                    recent_fps = (
                        (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                        if len(frame_times) > 1 and frame_times[-1] > frame_times[0]
                        else 0.0
                    )
                    eye_mode = "static-reuse" if perception["is_static"] else "dynamic-pass"
                    self.event_sink(
                        f"Live pixels #{frame_count} | recent {recent_fps:.1f} FPS | "
                        f"capture={capture_latency_ms:.1f}ms | "
                        f"eye={perception['latency_ms']:.1f}ms ({eye_mode}) | "
                        f"motion={perception['motion_magnitude']:.4f} | "
                        f"frame {crop.shape[1]}x{crop.shape[0]} | "
                        f"weights_loaded={perception['weights_loaded']} | "
                        f"cursor screen={cursor}, crop=({local_x}, {local_y}), "
                        f"normalized=({point_norm[0]:.4f}, {point_norm[1]:.4f})"
                    )
                    last_report_at = now
                last_focus_message = False
                del frame, crop, perception
                delay = self.capture_interval - (time.monotonic() - tick_started)
                if delay > 0:
                    time.sleep(delay)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            try:
                if eye_created_here and eye is not None:
                    eye.close()
            finally:
                if capture_open:
                    self.capture.close()
        return frame_count

    def _calibration_window_if_present(self) -> Optional[WindowInfo]:
        try:
            return self._selected_window()
        except RuntimeError as exc:
            message = str(exc)
            target_missing = (
                message.startswith("Expected exactly one visible window titled ")
                and message.endswith("found 0.")
            )
            if target_missing:
                if self.event_sink:
                    self.event_sink("Selected window closed; live calibration stopped.")
                return None
            raise

    def _next_label(self) -> Optional[str]:
        if self._target_labels:
            return self._target_labels.popleft()
        try:
            label = self.label_queue.get_nowait().strip()
        except queue.Empty:
            return None
        if not label or len(label.encode("utf-8")) > MAX_QUERY_BYTES:
            return None
        return label

    def _all_predefined_targets_recorded(self, sample_count: int) -> bool:
        return bool(
            self._expected_click_count and sample_count >= self._expected_click_count
        )

    def _announce_next_label(self, sample_count: int) -> None:
        if self._target_labels and self.event_sink:
            self.event_sink(
                f"Next target ({sample_count + 1}/{sample_count + len(self._target_labels)}): "
                f"{self._target_labels[0]}"
            )

    def stop(self) -> None:
        self._stop.set()


def _read_target_labels(recorder: WindowsDemoRecorder) -> None:
    while not recorder._stop.is_set():
        try:
            label = input(
                "Next click target label (or :stop): "
            ).strip()
        except EOFError:
            recorder.stop()
            return
        if label.lower() == ":stop":
            recorder.stop()
            return
        if not label:
            print("Label cannot be empty; that click will not be recorded.", file=sys.stderr)
            continue
        if len(label.encode("utf-8")) > MAX_QUERY_BYTES:
            print(
                f"Label exceeds {MAX_QUERY_BYTES} UTF-8 bytes; try a shorter description.",
                file=sys.stderr,
            )
            continue
        recorder.label_queue.put(label)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record labeled clicks or no-click cursor calibration from one isolated "
            "Windows window."
        )
    )
    parser.add_argument("--window-title", required=True, help="Exact, unique visible window title.")
    parser.add_argument("--task", required=True, help="The operator's task for this demonstration.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New local directory for cropped frames, records, audit images, and manifest.",
    )
    parser.add_argument(
        "--confirm-isolated-window",
        action="store_true",
        help="Confirm this is a disposable test window with no private content.",
    )
    parser.add_argument(
        "--target-label",
        action="append",
        dest="target_labels",
        help="Predeclare the next click target; repeat for a guided, non-interactive session.",
    )
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Stream live pixels and cursor coordinates in memory; never save frames.",
    )
    args = parser.parse_args(argv)
    if not args.confirm_isolated_window:
        parser.error("Pass --confirm-isolated-window only for a disposable, non-sensitive test window.")
    if args.calibrate_only and args.target_labels:
        parser.error("--calibrate-only cannot be combined with --target-label.")
    if args.calibrate_only and args.output_dir is not None:
        parser.error("--calibrate-only is an in-memory live stream and does not accept --output-dir.")
    if not args.calibrate_only and args.output_dir is None:
        parser.error("--output-dir is required when recording labeled demonstrations.")
    if not args.calibrate_only and not args.target_labels and not sys.stdin.isatty():
        parser.error("Interactive target labels require a terminal; no capture was started.")
    if args.target_labels and any(
        not label.strip()
        or len(label.strip().encode("utf-8")) > MAX_QUERY_BYTES
        for label in args.target_labels
    ):
        parser.error(
            f"Each --target-label must contain 1-{MAX_QUERY_BYTES} non-whitespace UTF-8 bytes."
        )

    recorder = WindowsDemoRecorder(
        window_title=args.window_title,
        task=args.task,
        output_dir=args.output_dir,
        target_labels=args.target_labels,
        calibration_only=args.calibrate_only,
        event_sink=print,
    )
    if not args.calibrate_only and not args.target_labels:
        reader = threading.Thread(target=_read_target_labels, args=(recorder,), daemon=True)
        reader.start()
    try:
        count = recorder.run()
    except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
        print(f"Recorder stopped without claiming success: {exc}", file=sys.stderr)
        return 1
    if args.calibrate_only:
        print(
            f"Live calibration stopped after {count} in-memory frames; no files were saved."
        )
    else:
        print(f"Recorded {count} labeled click(s) locally in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
