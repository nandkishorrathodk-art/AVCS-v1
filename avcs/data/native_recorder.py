"""Phase 2: Zero-Lag Windows Native Recorder & LeRobot v3.0 Exporter.

Captures pure-RGB desktop frames with monotonic `time.perf_counter_ns()` timestamps,
captures OS input events via Win32 Raw Input (`WM_INPUT`) without polling gaps,
binds strictly pre-action frames (`frame_present_ns <= action_event_ns - 2ms`),
curates trajectories via `TrajectoryCleaner`, audits `<5ms` sync via `QADoctor`,
and exports canonical `LeRobotV3TrajectoryEpisode` Parquet/JSONL shards.

Architecture & Privacy Invariants:
- Pure-RGB only: zero window-title scraping, zero DOM/UIA/OCR text extraction.
- Headless Linux / Kaggle import safe: zero module-level Win32/ctypes.windll calls.
"""
from __future__ import annotations

import argparse
from collections import deque
import ctypes
from dataclasses import dataclass, field
import io
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union
import uuid

import numpy as np
from PIL import Image

from avcs.data.lerobot_v3 import (
    LeRobotV3TrajectoryEpisode,
    QADoctor,
    TrajectoryCleaner,
    load_lerobot_v3_shards,
    save_lerobot_v3_shards,
)
from avcs.runtime.capture_adapter import (
    BaseCaptureAdapter,
    DisplayGeometry,
    InteractiveSyntheticCaptureAdapter,
    SyntheticCaptureAdapter,
)

# Canonical 10-action vocabulary matching avcs.data.lerobot_v3 & train_vla_gui_actor
ACTION_LEFT_CLICK = 0
ACTION_RIGHT_CLICK = 1
ACTION_DOUBLE_CLICK = 2
ACTION_MOVE = 3
ACTION_DRAG = 4
ACTION_SCROLL = 5
ACTION_HOTKEY = 6
ACTION_TYPE = 7
ACTION_WAIT = 8
ACTION_STOP = 9

# Win32 Raw Input constants (used only inside Windows runtime paths)
WM_INPUT = 0x00FF
WM_CLOSE = 0x0010
WM_DESTROY = 0x0002
RIM_TYPEMOUSE = 0
RIM_TYPEKEYBOARD = 1
RID_INPUT = 0x10000003
RIDEV_INPUTSINK = 0x00000100

RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
RI_MOUSE_LEFT_BUTTON_UP = 0x0002
RI_MOUSE_RIGHT_BUTTON_DOWN = 0x0004
RI_MOUSE_RIGHT_BUTTON_UP = 0x0008
RI_MOUSE_WHEEL = 0x0400
RI_MOUSE_HWHEEL = 0x0800
RI_KEY_BREAK = 0x0001

VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_ESCAPE = 0x1B
VK_RETURN = 0x0D
VK_TAB = 0x09
VK_BACK = 0x08
VK_SPACE = 0x20
VK_F12 = 0x7B


@dataclass
class CapturedFrame:
    """Single monotonic pre-action frame entry stored in the ring buffer."""

    seq: int
    present_ns: int
    width: int
    height: int
    dpi_scale: float
    preview_32x32: np.ndarray
    image_bytes: bytes = b""
    raw_rgb: Optional[np.ndarray] = None
    _encode_size: Optional[Tuple[int, int]] = None
    _jpeg_quality: int = 85

    def ensure_jpeg_bytes(self) -> bytes:
        """Lazily compresses the buffered RGB frame to JPEG only when bound to an action step."""
        if self.image_bytes:
            return self.image_bytes
        if self.raw_rgb is None:
            return b""
        pil = Image.fromarray(self.raw_rgb, mode="RGB")
        if self._encode_size is not None and pil.size != self._encode_size:
            pil = pil.resize(self._encode_size, Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self._jpeg_quality)
        self.image_bytes = buf.getvalue()
        self.raw_rgb = None
        return self.image_bytes


@dataclass(frozen=True)
class RawInputEvent:
    """Normalized low-latency OS input event timestamped with `perf_counter_ns()`."""

    event_ns: int
    kind: str  # mouse_move, left_down, left_up, right_down, right_up, double_click, wheel, key_down, key_up, stop
    screen_xy: Tuple[int, int] = (0, 0)
    wheel_delta: Tuple[float, float] = (0.0, 0.0)
    vk_code: int = 0
    char: str = ""
    modifiers: Tuple[str, ...] = ()


@dataclass
class RecorderConfig:
    """Configuration for ZeroLagWindowsRecorder."""

    task_instruction: str = "Complete desktop workflow cleanly"
    backend: str = "auto"  # auto, wgc, dxgi, gdi, mss, synthetic
    target_fps: float = 120.0
    max_sync_target_ms: float = 5.0
    min_pre_action_margin_ns: int = 2_000_000
    jpeg_quality: int = 85
    capture_size: Optional[Tuple[int, int]] = None  # optional (W, H) resize at encode
    ring_capacity: int = 128
    min_drag_px: float = 25.0
    min_move_px: float = 24.0
    double_click_window_ns: int = 350_000_000
    idle_wait_interval_ns: int = 1_000_000_000
    retain_recovery_fraction: float = 0.12
    redact_boxes_norm: Optional[Sequence[Tuple[float, float, float, float]]] = None
    os_platform: str = "windows"


def _enable_windows_dpi_awareness() -> float:
    """Enables Per-Monitor DPI Awareness v2 on Windows and returns the active display DPI scale."""
    if sys.platform != "win32":
        return 1.0
    try:
        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
        try:
            dpi = int(user32.GetDpiForSystem())
            if dpi > 0:
                return float(dpi) / 96.0
        except Exception:
            pass
    except Exception:
        pass
    return 1.0


class _MSSCaptureAdapter(BaseCaptureAdapter):
    """Lightweight fallback capture adapter backed by `mss` when selected."""

    def __init__(self) -> None:
        self._sct: Any = None
        self._geo: Optional[DisplayGeometry] = None

    def open(self) -> "_MSSCaptureAdapter":
        import mss

        self._sct = mss.mss()
        mon = self._sct.monitors[0]
        self._geo = DisplayGeometry(
            virtual_left=int(mon["left"]),
            virtual_top=int(mon["top"]),
            virtual_width=int(mon["width"]),
            virtual_height=int(mon["height"]),
            dpi_scale=1.0,
            is_primary=True,
        )
        return self

    def grab(self) -> np.ndarray:
        if self._sct is None:
            self.open()
        raw = self._sct.grab(self._sct.monitors[0])
        bgra = np.frombuffer(raw.bgra, dtype=np.uint8).reshape((raw.height, raw.width, 4))
        return bgra[:, :, 2::-1].copy()

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            finally:
                self._sct = None

    def get_geometry(self) -> DisplayGeometry:
        if self._geo is None:
            self.open()
        assert self._geo is not None
        return self._geo

    def supported_modes(self) -> List[str]:
        return ["mss_bgra_rgb"]


class _DXGICaptureAdapter(BaseCaptureAdapter):
    """Optional zero-copy Desktop Duplication backend when `dxcam` is installed."""

    def __init__(self) -> None:
        self._cam: Any = None
        self._geo: Optional[DisplayGeometry] = None

    def open(self) -> "_DXGICaptureAdapter":
        import dxcam  # type: ignore[import-not-found]

        self._cam = dxcam.create(output_color="RGB")
        w, h = self._cam.width, self._cam.height
        self._geo = DisplayGeometry(0, 0, int(w), int(h), 1.0, True)
        return self

    def grab(self) -> np.ndarray:
        if self._cam is None:
            self.open()
        frame = self._cam.grab()
        if frame is None:
            raise RuntimeError("DXGI returned empty frame")
        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        if self._cam is not None:
            try:
                del self._cam
            finally:
                self._cam = None

    def get_geometry(self) -> DisplayGeometry:
        if self._geo is None:
            self.open()
        assert self._geo is not None
        return self._geo

    def supported_modes(self) -> List[str]:
        return ["dxgi_desktop_duplication"]


class _WGCCaptureAdapter(BaseCaptureAdapter):
    """Windows.Graphics.Capture (`windows_capture`) zero-lag backend when installed."""

    def __init__(self) -> None:
        self._latest: Optional[np.ndarray] = None
        self._latest_ns: int = 0
        self._frame_seq: int = 0
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._new_frame_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._control: Any = None
        self._geo: Optional[DisplayGeometry] = None

    def open(self) -> "_WGCCaptureAdapter":
        dpi_scale = _enable_windows_dpi_awareness()
        from windows_capture import CaptureControl, Frame, InternalCaptureControl, WindowsCapture  # type: ignore[import-not-found]

        capture = WindowsCapture(cursor_capture=True, draw_border=False)

        @capture.event
        def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl) -> None:
            captured_ns = time.perf_counter_ns()
            self._control = capture_control
            buf = np.asarray(frame.convert_to_bgr().frame_buffer, dtype=np.uint8)[:, :, 2::-1].copy()
            with self._lock:
                self._latest = buf
                self._latest_ns = captured_ns
                self._frame_seq += 1
                if self._geo is None:
                    self._geo = DisplayGeometry(0, 0, buf.shape[1], buf.shape[0], dpi_scale, True)
            self._event.set()
            self._new_frame_event.set()

        @capture.event
        def on_closed() -> None:
            pass

        def _run() -> None:
            capture.start()

        self._thread = threading.Thread(target=_run, name="AVCS-WGCCapture", daemon=True)
        self._thread.start()
        self._event.wait(timeout=3.0)
        if self._geo is None:
            raise RuntimeError("Windows.Graphics.Capture did not deliver initial frame within 3s")
        return self

    def grab_with_timestamp(self, timeout_s: float = 0.0) -> Tuple[np.ndarray, int, int]:
        """Waits up to `timeout_s` for a fresh WGC callback and returns (rgb, captured_ns, seq)."""
        if self._thread is None:
            self.open()
        if timeout_s > 0.0:
            self._new_frame_event.wait(timeout=timeout_s)
            self._new_frame_event.clear()
        with self._lock:
            if self._latest is None:
                raise RuntimeError("No WGC frame buffered")
            return self._latest, self._latest_ns, self._frame_seq

    def grab(self) -> np.ndarray:
        if self._thread is None:
            self.open()
        with self._lock:
            if self._latest is None:
                raise RuntimeError("No WGC frame buffered")
            return self._latest.copy()

    def close(self) -> None:
        if self._control is not None:
            try:
                self._control.stop()
            except Exception:
                pass
            self._control = None

    def get_geometry(self) -> DisplayGeometry:
        if self._geo is None:
            self.open()
        assert self._geo is not None
        return self._geo

    def supported_modes(self) -> List[str]:
        return ["windows_graphics_capture"]


def select_capture_backend(
    backend: str = "auto",
    width: int = 1280,
    height: int = 800,
) -> Tuple[str, BaseCaptureAdapter]:
    """Selects the fastest available capture adapter cleanly without failing imports on Linux."""
    req = (backend or "auto").strip().lower()
    if req == "synthetic":
        return "synthetic", InteractiveSyntheticCaptureAdapter(width=width, height=height)

    if sys.platform != "win32" and req == "auto":
        return "synthetic", InteractiveSyntheticCaptureAdapter(width=width, height=height)

    _enable_windows_dpi_awareness()

    candidates: List[str]
    if req == "auto":
        candidates = ["wgc", "dxgi", "gdi", "mss"]
    else:
        candidates = [req]

    errors: List[str] = []
    for cand in candidates:
        try:
            if cand == "wgc":
                adp: BaseCaptureAdapter = _WGCCaptureAdapter().open()
                return "wgc", adp
            if cand == "dxgi":
                adp = _DXGICaptureAdapter().open()
                return "dxgi", adp
            if cand == "gdi":
                from avcs.runtime.capture_adapter import Win32GDICaptureAdapter

                adp = Win32GDICaptureAdapter().open()
                return "gdi", adp
            if cand == "mss":
                adp = _MSSCaptureAdapter().open()
                return "mss", adp
        except Exception as exc:
            errors.append(f"{cand}: {exc}")

    if req == "auto":
        return "synthetic", InteractiveSyntheticCaptureAdapter(width=width, height=height)
    raise RuntimeError(f"Requested capture backend '{req}' unavailable ({'; '.join(errors)})")


class PreActionFrameRingBuffer:
    """Thread-safe monotonic ring buffer binding pre-action RGB frames to input events."""

    def __init__(
        self,
        capacity: int = 128,
        jpeg_quality: int = 85,
        encode_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self._frames: Deque[CapturedFrame] = deque(maxlen=max(4, int(capacity)))
        self._lock = threading.Lock()
        self._jpeg_quality = int(jpeg_quality)
        self._encode_size = encode_size
        self._last_present_ns = 0
        self._seq = 0

    @staticmethod
    def _downscale_32x32(rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        ys = np.linspace(0, max(0, h - 1), 32, dtype=np.int64)
        xs = np.linspace(0, max(0, w - 1), 32, dtype=np.int64)
        return rgb[np.ix_(ys, xs)].astype(np.float32)

    def push_frame(
        self,
        rgb: np.ndarray,
        dpi_scale: float = 1.0,
        present_ns: Optional[int] = None,
        *,
        copy_rgb: bool = True,
        cached_preview: Optional[np.ndarray] = None,
    ) -> CapturedFrame:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB (H, W, 3) frame, got shape {rgb.shape}")
        preview = cached_preview if cached_preview is not None else self._downscale_32x32(rgb)
        raw_ref = rgb.copy() if copy_rgb else rgb
        raw_ns = time.perf_counter_ns() if present_ns is None else int(present_ns)
        with self._lock:
            mono_ns = max(raw_ns, self._last_present_ns + 1)
            self._last_present_ns = mono_ns
            self._seq += 1
            entry = CapturedFrame(
                seq=self._seq,
                present_ns=mono_ns,
                width=int(rgb.shape[1]),
                height=int(rgb.shape[0]),
                dpi_scale=float(dpi_scale),
                preview_32x32=preview,
                raw_rgb=raw_ref,
                _encode_size=self._encode_size,
                _jpeg_quality=self._jpeg_quality,
            )
            self._frames.append(entry)
            return entry

    def bind_pre_action_frame(
        self,
        action_event_ns: int,
        min_margin_ns: int = 2_000_000,
    ) -> Optional[CapturedFrame]:
        """Returns the newest captured frame strictly preceding `action_event_ns - min_margin_ns`."""
        deadline_ns = int(action_event_ns) - int(min_margin_ns)
        with self._lock:
            for frame in reversed(self._frames):
                if frame.present_ns <= deadline_ns:
                    return frame
        return None

    def latest_frame(self) -> Optional[CapturedFrame]:
        with self._lock:
            return self._frames[-1] if self._frames else None

    @staticmethod
    def compute_pixel_diff_ratio(
        prev_frame: Optional[CapturedFrame],
        curr_frame: CapturedFrame,
    ) -> float:
        if prev_frame is None:
            return 0.01
        delta = np.abs(curr_frame.preview_32x32 - prev_frame.preview_32x32).mean() / 255.0
        return float(min(1.0, max(0.0, delta)))


def _vk_to_label_and_char(vk_code: int, modifiers: Sequence[str]) -> Tuple[bool, str, str]:
    """Maps a Win32 virtual key code cleanly into (is_hotkey, hotkey_label, typed_char)."""
    ctrl_or_alt = any(m in ("ctrl", "alt", "win") for m in modifiers)
    special_names = {
        VK_ESCAPE: "esc",
        VK_RETURN: "enter",
        VK_TAB: "tab",
        VK_BACK: "backspace",
        VK_F12: "f12",
        0x5B: "win",
        0x5C: "win",
        0x2E: "delete",
        0x24: "home",
        0x23: "end",
        0x25: "left",
        0x26: "up",
        0x27: "right",
        0x28: "down",
    }
    if vk_code in special_names:
        key_name = special_names[vk_code]
        combo = "+".join([*modifiers, key_name]) if modifiers else key_name
        return True, combo, ""

    if ord("A") <= vk_code <= ord("Z"):
        ch = chr(vk_code).lower() if "shift" not in modifiers else chr(vk_code)
        if ctrl_or_alt:
            combo = "+".join([*modifiers, chr(vk_code).lower()])
            return True, combo, ""
        return False, "", ch

    if ord("0") <= vk_code <= ord("9"):
        shift_digits = ")!@#$%^&*("
        ch = shift_digits[vk_code - ord("0")] if "shift" in modifiers else chr(vk_code)
        if ctrl_or_alt:
            return True, "+".join([*modifiers, chr(vk_code)]), ""
        return False, "", ch

    oem_chars = {
        0xBA: (";", ":"),
        0xBB: ("=", "+"),
        0xBC: (",", "<"),
        0xBD: ("-", "_"),
        0xBE: (".", ">"),
        0xBF: ("/", "?"),
        0xC0: ("`", "~"),
        0xDB: ("[", "{"),
        0xDC: ("\\", "|"),
        0xDD: ("]", "}"),
        0xDE: ("'", '"'),
    }
    if vk_code in oem_chars:
        unshifted, shifted = oem_chars[vk_code]
        ch = shifted if "shift" in modifiers else unshifted
        if ctrl_or_alt:
            return True, "+".join([*modifiers, unshifted]), ""
        return False, "", ch

    if vk_code == VK_SPACE and not ctrl_or_alt:
        return False, "", " "

    if ctrl_or_alt:
        return True, "+".join([*modifiers, f"vk_{vk_code}"]), ""
    return False, "", ""


class Win32RawInputListener:
    """Dedicated message-only window (`HWND_MESSAGE`) listening for Win32 `WM_INPUT`."""

    def __init__(self, event_queue: queue.Queue[RawInputEvent]) -> None:
        if sys.platform != "win32":
            raise OSError("Win32RawInputListener requires Windows OS.")
        self._queue = event_queue
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._hwnd: int = 0
        self._error: Optional[BaseException] = None
        self._down_mods: set[str] = set()

    def start(self) -> None:
        if self._thread is not None:
            return
        _enable_windows_dpi_awareness()
        self._thread = threading.Thread(target=self._run_message_loop, name="AVCS-RawInput", daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=3.0)
        if self._error is not None:
            raise RuntimeError(f"Win32RawInputListener failed to initialize: {self._error}")

    def stop(self) -> None:
        self._stop_event.set()
        if self._hwnd:
            try:
                ctypes.windll.user32.PostMessageW(self._hwnd, WM_CLOSE, 0, 0)
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _run_message_loop(self) -> None:
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        LRESULT = ctypes.c_ssize_t
        WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

        class WNDCLASSEXW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.UINT),
                ("style", wintypes.UINT),
                ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE),
                ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR),
                ("hIconSm", wintypes.HICON),
            ]

        class RAWINPUTDEVICE(ctypes.Structure):
            _fields_ = [
                ("usUsagePage", wintypes.USHORT),
                ("usUsage", wintypes.USHORT),
                ("dwFlags", wintypes.DWORD),
                ("hwndTarget", wintypes.HWND),
            ]

        class RAWINPUTHEADER(ctypes.Structure):
            _fields_ = [
                ("dwType", wintypes.DWORD),
                ("dwSize", wintypes.DWORD),
                ("hDevice", wintypes.HANDLE),
                ("wParam", wintypes.WPARAM),
            ]

        class RAWMOUSE_BUTTONS(ctypes.Structure):
            _fields_ = [
                ("usButtonFlags", wintypes.USHORT),
                ("usButtonData", wintypes.USHORT),
            ]

        class RAWMOUSE_UNION(ctypes.Union):
            _fields_ = [("ulButtons", wintypes.ULONG), ("buttons", RAWMOUSE_BUTTONS)]

        class RAWMOUSE(ctypes.Structure):
            _fields_ = [
                ("usFlags", wintypes.USHORT),
                ("u", RAWMOUSE_UNION),
                ("ulRawButtons", wintypes.ULONG),
                ("lLastX", wintypes.LONG),
                ("lLastY", wintypes.LONG),
                ("ulExtraInformation", wintypes.ULONG),
            ]

        class RAWKEYBOARD(ctypes.Structure):
            _fields_ = [
                ("MakeCode", wintypes.USHORT),
                ("Flags", wintypes.USHORT),
                ("Reserved", wintypes.USHORT),
                ("VKey", wintypes.USHORT),
                ("Message", wintypes.UINT),
                ("ExtraInformation", wintypes.ULONG),
            ]

        class RAWINPUT_UNION(ctypes.Union):
            _fields_ = [("mouse", RAWMOUSE), ("keyboard", RAWKEYBOARD)]

        class RAWINPUT(ctypes.Structure):
            _fields_ = [("header", RAWINPUTHEADER), ("data", RAWINPUT_UNION)]

        class MSLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [
                ("pt", wintypes.POINT),
                ("mouseData", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        WH_MOUSE_LL = 14
        WM_MOUSEWHEEL = 0x020A
        WM_MOUSEHWHEEL = 0x020E
        HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        user32.RegisterClassExW.argtypes = [ctypes.c_void_p]
        user32.RegisterClassExW.restype = wintypes.ATOM
        user32.CreateWindowExW.argtypes = [
            wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
        ]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.DestroyWindow.argtypes = [wintypes.HWND]
        user32.DestroyWindow.restype = wintypes.BOOL
        user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
        user32.UnregisterClassW.restype = wintypes.BOOL
        user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.PostMessageW.restype = wintypes.BOOL
        user32.GetMessageW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.restype = wintypes.BOOL
        user32.TranslateMessage.argtypes = [ctypes.c_void_p]
        user32.TranslateMessage.restype = wintypes.BOOL
        user32.DispatchMessageW.argtypes = [ctypes.c_void_p]
        user32.DispatchMessageW.restype = LRESULT
        user32.PostQuitMessage.argtypes = [ctypes.c_int]
        user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.DefWindowProcW.restype = LRESULT
        user32.GetRawInputData.argtypes = [wintypes.HANDLE, wintypes.UINT, ctypes.c_void_p, ctypes.POINTER(wintypes.UINT), wintypes.UINT]
        user32.GetRawInputData.restype = wintypes.UINT
        user32.RegisterRawInputDevices.argtypes = [ctypes.POINTER(RAWINPUTDEVICE), wintypes.UINT, wintypes.UINT]
        user32.RegisterRawInputDevices.restype = wintypes.BOOL
        user32.GetPhysicalCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        user32.GetPhysicalCursorPos.restype = wintypes.BOOL
        user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
        user32.SetWindowsHookExW.restype = wintypes.HANDLE
        user32.UnhookWindowsHookEx.argtypes = [wintypes.HANDLE]
        user32.UnhookWindowsHookEx.restype = wintypes.BOOL
        user32.CallNextHookEx.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
        user32.CallNextHookEx.restype = LRESULT

        wheel_hook: List[Any] = [None]

        def _cursor_xy() -> Tuple[int, int]:
            pt = wintypes.POINT()
            if user32.GetPhysicalCursorPos(ctypes.byref(pt)):
                return int(pt.x), int(pt.y)
            return (0, 0)

        def _mouse_ll_proc(n_code: int, wparam: int, lparam: int) -> int:
            if n_code >= 0 and wparam in (WM_MOUSEWHEEL, WM_MOUSEHWHEEL) and lparam:
                now_ns = time.perf_counter_ns()
                ms = ctypes.cast(ctypes.c_void_p(int(lparam) & 0xFFFFFFFFFFFFFFFF), ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                signed_delta = ctypes.c_short((int(ms.mouseData) >> 16) & 0xFFFF).value / 120.0
                xy = _cursor_xy()
                if wparam == WM_MOUSEWHEEL:
                    self._queue.put(RawInputEvent(now_ns, "wheel", screen_xy=xy, wheel_delta=(0.0, float(signed_delta))))
                else:
                    self._queue.put(RawInputEvent(now_ns, "wheel", screen_xy=xy, wheel_delta=(float(signed_delta), 0.0)))
            return int(user32.CallNextHookEx(wheel_hook[0], n_code, wparam, lparam))

        def _wnd_proc(hwnd: int, msg: int, wparam: int, lparam: int) -> int:
            if msg == WM_INPUT:
                now_ns = time.perf_counter_ns()
                sz = wintypes.UINT(0)
                hdr_sz = ctypes.sizeof(RAWINPUTHEADER)
                h_raw = ctypes.c_void_p(int(lparam) & 0xFFFFFFFFFFFFFFFF)
                user32.GetRawInputData(h_raw, RID_INPUT, None, ctypes.byref(sz), hdr_sz)
                if sz.value > 0:
                    buf = (ctypes.c_ubyte * sz.value)()
                    copied = user32.GetRawInputData(h_raw, RID_INPUT, buf, ctypes.byref(sz), hdr_sz)
                    if copied > 0:
                        raw = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
                        if raw.header.dwType == RIM_TYPEMOUSE:
                            rm = raw.data.mouse
                            flags = int(rm.u.buttons.usButtonFlags)
                            xy = _cursor_xy()
                            if flags & RI_MOUSE_LEFT_BUTTON_DOWN:
                                self._queue.put(RawInputEvent(now_ns, "left_down", screen_xy=xy))
                            if flags & RI_MOUSE_LEFT_BUTTON_UP:
                                self._queue.put(RawInputEvent(now_ns, "left_up", screen_xy=xy))
                            if flags & RI_MOUSE_RIGHT_BUTTON_DOWN:
                                self._queue.put(RawInputEvent(now_ns, "right_down", screen_xy=xy))
                            if flags & RI_MOUSE_RIGHT_BUTTON_UP:
                                self._queue.put(RawInputEvent(now_ns, "right_up", screen_xy=xy))
                            if (flags & RI_MOUSE_WHEEL) and not wheel_hook[0]:
                                signed_delta = ctypes.c_short(rm.u.buttons.usButtonData).value / 120.0
                                self._queue.put(RawInputEvent(now_ns, "wheel", screen_xy=xy, wheel_delta=(0.0, float(signed_delta))))
                            if (flags & RI_MOUSE_HWHEEL) and not wheel_hook[0]:
                                signed_delta = ctypes.c_short(rm.u.buttons.usButtonData).value / 120.0
                                self._queue.put(RawInputEvent(now_ns, "wheel", screen_xy=xy, wheel_delta=(float(signed_delta), 0.0)))
                            if not (flags & 0x0FFF) and (rm.lLastX != 0 or rm.lLastY != 0):
                                self._queue.put(RawInputEvent(now_ns, "mouse_move", screen_xy=xy))
                        elif raw.header.dwType == RIM_TYPEKEYBOARD:
                            rk = raw.data.keyboard
                            vk = int(rk.VKey)
                            is_up = bool(int(rk.Flags) & RI_KEY_BREAK)
                            mod_map = {VK_CONTROL: "ctrl", VK_MENU: "alt", VK_SHIFT: "shift"}
                            if vk in mod_map:
                                mname = mod_map[vk]
                                if is_up:
                                    self._down_mods.discard(mname)
                                else:
                                    self._down_mods.add(mname)
                            elif not is_up:
                                mods = tuple(sorted(self._down_mods))
                                is_hotkey, combo, ch = _vk_to_label_and_char(vk, mods)
                                if combo == "ctrl+shift+f12":
                                    self._queue.put(RawInputEvent(now_ns, "stop", vk_code=vk, modifiers=mods))
                                elif is_hotkey or ch:
                                    self._queue.put(
                                        RawInputEvent(
                                            now_ns,
                                            "key_down",
                                            screen_xy=_cursor_xy(),
                                            vk_code=vk,
                                            char=ch,
                                            modifiers=mods,
                                        )
                                    )
                return 0
            if msg == WM_CLOSE:
                user32.DestroyWindow(hwnd)
                return 0
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
            return int(user32.DefWindowProcW(hwnd, msg, wparam, lparam))

        wnd_proc_cb = WNDPROC(_wnd_proc)
        h_inst = kernel32.GetModuleHandleW(None)
        cls_name = f"AVCSRawInputSink_{uuid.uuid4().hex[:8]}"
        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.lpfnWndProc = wnd_proc_cb
        wc.hInstance = h_inst
        wc.lpszClassName = cls_name

        atom = user32.RegisterClassExW(ctypes.byref(wc))
        if not atom:
            self._error = OSError("RegisterClassExW failed for RawInput sink.")
            self._ready_event.set()
            return

        HWND_MESSAGE = wintypes.HWND(-3)
        hwnd = user32.CreateWindowExW(0, cls_name, None, 0, 0, 0, 0, 0, HWND_MESSAGE, None, h_inst, None)
        if not hwnd:
            user32.UnregisterClassW(cls_name, h_inst)
            self._error = OSError("CreateWindowExW(HWND_MESSAGE) failed for RawInput sink.")
            self._ready_event.set()
            return

        self._hwnd = int(hwnd)
        devices = (RAWINPUTDEVICE * 2)()
        devices[0] = RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, hwnd)  # Mouse
        devices[1] = RAWINPUTDEVICE(0x01, 0x06, RIDEV_INPUTSINK, hwnd)  # Keyboard
        if not user32.RegisterRawInputDevices(devices, 2, ctypes.sizeof(RAWINPUTDEVICE)):
            err_code = ctypes.get_last_error()
            user32.DestroyWindow(hwnd)
            user32.UnregisterClassW(cls_name, h_inst)
            self._error = OSError(f"RegisterRawInputDevices failed (err={err_code}).")
            self._ready_event.set()
            return

        mouse_ll_cb = HOOKPROC(_mouse_ll_proc)
        wheel_hook[0] = user32.SetWindowsHookExW(WH_MOUSE_LL, mouse_ll_cb, None, 0)

        self._ready_event.set()
        msg_struct = wintypes.MSG()
        while not self._stop_event.is_set() and user32.GetMessageW(ctypes.byref(msg_struct), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg_struct))
            user32.DispatchMessageW(ctypes.byref(msg_struct))

        if wheel_hook[0]:
            try:
                user32.UnhookWindowsHookEx(wheel_hook[0])
            except Exception:
                pass
            wheel_hook[0] = None

        try:
            user32.UnregisterClassW(cls_name, h_inst)
        except Exception:
            pass
        self._hwnd = 0


class RawInputActionAssembler:
    """Converts Raw Input events + pre-action frames into canonical `LeRobotV3TrajectoryEpisode` steps."""

    def __init__(
        self,
        *,
        episode_id: str,
        geometry: DisplayGeometry,
        ring_buffer: PreActionFrameRingBuffer,
        config: RecorderConfig,
        defer_jpeg_encode: bool = False,
    ) -> None:
        self.episode_id = episode_id
        self.geometry = geometry
        self.ring_buffer = ring_buffer
        self.config = config
        self.defer_jpeg_encode = bool(defer_jpeg_encode)
        self._steps: List[LeRobotV3TrajectoryEpisode] = []
        self._bound_frames: List[CapturedFrame] = []
        self._prev_bound_frame: Optional[CapturedFrame] = None
        self._last_cursor_norm: Tuple[float, float] = (0.5, 0.5)
        self._last_left_click_ns: int = -1
        self._last_left_click_norm: Tuple[float, float] = (0.5, 0.5)
        self._active_drag: Optional[Dict[str, Any]] = None
        self._last_action_event_ns: int = 0
        self.dropped_unsynced_events: int = 0

    def materialize_frames(self) -> None:
        """Encodes deferred step JPEG payloads cleanly after live recording stops."""
        for step, frame in zip(self._steps, self._bound_frames):
            if not step.image_bytes:
                step.image_bytes = frame.ensure_jpeg_bytes()

    @property
    def steps(self) -> List[LeRobotV3TrajectoryEpisode]:
        return list(self._steps)

    def _norm(self, screen_xy: Tuple[int, int]) -> Tuple[float, float]:
        return self.geometry.to_normalized(int(screen_xy[0]), int(screen_xy[1]))

    def _px_dist(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = (a[0] - b[0]) * max(1, self.geometry.virtual_width)
        dy = (a[1] - b[1]) * max(1, self.geometry.virtual_height)
        return math.hypot(dx, dy)

    def _emit_step(
        self,
        *,
        action_kind: int,
        action_event_ns: int,
        cursor_pre_xy: Tuple[float, float],
        target_xy: Tuple[float, float],
        drag_trajectory_xy: Optional[List[Tuple[float, float]]] = None,
        scroll_delta: Tuple[float, float] = (0.0, 0.0),
        hotkey_ids: Optional[List[int]] = None,
        typed_text: str = "",
        bound_frame: Optional[CapturedFrame] = None,
    ) -> Optional[LeRobotV3TrajectoryEpisode]:
        mono_event_ns = max(int(action_event_ns), self._last_action_event_ns + 1)
        frame = bound_frame or self.ring_buffer.bind_pre_action_frame(
            mono_event_ns,
            min_margin_ns=self.config.min_pre_action_margin_ns,
        )
        if frame is None:
            self.dropped_unsynced_events += 1
            return None

        # Enforce strictly monotonic frame_present_ns across steps
        if self._steps and frame.present_ns < self._steps[-1].frame_present_ns:
            self.dropped_unsynced_events += 1
            return None

        self._last_action_event_ns = mono_event_ns
        diff_ratio = PreActionFrameRingBuffer.compute_pixel_diff_ratio(self._prev_bound_frame, frame)
        self._prev_bound_frame = frame

        step = LeRobotV3TrajectoryEpisode(
            episode_id=self.episode_id,
            step_index=len(self._steps),
            video_frame_index=max(0, frame.seq - 1),
            frame_present_ns=frame.present_ns,
            action_event_ns=mono_event_ns,
            screen_width=frame.width,
            screen_height=frame.height,
            dpi_scale=frame.dpi_scale,
            os_platform=self.config.os_platform,
            task_instruction=self.config.task_instruction,
            cursor_pre_xy=cursor_pre_xy,
            action_kind=action_kind,
            target_xy=target_xy,
            drag_trajectory_xy=drag_trajectory_xy or [],
            scroll_delta=scroll_delta,
            hotkey_ids=hotkey_ids or [],
            typed_text=typed_text,
            pixel_diff_ratio=diff_ratio,
            image_bytes=b"" if self.defer_jpeg_encode else frame.ensure_jpeg_bytes(),
            task_success=True,
        )
        self._steps.append(step)
        self._bound_frames.append(frame)
        return step

    def process_event(self, ev: RawInputEvent) -> Optional[LeRobotV3TrajectoryEpisode]:
        norm_xy = self._norm(ev.screen_xy)
        kind = ev.kind

        if kind == "mouse_move":
            if self._active_drag is not None:
                self._active_drag["points"].append(norm_xy)
                self._last_cursor_norm = norm_xy
                return None
            dist_px = self._px_dist(self._last_cursor_norm, norm_xy)
            if dist_px >= self.config.min_move_px:
                pre_xy = self._last_cursor_norm
                self._last_cursor_norm = norm_xy
                return self._emit_step(
                    action_kind=ACTION_MOVE,
                    action_event_ns=ev.event_ns,
                    cursor_pre_xy=pre_xy,
                    target_xy=norm_xy,
                )
            self._last_cursor_norm = norm_xy
            return None

        if kind == "left_down":
            pre_frame = self.ring_buffer.bind_pre_action_frame(
                ev.event_ns,
                min_margin_ns=self.config.min_pre_action_margin_ns,
            )
            self._active_drag = {
                "start_ns": ev.event_ns,
                "start_xy": norm_xy,
                "pre_xy": self._last_cursor_norm,
                "points": [norm_xy],
                "bound_frame": pre_frame,
            }
            self._last_cursor_norm = norm_xy
            return None

        if kind == "left_up":
            drag_info = self._active_drag
            self._active_drag = None
            if drag_info is None:
                pre_xy = self._last_cursor_norm
                self._last_cursor_norm = norm_xy
                return self._emit_step(
                    action_kind=ACTION_LEFT_CLICK,
                    action_event_ns=ev.event_ns,
                    cursor_pre_xy=pre_xy,
                    target_xy=norm_xy,
                )

            start_xy = drag_info["start_xy"]
            onset_ns = int(drag_info["start_ns"])
            use_bound = drag_info["bound_frame"] if onset_ns > self._last_action_event_ns else None
            emit_ns = onset_ns if use_bound is not None else ev.event_ns
            dist_px = self._px_dist(start_xy, norm_xy)
            self._last_cursor_norm = norm_xy
            if dist_px >= self.config.min_drag_px:
                pts = drag_info["points"]
                if pts[-1] != norm_xy:
                    pts.append(norm_xy)
                return self._emit_step(
                    action_kind=ACTION_DRAG,
                    action_event_ns=emit_ns,
                    cursor_pre_xy=start_xy,
                    target_xy=norm_xy,
                    drag_trajectory_xy=pts,
                    bound_frame=use_bound,
                )

            is_double = (
                self._last_left_click_ns > 0
                and (onset_ns - self._last_left_click_ns) <= self.config.double_click_window_ns
                and self._px_dist(self._last_left_click_norm, norm_xy) <= 4.0
            )
            self._last_left_click_ns = onset_ns
            self._last_left_click_norm = norm_xy
            act_kind = ACTION_DOUBLE_CLICK if is_double else ACTION_LEFT_CLICK
            return self._emit_step(
                action_kind=act_kind,
                action_event_ns=emit_ns,
                cursor_pre_xy=drag_info["pre_xy"],
                target_xy=norm_xy,
                bound_frame=use_bound,
            )

        if kind == "double_click":
            pre_xy = self._last_cursor_norm
            self._last_cursor_norm = norm_xy
            return self._emit_step(
                action_kind=ACTION_DOUBLE_CLICK,
                action_event_ns=ev.event_ns,
                cursor_pre_xy=pre_xy,
                target_xy=norm_xy,
            )

        if kind == "right_down":
            pre_xy = self._last_cursor_norm
            self._last_cursor_norm = norm_xy
            return self._emit_step(
                action_kind=ACTION_RIGHT_CLICK,
                action_event_ns=ev.event_ns,
                cursor_pre_xy=pre_xy,
                target_xy=norm_xy,
            )

        if kind == "wheel":
            pre_xy = self._last_cursor_norm
            return self._emit_step(
                action_kind=ACTION_SCROLL,
                action_event_ns=ev.event_ns,
                cursor_pre_xy=pre_xy,
                target_xy=norm_xy,
                scroll_delta=(float(ev.wheel_delta[0]), float(ev.wheel_delta[1])),
            )

        if kind == "key_down":
            is_hotkey, combo, ch = _vk_to_label_and_char(ev.vk_code, ev.modifiers)
            if ev.char and not ev.modifiers:
                ch = ev.char
                is_hotkey = False
            if is_hotkey:
                return self._emit_step(
                    action_kind=ACTION_HOTKEY,
                    action_event_ns=ev.event_ns,
                    cursor_pre_xy=self._last_cursor_norm,
                    target_xy=self._last_cursor_norm,
                    hotkey_ids=[int(ev.vk_code)],
                    typed_text=combo,
                )
            if ch:
                return self._emit_step(
                    action_kind=ACTION_TYPE,
                    action_event_ns=ev.event_ns,
                    cursor_pre_xy=self._last_cursor_norm,
                    target_xy=self._last_cursor_norm,
                    typed_text=ch,
                )
            return None

        if kind == "stop":
            return self._emit_step(
                action_kind=ACTION_STOP,
                action_event_ns=ev.event_ns,
                cursor_pre_xy=self._last_cursor_norm,
                target_xy=self._last_cursor_norm,
            )

        return None

    def emit_idle_wait(self, event_ns: int) -> Optional[LeRobotV3TrajectoryEpisode]:
        return self._emit_step(
            action_kind=ACTION_WAIT,
            action_event_ns=event_ns,
            cursor_pre_xy=self._last_cursor_norm,
            target_xy=self._last_cursor_norm,
        )


class ZeroLagWindowsRecorder:
    """High-FPS pure-RGB screen capture + Raw Input recorder outputting LeRobot v3.0 shards."""

    def __init__(
        self,
        config: Optional[RecorderConfig] = None,
        *,
        capture_adapter: Optional[BaseCaptureAdapter] = None,
        backend_name: Optional[str] = None,
        event_queue: Optional[queue.Queue[RawInputEvent]] = None,
        use_os_raw_input: Optional[bool] = None,
    ) -> None:
        self.config = config or RecorderConfig()
        if capture_adapter is not None:
            self.backend_name = backend_name or "custom"
            self.capture = capture_adapter.open()
        else:
            self.backend_name, self.capture = select_capture_backend(self.config.backend)
            if self.config.backend == "auto" and self.backend_name != "synthetic":
                self.capture.open()

        self.geometry = self.capture.get_geometry()
        self.ring_buffer = PreActionFrameRingBuffer(
            capacity=self.config.ring_capacity,
            jpeg_quality=self.config.jpeg_quality,
            encode_size=self.config.capture_size,
        )
        self.event_queue: queue.Queue[RawInputEvent] = event_queue or queue.Queue()
        self._use_os_input = (
            (sys.platform == "win32" and self.backend_name != "synthetic")
            if use_os_raw_input is None
            else bool(use_os_raw_input)
        )
        self._raw_listener: Optional[Win32RawInputListener] = None
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._frames_captured = 0

    def _capture_loop(self) -> None:
        interval_s = 1.0 / max(1.0, float(self.config.target_fps))
        has_ts_grab = hasattr(self.capture, "grab_with_timestamp")
        last_seq = -1
        last_preview: Optional[np.ndarray] = None
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                if has_ts_grab:
                    rgb, cap_ns, frame_seq = self.capture.grab_with_timestamp(timeout_s=interval_s)
                    if frame_seq != last_seq:
                        pushed = self.ring_buffer.push_frame(
                            rgb,
                            dpi_scale=self.geometry.dpi_scale,
                            present_ns=cap_ns,
                            copy_rgb=False,
                        )
                        last_seq = frame_seq
                        last_preview = pushed.preview_32x32
                    else:
                        # Static screen heartbeat: keep ring buffer fresh without full-frame copy
                        pushed = self.ring_buffer.push_frame(
                            rgb,
                            dpi_scale=self.geometry.dpi_scale,
                            present_ns=time.perf_counter_ns(),
                            copy_rgb=False,
                            cached_preview=last_preview,
                        )
                        last_preview = pushed.preview_32x32
                    self._frames_captured += 1
                    continue
                rgb = self.capture.grab()
                self.ring_buffer.push_frame(rgb, dpi_scale=self.geometry.dpi_scale)
                self._frames_captured += 1
            except Exception:
                pass
            dt = time.perf_counter() - t0
            sleep_s = interval_s - dt
            if sleep_s > 0.0005:
                self._stop_event.wait(timeout=sleep_s)

    def record_session(
        self,
        *,
        duration_s: float = 10.0,
        output_dir: Optional[Union[str, Path]] = None,
        episode_id: Optional[str] = None,
        clean: bool = True,
        render_audit: bool = False,
        on_step: Optional[Callable[[LeRobotV3TrajectoryEpisode], None]] = None,
    ) -> Dict[str, Any]:
        """Records frames + input events for `duration_s`, curates, QA-audits, and exports shards."""
        ep_id = episode_id or f"ep_{uuid.uuid4().hex[:10]}"
        assembler = RawInputActionAssembler(
            episode_id=ep_id,
            geometry=self.geometry,
            ring_buffer=self.ring_buffer,
            config=self.config,
            defer_jpeg_encode=True,
        )

        winmm = None
        if sys.platform == "win32":
            try:
                winmm = ctypes.windll.winmm
                winmm.timeBeginPeriod(1)
            except Exception:
                winmm = None

        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, name="AVCS-ZeroLagCapture", daemon=True)
        self._capture_thread.start()

        # Wait briefly for at least 1 initial frame before starting input assembly
        wait_deadline = time.perf_counter() + 2.0
        while self.ring_buffer.latest_frame() is None and time.perf_counter() < wait_deadline:
            time.sleep(0.002)

        if self._use_os_input:
            self._raw_listener = Win32RawInputListener(self.event_queue)
            self._raw_listener.start()

        end_ns = time.perf_counter_ns() + int(max(0.05, duration_s) * 1_000_000_000)
        start_ns = time.perf_counter_ns()
        last_activity_ns = start_ns
        stop_requested = False
        raw_events_received = 0

        try:
            while time.perf_counter_ns() < end_ns and not stop_requested:
                try:
                    ev = self.event_queue.get(timeout=0.02)
                    raw_events_received += 1
                    step = assembler.process_event(ev)
                    last_activity_ns = ev.event_ns
                    if step is not None and on_step is not None:
                        on_step(step)
                    if ev.kind == "stop":
                        stop_requested = True
                except queue.Empty:
                    now_ns = time.perf_counter_ns()
                    if (now_ns - last_activity_ns) >= self.config.idle_wait_interval_ns:
                        step = assembler.emit_idle_wait(now_ns)
                        last_activity_ns = now_ns
                        if step is not None and on_step is not None:
                            on_step(step)
        finally:
            if self._raw_listener is not None:
                self._raw_listener.stop()
                self._raw_listener = None
            self._stop_event.set()
            if self._capture_thread is not None:
                self._capture_thread.join(timeout=2.0)
                self._capture_thread = None
            if winmm is not None:
                try:
                    winmm.timeEndPeriod(1)
                except Exception:
                    pass

        elapsed_s = max(1e-6, (time.perf_counter_ns() - start_ns) / 1_000_000_000.0)
        capture_fps = self._frames_captured / elapsed_s

        assembler.materialize_frames()
        raw_steps = assembler.steps
        if not raw_steps:
            # Ensure deterministic single WAIT/STOP terminal frame if zero input arrived
            now_ns = time.perf_counter_ns()
            s = assembler.emit_idle_wait(now_ns)
            if s is not None:
                assembler.materialize_frames()
                raw_steps = assembler.steps

        atomic_syncs = sorted(
            max(0, s.action_event_ns - s.frame_present_ns) / 1_000_000.0
            for s in raw_steps
            if s.action_kind in (ACTION_LEFT_CLICK, ACTION_RIGHT_CLICK, ACTION_DOUBLE_CLICK, ACTION_MOVE, ACTION_SCROLL, ACTION_HOTKEY, ACTION_TYPE)
        )
        atomic_p95_sync_ms = (
            atomic_syncs[int(0.95 * (len(atomic_syncs) - 1))] if atomic_syncs else 0.0
        )

        if self.config.redact_boxes_norm:
            raw_steps = TrajectoryCleaner.redact_pii_regions(raw_steps, self.config.redact_boxes_norm)

        curated_steps = (
            TrajectoryCleaner.clean_trajectory(
                raw_steps,
                retain_recovery_fraction=self.config.retain_recovery_fraction,
            )
            if clean
            else raw_steps
        )
        episodes = [curated_steps] if curated_steps else []
        qa_report = QADoctor.audit_dataset(episodes) if episodes else {
            "total_episodes": 0,
            "passed_episodes": 0,
            "pass_rate": 0.0,
            "p95_sync_error_ms": 0.0,
            "corrupted_episodes": 0,
            "quarantined_issues": {},
        }

        shard_info: Dict[str, Any] = {}
        audit_files: List[str] = []
        if output_dir is not None and episodes:
            out_path = Path(output_dir)
            chunk0 = out_path / "data" / "chunk-000"
            next_file_idx = 0
            if chunk0.is_dir():
                existing = list(chunk0.glob("file-*.parquet")) + list(chunk0.glob("file-*.jsonl"))
                for ef in existing:
                    try:
                        idx_val = int(ef.stem.split("-")[1])
                        next_file_idx = max(next_file_idx, idx_val + 1)
                    except Exception:
                        pass
            shard_info = save_lerobot_v3_shards(episodes, out_path, start_file_index=next_file_idx)
            if render_audit:
                audit_files = [
                    str(p)
                    for p in QADoctor.render_visual_replay_audit(
                        curated_steps,
                        out_path / "audit_frames",
                        max_frames=16,
                    )
                ]
            meta_dir = out_path / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "episode_id": ep_id,
                "capture_backend": self.backend_name,
                "frames_captured": self._frames_captured,
                "capture_fps": round(capture_fps, 2),
                "atomic_p95_sync_ms": round(atomic_p95_sync_ms, 3),
                "raw_events_received": raw_events_received,
                "raw_step_count": len(raw_steps),
                "cleaned_step_count": len(curated_steps),
                "dropped_unsynced_events": assembler.dropped_unsynced_events,
                "qa_report": qa_report,
                "shard_info": shard_info,
                "zero_title_pure_rgb": True,
            }
            (meta_dir / "recording_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "episode_id": ep_id,
            "capture_backend": self.backend_name,
            "frames_captured": self._frames_captured,
            "capture_fps": round(capture_fps, 2),
            "atomic_p95_sync_ms": round(atomic_p95_sync_ms, 3),
            "raw_events_received": raw_events_received,
            "raw_step_count": len(raw_steps),
            "cleaned_step_count": len(curated_steps),
            "dropped_unsynced_events": assembler.dropped_unsynced_events,
            "qa_report": qa_report,
            "shard_info": shard_info,
            "audit_files": audit_files,
            "steps": curated_steps,
        }

    def close(self) -> None:
        self._stop_event.set()
        if self._raw_listener is not None:
            self._raw_listener.stop()
            self._raw_listener = None
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        try:
            self.capture.close()
        except Exception:
            pass


def run_recorder_smoke_test(
    output_dir: Union[str, Path] = "runs/phase2_recorder_smoke",
    episode_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs a deterministic end-to-end zero-lag recorder + Raw Input + LeRobot v3 smoke verification."""
    cfg = RecorderConfig(
        task_instruction="Open notepad and type test greeting",
        backend="synthetic",
        target_fps=240.0,
        min_pre_action_margin_ns=2_000_000,
        idle_wait_interval_ns=2_000_000_000,
        redact_boxes_norm=[(0.0, 0.0, 0.12, 0.06)],
    )
    adapter = InteractiveSyntheticCaptureAdapter(width=1280, height=800)
    ev_queue: queue.Queue[RawInputEvent] = queue.Queue()
    rec = ZeroLagWindowsRecorder(
        cfg,
        capture_adapter=adapter,
        backend_name="synthetic",
        event_queue=ev_queue,
        use_os_raw_input=False,
    )

    def _injector() -> None:
        # Wait slightly so pre-action frames enter ring buffer strictly before inputs
        time.sleep(0.03)
        base = time.perf_counter_ns()
        seq_events = [
            RawInputEvent(base + 12_000_000, "mouse_move", screen_xy=(320, 240)),
            RawInputEvent(base + 24_000_000, "left_down", screen_xy=(320, 240)),
            RawInputEvent(base + 30_000_000, "left_up", screen_xy=(320, 240)),
            RawInputEvent(base + 45_000_000, "key_down", screen_xy=(320, 240), vk_code=ord("H"), char="h"),
            RawInputEvent(base + 55_000_000, "key_down", screen_xy=(320, 240), vk_code=ord("I"), char="i"),
            RawInputEvent(base + 70_000_000, "wheel", screen_xy=(320, 240), wheel_delta=(0.0, -2.0)),
            RawInputEvent(base + 85_000_000, "left_down", screen_xy=(400, 300)),
            RawInputEvent(base + 95_000_000, "mouse_move", screen_xy=(500, 360)),
            RawInputEvent(base + 105_000_000, "left_up", screen_xy=(600, 400)),
            RawInputEvent(base + 120_000_000, "stop", screen_xy=(600, 400)),
        ]
        for ev in seq_events:
            while time.perf_counter_ns() < ev.event_ns:
                time.sleep(0.001)
            adapter.trigger_motion(0.5, 0.5)
            ev_queue.put(ev)

    inj = threading.Thread(target=_injector, daemon=True)
    inj.start()
    try:
        res = rec.record_session(
            duration_s=1.0,
            output_dir=output_dir,
            episode_id=episode_id or f"smoke_win32_{uuid.uuid4().hex[:6]}",
            clean=True,
            render_audit=True,
        )
    finally:
        inj.join(timeout=1.0)
        rec.close()

    reloaded = load_lerobot_v3_shards(output_dir)
    res["reloaded_episodes"] = len(reloaded)
    res["reloaded_steps"] = sum(len(ep) for ep in reloaded)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Zero-Lag Windows Native Recorder (Pure-RGB -> LeRobot v3)")
    parser.add_argument("--smoke-test", action="store_true", help="Run deterministic synthetic zero-lag recorder smoke check")
    parser.add_argument("--record", action="store_true", help="Start live Windows desktop Raw Input + pure-RGB capture session")
    parser.add_argument("--task", type=str, default="Perform desktop task", help="Task instruction attached to recorded steps")
    parser.add_argument("--duration", type=float, default=15.0, help="Recording duration in seconds (Ctrl+Shift+F12 stops early)")
    parser.add_argument("--fps", type=float, default=120.0, help="Target screen capture polling FPS (default: 120)")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "wgc", "dxgi", "gdi", "mss", "synthetic"])
    parser.add_argument("--output", type=str, default="runs/recordings/session_01", help="Output LeRobot v3 shard directory")
    parser.add_argument("--render-audit", action="store_true", help="Render spot-audit JPEG frames with click markers")
    args = parser.parse_args()

    if args.smoke_test:
        res = run_recorder_smoke_test(output_dir=args.output if "--output" in sys.argv else "runs/phase2_recorder_smoke")
        qa = res["qa_report"]
        print(
            f"PHASE2-RECORDER-SMOKE: pass_rate={qa['pass_rate']:.2f}, "
            f"capture_fps={res['capture_fps']:.1f}, "
            f"atomic_p95_sync_ms={res['atomic_p95_sync_ms']:.2f}ms, "
            f"p95_sync_error_ms={qa['p95_sync_error_ms']:.2f}ms, "
            f"raw_steps={res['raw_step_count']}, cleaned_steps={res['cleaned_step_count']}, "
            f"reloaded_steps={res['reloaded_steps']}"
        )
        return

    if args.record:
        cfg = RecorderConfig(
            task_instruction=args.task,
            backend=args.backend,
            target_fps=args.fps,
        )
        rec = ZeroLagWindowsRecorder(cfg)
        print(
            f"Recording pure-RGB ({rec.backend_name}) + Raw Input for {args.duration:.1f}s "
            f"(press Ctrl+Shift+F12 to stop early)..."
        )
        try:
            res = rec.record_session(
                duration_s=args.duration,
                output_dir=args.output,
                clean=True,
                render_audit=args.render_audit,
            )
        finally:
            rec.close()
        qa = res["qa_report"]
        print(
            f"Recording saved -> {args.output} | format={res['shard_info'].get('format', 'unknown')} | "
            f"backend={res['capture_backend']} | capture_fps={res['capture_fps']:.1f} | "
            f"raw_events={res['raw_events_received']} | cleaned_steps={res['cleaned_step_count']} | "
            f"pass_rate={qa['pass_rate']:.2f} | qa_p95_sync={qa['p95_sync_error_ms']:.2f}ms | "
            f"non_wait_p95={qa.get('non_wait_p95_sync_error_ms', 0.0):.2f}ms | "
            f"atomic_p95_sync={res['atomic_p95_sync_ms']:.2f}ms | "
            f"sync_under_5ms={qa.get('sync_within_5ms', False)}"
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
