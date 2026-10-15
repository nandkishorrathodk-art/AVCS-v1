"""Optional operating-system desktop context, separate from pixel perception."""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DesktopWindow:
    handle: int
    title: str
    process_id: int
    process_name: str | None
    process_name_status: str
    bounds: tuple[int, int, int, int]
    is_foreground: bool

    def to_dict(self) -> dict:
        return {
            "handle": self.handle,
            "title": self.title,
            "process_id": self.process_id,
            "process_name": self.process_name,
            "process_name_status": self.process_name_status,
            "bounds": list(self.bounds),
            "is_foreground": self.is_foreground,
        }


class DesktopContextProvider(Protocol):
    """Supplies OS-derived context; it is not visual perception."""

    def snapshot(self) -> dict: ...


class Win32DesktopContextProvider:
    """Enumerates visible, titled top-level windows on Windows."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("Win32 desktop context is only available on Windows.")

        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self._user32.IsWindowVisible.restype = wintypes.BOOL
        self._user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self._user32.GetWindowTextLengthW.restype = ctypes.c_int
        self._user32.GetWindowTextW.argtypes = [
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        ]
        self._user32.GetWindowTextW.restype = ctypes.c_int
        self._user32.GetWindowRect.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.RECT),
        ]
        self._user32.GetWindowRect.restype = ctypes.wintypes.BOOL
        self._user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        callback_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HWND,
            wintypes.LPARAM,
        )
        self._callback_type = callback_type
        self._user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
        self._user32.EnumWindows.restype = wintypes.BOOL

    def _window_title(self, handle: int) -> str:
        length = int(self._user32.GetWindowTextLengthW(handle))
        if length <= 0:
            return ""
        title = ctypes.create_unicode_buffer(length + 1)
        self._user32.GetWindowTextW(handle, title, length + 1)
        return title.value

    def _process_name(self, process_id: int) -> tuple[str | None, str]:
        process = self._kernel32.OpenProcess(0x1000, False, process_id)
        if not process:
            return None, "unavailable"
        try:
            image_path = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(image_path))
            if not self._kernel32.QueryFullProcessImageNameW(
                process, 0, image_path, ctypes.byref(size)
            ):
                return None, "unavailable"
            return os.path.basename(image_path.value), "ok"
        finally:
            self._kernel32.CloseHandle(process)

    def snapshot(self) -> dict:
        foreground = int(self._user32.GetForegroundWindow() or 0)
        windows: list[DesktopWindow] = []
        skipped_during_enumeration = 0

        def visit(handle: int, _lparam: int) -> bool:
            nonlocal skipped_during_enumeration
            if not self._user32.IsWindowVisible(handle):
                return True
            title = self._window_title(handle)
            if not title:
                return True

            rect = wintypes.RECT()
            if not self._user32.GetWindowRect(handle, ctypes.byref(rect)):
                skipped_during_enumeration += 1
                return True

            process_id = wintypes.DWORD()
            self._user32.GetWindowThreadProcessId(handle, ctypes.byref(process_id))
            process_name, process_name_status = self._process_name(
                int(process_id.value)
            )
            windows.append(
                DesktopWindow(
                    handle=int(handle),
                    title=title,
                    process_id=int(process_id.value),
                    process_name=process_name,
                    process_name_status=process_name_status,
                    bounds=(rect.left, rect.top, rect.right, rect.bottom),
                    is_foreground=int(handle) == foreground,
                )
            )
            return True

        callback = self._callback_type(visit)
        if not self._user32.EnumWindows(callback, 0):
            error = ctypes.get_last_error()
            raise OSError(error, "Win32 EnumWindows failed.")

        return {
            "source": "win32_user32",
            "status": "ok",
            "count": len(windows),
            "count_semantics": "visible_titled_top_level_windows",
            "foreground_handle": foreground or None,
            "windows": [window.to_dict() for window in windows],
            "windows_skipped_during_enumeration": skipped_during_enumeration,
        }
