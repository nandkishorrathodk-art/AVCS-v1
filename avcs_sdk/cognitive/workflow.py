"""Cross-Application Workflow Coordination for AVCS System 2.

Manages multi-window desktop workflows across abstract application roles
(editors, data tables, terminals, viewers, file managers) without hardcoding
any proprietary application names. Supports window context tracking, focus
switching primitives, and cross-application reference data transfer.

Invariants:
- Pure-pixel security: zero OCR, zero DOM, zero UIA.
- Application agnosticism: zero hardcoded proprietary application names.
- Pure ASCII: 100% pure ASCII characters.
- Coordinate privacy: zero coordinate leakage into prompts or state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ApplicationRole(str, Enum):
    """Abstract application categories for desktop workflow coordination."""
    EDITOR = "editor"
    DATA_TABLE = "data_table"
    TERMINAL = "terminal"
    VIEWER = "viewer"
    FILE_MANAGER = "file_manager"
    UNKNOWN = "unknown"


@dataclass
class WindowContext:
    """State and role metadata for a desktop application window."""
    title: str
    role: ApplicationRole
    handle: Optional[int] = None
    is_focused: bool = False
    last_seen_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "role": self.role.value if isinstance(self.role, ApplicationRole) else str(self.role),
            "handle": self.handle,
            "is_focused": self.is_focused,
            "last_seen_time": self.last_seen_time,
        }


@dataclass
class DataTransferBuffer:
    """Structured shared buffer for cross-application reference data exchange."""
    content: str
    source_role: ApplicationRole
    target_role: Optional[ApplicationRole] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowCoordinator:
    """Coordinates multi-window workflows across abstract desktop application roles.

    Maintains active desktop window contexts, infers abstract roles from semantic
    window headers, provides switching primitives, and brokers cross-application
    reference data transfer.
    """

    def __init__(self) -> None:
        self.windows: Dict[str, WindowContext] = {}
        self.focused_window_title: Optional[str] = None
        self.transfer_buffer: Optional[DataTransferBuffer] = None

    @staticmethod
    def infer_role(window_title: str) -> ApplicationRole:
        """Infer an abstract application role from semantic title keywords.

        Zero proprietary application names are checked; uses strictly generic
        functional descriptors.
        """
        title_lower = window_title.lower().strip()

        # Data table / tabular analysis indicators
        if any(kw in title_lower for kw in ("table", "sheet", "csv", "tsv", "grid", "calc", "dataset")):
            return ApplicationRole.DATA_TABLE

        # Terminal / shell / console indicators
        if any(kw in title_lower for kw in ("terminal", "bash", "cmd", "shell", "powershell", "console", "cli", "prompt")):
            return ApplicationRole.TERMINAL

        # Directory / file manager indicators
        if any(kw in title_lower for kw in ("folder", "dir", "directory", "explorer", "files", "file manager", "storage")):
            return ApplicationRole.FILE_MANAGER

        # Editor / code / source document indicators
        if any(kw in title_lower for kw in (
            "edit", "code", "workspace", "document", "source", "script",
            ".py", ".rs", ".js", ".ts", ".cpp", ".c", ".h", ".json", ".md", ".txt", ".yaml", ".html",
        )):
            return ApplicationRole.EDITOR

        # Document / asset viewer indicators
        if any(kw in title_lower for kw in ("pdf", "view", "preview", "reader", "image", "media")):
            return ApplicationRole.VIEWER

        return ApplicationRole.UNKNOWN

    def register_window(
        self,
        title: str,
        role: Optional[ApplicationRole | str] = None,
        handle: Optional[int] = None,
    ) -> WindowContext:
        """Register or update a desktop application window context."""
        if role is None:
            assigned_role = self.infer_role(title)
        elif isinstance(role, ApplicationRole):
            assigned_role = role
        else:
            try:
                assigned_role = ApplicationRole(role.lower())
            except ValueError:
                assigned_role = ApplicationRole.UNKNOWN

        context = WindowContext(
            title=title,
            role=assigned_role,
            handle=handle,
            last_seen_time=time.time(),
        )
        self.windows[title] = context
        return context

    def set_focused_context(self, title_or_role: str) -> Optional[WindowContext]:
        """Update focused window context by matching window title or abstract role."""
        target_context: Optional[WindowContext] = None

        # Direct title match
        if title_or_role in self.windows:
            target_context = self.windows[title_or_role]
        else:
            # Role-based match
            for win in self.windows.values():
                if win.role.value == title_or_role.lower():
                    target_context = win
                    break

        if target_context:
            for win in self.windows.values():
                win.is_focused = (win.title == target_context.title)
            self.focused_window_title = target_context.title
            target_context.last_seen_time = time.time()
            return target_context

        return None

    def get_focused_context(self) -> Optional[WindowContext]:
        """Retrieve current foreground application window context."""
        if self.focused_window_title and self.focused_window_title in self.windows:
            return self.windows[self.focused_window_title]
        return None

    def switch_window(self, target_role_or_title: str) -> Dict[str, Any]:
        """Generate window switching directive to bring target application to foreground."""
        target_role = target_role_or_title.lower()
        matched_window: Optional[WindowContext] = None

        if target_role_or_title in self.windows:
            matched_window = self.windows[target_role_or_title]
        else:
            for win in self.windows.values():
                if win.role.value == target_role:
                    matched_window = win
                    break

        if matched_window:
            self.set_focused_context(matched_window.title)
            return {
                "action": "switch_app",
                "method": "focus_target",
                "target_title": matched_window.title,
                "target_role": matched_window.role.value,
                "keys": ["alt", "tab"],
            }

        # Fallback to general application switch hotkey
        return {
            "action": "switch_app",
            "method": "hotkey",
            "target_role": target_role,
            "keys": ["alt", "tab"],
        }

    def transfer_data(
        self,
        source_role: ApplicationRole | str,
        target_role: Optional[ApplicationRole | str],
        data: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataTransferBuffer:
        """Store reference data in the shared buffer for cross-application paste/injection."""
        src = source_role if isinstance(source_role, ApplicationRole) else ApplicationRole(source_role.lower())
        tgt = None
        if target_role:
            tgt = target_role if isinstance(target_role, ApplicationRole) else ApplicationRole(target_role.lower())

        self.transfer_buffer = DataTransferBuffer(
            content=data,
            source_role=src,
            target_role=tgt,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        return self.transfer_buffer

    def get_transfer_data(self, requesting_role: Optional[ApplicationRole | str] = None) -> Optional[str]:
        """Retrieve shared reference data if destination role matches or is open."""
        if not self.transfer_buffer:
            return None

        if requesting_role is None or self.transfer_buffer.target_role is None:
            return self.transfer_buffer.content

        req = requesting_role if isinstance(requesting_role, ApplicationRole) else ApplicationRole(requesting_role.lower())
        if self.transfer_buffer.target_role == req:
            return self.transfer_buffer.content

        return None

    def list_windows(self) -> List[WindowContext]:
        """Return list of all registered window contexts."""
        return list(self.windows.values())

    def get_windows_by_role(self, role: ApplicationRole | str) -> List[WindowContext]:
        """Retrieve all registered windows assigned to a specific abstract role."""
        target_role = role.value if isinstance(role, ApplicationRole) else str(role).lower()
        return [win for win in self.windows.values() if win.role.value == target_role]
