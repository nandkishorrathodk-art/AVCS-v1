"""Legacy shim for avcs.ace.actions.

Re-exports action schemas and adapters from the canonical avcs.protocol package.
"""
from __future__ import annotations

import avcs.protocol.actions as _actions
from avcs.protocol.actions import *  # noqa: F401, F403
from avcs.protocol.actions import (
    Action,
    ActionAdapter,
    ActionKind,
    HID_CODE_TO_KEY_TABLE,
    HID_KEYBOARD_TABLE,
    HID_MOUSE_BUTTON_TABLE,
    denormalize_coordinate,
    hid_to_key,
    interpolate_drag_chunks,
    key_to_hid,
    mouse_button_to_hid,
    normalize_coordinate,
)

__all__ = [
    "Action",
    "ActionAdapter",
    "ActionKind",
    "HID_CODE_TO_KEY_TABLE",
    "HID_KEYBOARD_TABLE",
    "HID_MOUSE_BUTTON_TABLE",
    "denormalize_coordinate",
    "hid_to_key",
    "interpolate_drag_chunks",
    "key_to_hid",
    "mouse_button_to_hid",
    "normalize_coordinate",
]


def __getattr__(name: str):
    return getattr(_actions, name)

