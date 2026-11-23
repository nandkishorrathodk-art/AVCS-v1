"""AVCS Protocol Department.

Provides standardized wire protocols, action definitions, coordinate
normalization, and HID code mapping.
"""
from __future__ import annotations

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
