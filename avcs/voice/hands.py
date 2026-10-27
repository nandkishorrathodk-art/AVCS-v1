"""One place that turns a gated action intent into a real hand operation.

The assistant used to call ``hand.click(*point)`` for *every* allowed action,
so a spoken "type this" or "scroll down" still produced a click. This module is
the single dispatch point: the op decides which hand method runs, and an op with
no implementation fails closed instead of silently becoming a click.

The gate is upstream and unchanged: nothing here runs until
``CapabilityKernel.authorize`` has returned ``allowed=True`` and supplied a
grounded coordinate.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

# Ops that change machine state; everything else is free to observe.
MUTATING_OPS = frozenset({"click", "double_click", "type", "type_text", "press_key", "scroll"})


@dataclasses.dataclass
class HandAction:
    """The result of one dispatched operation, with honest provenance."""

    op: str
    performed: bool
    point: Optional[tuple[int, int]] = None
    detail: str = ""
    error: Optional[str] = None


def _resolve(obj: Any, *names: str) -> Optional[Callable[..., Any]]:
    """First callable attribute among ``names``, or None."""
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn
    return None


def _normalized_to_pixels(hand: Any, point: tuple[float, float]) -> tuple[int, int]:
    """Map a normalized (0..1) point to the hand's pixel space.

    A `VirtualHand` records normalized coordinates; a `HandMotor` needs absolute
    screen pixels. `get_screen_size` tells them apart without a hard dependency
    on either class.
    """
    size = getattr(hand, "get_screen_size", None)
    if callable(size):
        try:
            w, h = size()
            return int(round(point[0] * (w - 1))), int(round(point[1] * (h - 1)))
        except Exception:
            pass
    w = getattr(hand, "width", None)
    h = getattr(hand, "height", None)
    if w and h:
        return int(round(point[0] * (w - 1))), int(round(point[1] * (h - 1)))
    return int(round(point[0])), int(round(point[1]))


def _press_hotkey(hand: Any, key_name: str) -> Optional[bool]:
    """Send a named key or a ``ctrl+s`` style chord through a HandMotor.

    `HandMotor` has no public press method, so this uses the same ctypes binding
    the motor already holds. Returns None when the platform or binding cannot do
    it, so the caller reports "not performed" rather than a false success.
    """
    keybd = getattr(hand, "_keybd_event", None)
    vk_scan = getattr(hand, "_VkKeyScanW", None)
    if keybd is None or vk_scan is None:
        return None

    VK_SHIFT, VK_CONTROL, VK_MENU = 0x10, 0x11, 0x12
    KEYEVENTF_KEYUP = 0x0002
    modifiers = {"ctrl": VK_CONTROL, "shift": VK_SHIFT, "alt": VK_MENU}
    aliases = {
        "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
        "space": 0x20, "backspace": 0x08, "delete": 0x2E, "up": 0x26,
        "down": 0x28, "left": 0x25, "right": 0x27, "home": 0x24, "end": 0x23,
    }

    parts = [p.strip().lower() for p in str(key_name).split("+") if p.strip()]
    if not parts:
        return None
    held = []
    try:
        for mod in parts[:-1]:
            vk = modifiers.get(mod)
            if vk is None:
                return None
            keybd(vk, 0, 0, 0)
            held.append(vk)
        base = parts[-1]
        vk = aliases.get(base)
        if vk is None:
            if len(base) != 1:
                return None
            code = vk_scan(ord(base))
            if code == -1:
                return None
            vk = code & 0xFF
            if (code >> 8) & 1:
                keybd(VK_SHIFT, 0, 0, 0)
                held.append(VK_SHIFT)
        keybd(vk, 0, 0, 0)
        keybd(vk, 0, KEYEVENTF_KEYUP, 0)
        return True
    finally:
        for vk in reversed(held):
            keybd(vk, 0, KEYEVENTF_KEYUP, 0)


def dispatch(
    hand: Any,
    op: str,
    *,
    point: Optional[tuple[float, float]] = None,
    text: Optional[str] = None,
    key_name: Optional[str] = None,
    scroll_delta: Optional[int] = None,
) -> HandAction:
    """Run the hand method the op names. Fail closed on anything else."""
    clean = (op or "").strip().lower()
    if clean == "type":
        clean = "type_text"

    if clean in ("click", "double_click"):
        if point is None:
            return HandAction(clean, False, error="no grounded coordinate for click")
        click = _resolve(hand, "click")
        if click is None:
            return HandAction(clean, False, error="hand has no click()")
        # Two hand contracts exist: `VirtualHand` records normalized (0..1)
        # coordinates (OneModelLoop relies on that), while `HandMotor` drives the
        # native cursor in absolute pixels. `get_screen_size` marks the latter.
        if callable(getattr(hand, "get_screen_size", None)):
            arg = _normalized_to_pixels(hand, point)
        else:
            arg = (float(point[0]), float(point[1]))
        try:
            ret = click(arg[0], arg[1])
        except TypeError:
            ret = click(*point)
        if clean == "double_click":
            ret = click(arg[0], arg[1])
        if ret is False:
            return HandAction(clean, False, error="hand reported the click did not happen")
        px = ret if isinstance(ret, tuple) else arg
        return HandAction(clean, True, point=px)

    if clean == "type_text":
        if not text:
            return HandAction(clean, False, error="no text to type")
        fn = _resolve(hand, "type_text")
        if fn is None:
            return HandAction(clean, False, error="hand has no type_text()")
        try:
            ok = fn(text)
        except TypeError:
            ok = fn(text, 25.0)
        if ok is False:
            return HandAction(clean, False, error="hand reported the text was not typed")
        return HandAction(clean, True, detail=f"{len(text)} chars")

    if clean == "press_key":
        if not key_name:
            return HandAction(clean, False, error="no key name given")
        fn = _resolve(hand, "press_key")
        if fn is not None:
            if fn(key_name) is False:
                return HandAction(clean, False, error=f"hand reported key '{key_name}' was not sent")
            return HandAction(clean, True, detail=key_name)
        if _press_hotkey(hand, key_name):
            return HandAction(clean, True, detail=key_name)
        return HandAction(clean, False, error=f"cannot send key '{key_name}' on this hand")

    if clean == "scroll":
        delta = scroll_delta if scroll_delta is not None else -3
        fn = _resolve(hand, "scroll")
        if fn is None:
            return HandAction(clean, False, error="hand has no scroll()")
        if fn(int(delta)) is False:
            return HandAction(clean, False, error="hand reported the scroll did not happen")
        return HandAction(clean, True, detail=str(int(delta)))

    return HandAction(clean, False, error=f"op '{clean}' has no hand implementation")
