"""CLI and high-level entrypoint for AVCS observation-only desktop assessment.

Provides clean natural-language synthesis of pure-pixel perception combined
with labeled OS window metadata. Zero motor actions authorized.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from forge.vision.desktop_context import Win32DesktopContextProvider
from forge.vision.pipeline import ScreenVision


def synthesize_natural_language_report(obs: dict[str, Any]) -> str:
    """Transforms a structured desktop observation into a grounded narrative summary."""
    lines: list[str] = []

    # 1. Desktop Window Environment
    desktop = obs.get("desktop_context", {})
    if desktop.get("status") == "ok":
        count = desktop.get("count", 0)
        windows = desktop.get("windows", [])
        fg_window = next((w for w in windows if w.get("is_foreground")), None)
        if fg_window:
            fg_title = fg_window.get("title") or "Untitled Window"
            proc = fg_window.get("process_name") or "unknown process"
            lines.append(
                f"{count} visible window(s) open. Foreground window: '{fg_title}' ({proc})."
            )
        else:
            lines.append(f"{count} visible window(s) open. No distinct foreground window detected.")
    else:
        lines.append("OS window context: not available or not configured.")

    # 2. Visual Perception & Motion
    perception = obs.get("visual_perception", {})
    desc = perception.get("description", "No description available.")
    signals = perception.get("signals", {})
    peak_motion = signals.get("peak_motion", 0.0)
    changed_area = signals.get("changed_area_fraction", 0.0)

    lines.append(f"Visual perception: {desc}")

    # 3. Activity Assessment (Calibrated, Non-Authoritative)
    if peak_motion > 0.15 and changed_area > 0.10:
        lines.append(
            "Visual activity assessment: Continuous high visual motion detected across a large area. "
            "Suggests video playback or active gameplay (exact activity unconfirmed)."
        )
    elif peak_motion > 0.03 or changed_area > 0.02:
        lines.append(
            "Visual activity assessment: Subtle localized motion detected (e.g. typing, scrolling, or minor animation)."
        )
    else:
        lines.append(
            "Visual activity assessment: Screen is essentially still. Static document, code editor, or idle desktop."
        )

    # 4. Safety & Governance Note
    lines.append("Safety: Observation-only mode. Motor actions authorized: False.")

    return "\n".join(lines)


def capture_desktop_observation(
    *,
    include_os_context: bool = True,
    wait_timeout_s: float = 2.0,
) -> dict[str, Any]:
    """Captures a fresh real-time observation of the current desktop screen."""
    desktop_ctx = None
    if include_os_context and sys.platform == "win32":
        try:
            desktop_ctx = Win32DesktopContextProvider()
        except OSError:
            desktop_ctx = None

    vision = ScreenVision(desktop_context=desktop_ctx)
    return vision.observe(wait_timeout_s=wait_timeout_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Observe the current desktop state using AVCS pure-pixel vision and labeled OS context."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw machine-readable JSON instead of natural language narrative.",
    )
    parser.add_argument(
        "--no-os-context",
        action="store_true",
        help="Omit Win32 window metadata; report pure pixel perception only.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Maximum seconds to wait for frame analysis settling (default: 2.0s).",
    )
    args = parser.parse_args(argv)

    try:
        obs = capture_desktop_observation(
            include_os_context=not args.no_os_context,
            wait_timeout_s=args.timeout,
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"Observation failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(obs, indent=2))
    else:
        print("=" * 70)
        print("AVCS // DESKTOP OBSERVATION SUMMARY")
        print("=" * 70)
        print(synthesize_natural_language_report(obs))
        print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
