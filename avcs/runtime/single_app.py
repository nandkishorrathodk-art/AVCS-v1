"""Live, operator-mediated single-application observation and System 2 loop."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from typing import Callable, Optional

from avcs.control.controller import AVCSController
from avcs.server.system2_client import TextOnlySystem2Client


def run_single_app_loop(
    app_context: str,
    task: str,
    client: TextOnlySystem2Client,
    *,
    controller_factory: Callable[..., AVCSController] = AVCSController,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Run fresh live observations and text-only reasoning, requiring consent each turn.

    The app name is operator-provided context, not an app identity recognized
    from pixels. System 2 goals remain advisory; this loop sends no input.
    """
    app_context = app_context.strip()
    task = task.strip()
    if not app_context:
        raise ValueError("app_context must not be empty")
    if not task:
        raise ValueError("task must not be empty")

    controller = controller_factory(
        source="desktop",
        device="cpu",
        force_virtual_motor=True,
        enable_streaming=False,
        require_real_capture=True,
    )
    try:
        output_fn(f"Live capture backend: {controller.capture_backend}")
        output_fn(f"App context (operator-provided): {app_context}")
        output_fn("No synthetic fallback. Mouse and keyboard dispatch are disabled.")

        system2_task = (
            f"Operator-provided app context: {app_context}\n"
            f"User task: {task}"
        )
        grab_with_sequence = getattr(
            getattr(controller, "source", None), "grab_with_sequence", None
        )
        if not callable(grab_with_sequence):
            raise TypeError(
                "Live desktop source must provide sequence-aware capture; refusing "
                "to treat an unsequenced or stale frame as fresh."
            )
        last_frame_sequence: Optional[int] = None  # noqa: UP045
        while True:
            deadline = time.monotonic() + 2.0
            while True:
                frame, frame_sequence = grab_with_sequence()
                if (
                    isinstance(frame_sequence, bool)
                    or not isinstance(frame_sequence, int)
                    or frame_sequence < 0
                ):
                    raise RuntimeError(
                        "Live capture returned an invalid frame sequence; refusing "
                        "to reason from unverified frame provenance."
                    )
                if last_frame_sequence is None or frame_sequence > last_frame_sequence:
                    break
                if frame_sequence < last_frame_sequence:
                    raise RuntimeError(
                        "Live capture sequence moved backwards; refusing stale input."
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Live capture did not produce a fresh frame within 2 seconds."
                    )
                time.sleep(0.01)

            observation = controller.describe_screen(
                frame=frame,
                frame_sequence=frame_sequence,
            )
            last_frame_sequence = frame_sequence
            payload = client.prepare_payload(system2_task, observation)
            output_fn(
                "Exact redacted, allowlisted payload for System 2 review:\n"
                + json.dumps(payload, ensure_ascii=True, sort_keys=True)
            )
            output_fn(
                "Only the task text and bounded text observation will be sent; "
                "no screenshot, frame, crop, or embedding is sent."
            )
            consent = input_fn(
                "Send this turn to System 2? Review the task and observation above "
                "[y/N]: "
            )
            if consent.strip().lower() not in {"y", "yes"}:
                output_fn("Not sent. Ending the live loop.")
                return 0

            goal = client.request_goal(
                payload["task"],
                payload["observation"],
                consent_confirmed=True,
            )
            output_fn(f"System 2 advisory goal: {goal}")
            output_fn(
                "Advisory only: AVCS did not click, type, or otherwise dispatch input."
            )
            next_turn = input_fn(
                "After you manually perform an approved step, press Enter for a "
                "fresh observation, or enter q to finish: "
            )
            if next_turn.strip().lower() in {"q", "quit", "exit"}:
                return 0
    finally:
        controller.close()


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: UP045
    parser = argparse.ArgumentParser(
        description=(
            "Run a live desktop observation-to-text-only System 2 advisory loop. "
            "This command does not automate mouse or keyboard input."
        )
    )
    parser.add_argument("--app", required=True, help="Operator-provided single-app context")
    parser.add_argument("--task", required=True, help="The task to discuss with System 2")
    parser.add_argument(
        "--system2-url",
        default=os.environ.get("AVCS_SYSTEM2_ENDPOINT"),
        help="Text-only bridge URL; defaults to AVCS_SYSTEM2_ENDPOINT",
    )
    args = parser.parse_args(argv)
    if not args.system2_url:
        parser.error(
            "provide --system2-url or set AVCS_SYSTEM2_ENDPOINT "
            "to the existing text-only System 2 bridge"
        )

    api_key = os.environ.get("AVCS_SYSTEM2_API_KEY")
    client = TextOnlySystem2Client(
        args.system2_url,
        api_key=api_key,
    )
    return run_single_app_loop(args.app, args.task, client)


if __name__ == "__main__":
    raise SystemExit(main())
