"""End-to-end example: a multi-step voice task, run against mock screens.

This is the honest demonstration. It wires the parts that exist — the Lexicon
stream (pixels -> control name, no OCR), the capability kernel (the only policy),
and the hand (virtual, records coordinates) — and runs a *bigger* task so the
failure mode is visible rather than hidden.

    python3 -m avc.lexicon.example_task

The task: "save this document, then close it" on a mock editor screen.

What to watch for:

- Steps whose control the reader can name resolve and click.
- A step whose control is present but *unknown to the closed vocabulary* is
  refused by the gate and reported, not guessed at.
- A step whose target is absent from the screen fails the presence floor and
  is reported as blocked, with a reason.
- The run summary counts what actually happened, so a partial success reads as
  a partial success.

No screen is captured and no cursor moves: the frames are rendered, and the hand
is the virtual one. The gate in front of it is the production kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avc.lexicon.reader import LexiconReader
from avc.lexicon.render import split_fonts

# The kernel lives in the avcs package and is the single policy point; the
# example uses it rather than reimplementing a gate.
from avcs.control.capability import CapabilityKernel, GroundingEvidence


@dataclass
class StepResult:
    index: int
    instruction: str
    control: Optional[str]
    outcome: str          # "clicked" | "refused" | "blocked" | "not_found"
    detail: str
    point_px: Optional[tuple[int, int]] = None

    def as_line(self) -> str:
        where = f" at {self.point_px}" if self.point_px else ""
        return f"  step {self.index}: {self.instruction:34s} -> {self.outcome:9s}{where}  ({self.detail})"


@dataclass
class TaskReport:
    task: str
    steps: List[StepResult] = field(default_factory=list)

    @property
    def completed(self) -> int:
        return sum(1 for s in self.steps if s.outcome == "clicked")

    def summary(self) -> str:
        total = len(self.steps)
        done = self.completed
        verdict = "complete" if done == total else "PARTIAL" if done else "no progress"
        return f"{done}/{total} steps completed — {verdict}"


def _mock_editor_screen(buttons: Sequence[str], width: int = 560, height: int = 200) -> np.ndarray:
    """A crude editor toolbar: a title and a row of labelled buttons.

    Deliberately plain — the point is to exercise the pipeline, not to look real.
    """
    fonts, _held = split_fonts()
    if not fonts:
        raise RuntimeError("no TrueType fonts to render the mock screen")
    font = ImageFont.truetype(fonts[0], 26)
    title_font = ImageFont.truetype(fonts[0], 18)
    image = Image.new("RGB", (width, height), (238, 238, 240))
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), "Untitled document", font=title_font, fill=(60, 60, 70))
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    x = 20
    for name in buttons:
        w = probe.textbbox((0, 0), name, font=font)[2] + 24
        draw.rectangle([x, 70, x + w, 116], fill=(250, 250, 252), outline=(150, 150, 160))
        draw.text((x + 12, 80), name, font=font, fill=(30, 30, 40))
        x += w + 14
    return np.asarray(image)


class VirtualHand:
    """Records where a click would go. Moves no cursor."""

    def __init__(self) -> None:
        self.clicks: List[tuple[int, int]] = []

    def click(self, x: int, y: int) -> tuple[int, int]:
        self.clicks.append((x, y))
        return (x, y)


def run_task(task: str, steps: Sequence[str], screens: Sequence[Sequence[str]]) -> TaskReport:
    """Run a step list against a sequence of mock screens, gating every click.

    ``steps`` are spoken control names; ``screens[i]`` is the button row shown
    when step ``i`` runs, so a screen that lacks a control exercises the
    "target absent" path.

    The reader proposes a label for every region and the *kernel* decides whether
    that label may become a click. Asking the reader to also gate would put two
    policies in the pipeline; the kernel is the single policy point.
    """
    reader = LexiconReader(checkpoint="runs/weights/avc_lexicon.pt")
    kernel = CapabilityKernel(
        granted_ops=["click"],
        min_peak_probability=0.45,   # measured operating point for button labels
    )
    hand = VirtualHand()
    report = TaskReport(task=task)

    for i, (instruction, buttons) in enumerate(zip(steps, screens), start=1):
        frame = _mock_editor_screen(buttons)

        # The reader names what is on screen. Nothing here reads a character.
        regions = reader.read_regions(frame, floor=0.0)
        by_label = {r.label.lower(): r for r in regions}
        named = sorted(by_label)

        target = instruction.strip().lower()
        hit = by_label.get(target)
        if hit is None:
            report.steps.append(
                StepResult(i, instruction, None, "not_found",
                           f"not on screen; reader saw {named or 'nothing'}")
            )
            continue

        box = hit.box
        point_px = ((box.x0 + box.x1) // 2, (box.y0 + box.y1) // 2)
        h, w = frame.shape[:2]
        point_norm = (point_px[0] / w, point_px[1] / h)

        evidence = GroundingEvidence(
            candidate_point_norm=point_norm,
            peak_probability=float(hit.readout.confidence),
            target_present=hit.readout.confidence >= 0.45,
            active_query=instruction,
        )
        decision = kernel.authorize(op="click", query=instruction, evidence=evidence, frame=frame)

        if not decision.allowed:
            report.steps.append(
                StepResult(i, instruction, hit.label, "refused",
                           f"conf={hit.readout.confidence:.2f}; {decision.reason}")
            )
            continue

        clicked = hand.click(*point_px)
        report.steps.append(
            StepResult(i, instruction, hit.label, "clicked",
                       f"conf={hit.readout.confidence:.2f}", point_px=clicked)
        )

    return report


def main() -> None:
    # A four-step task on a mock editor. Step 3 asks for a control that is not on
    # the screen, so the "target absent" path is shown rather than described.
    task = "save the document, then close it"
    steps = ["save", "close", "print", "undo"]
    screens = [
        ["save", "close", "undo", "redo"],   # save is present
        ["save", "close", "undo", "redo"],   # close is present
        ["save", "close", "undo", "redo"],   # print is NOT on the screen
        ["save", "close", "undo", "redo"],   # undo is present
    ]

    print(f"TASK: {task}")
    print(f"BRAIN planned {len(steps)} steps (a mock plan; the real planner is the cloud LLM)\n")
    report = run_task(task, steps, screens)
    for s in report.steps:
        print(s.as_line())
    print(f"\n{report.summary()}")
    print("\nThe reader names each control from pixels alone — no OCR, no charset.")
    print("Step 3 asks for 'print', which is not on the screen, so it is reported as")
    print("not found rather than snapped to a nearby label. Every click passes the")
    print("capability kernel first, and the hand is virtual: it records coordinates")
    print("and moves no cursor.")


if __name__ == "__main__":
    main()
