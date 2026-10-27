"""Closed loop: frame -> one model -> capability gate -> click -> verify.

This is the glue the system was missing. Perception is the *one* model
(`OneModel`); nothing else looks at the frame. The capability kernel decides
whether a grounded target may reach the hand, and the hand records the click.

Two things are deliberately separated:

* **Perception** -- one forward pass, one latent, every answer from it.
* **Policy** -- the kernel, which grants nothing by default. Safety is not a
  property of the model (it does not condition on its capabilities; measured),
  it is the gate.

Actuation is a `VirtualHand` that records coordinates and touches nothing. Swap
it for a real OS motor to actually click; the gate in front of it is unchanged.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Protocol, Sequence

import numpy as np
import torch
from PIL import Image

from avcs.control.capability import CapabilityKernel, GroundingEvidence
from avcs.control.verifier import VisualRippleVerifier
from avcs.system1.model import ACTION_NAMES, OneModel

# Ops that change machine state. Everything else (observe/wait/finish) is free.
MUTATING_OPS = frozenset({"click", "double_click", "type", "type_text", "press_key", "scroll"})


class Hand(Protocol):
    """A hand. The loop only ever calls `click` with normalized coordinates."""

    def click(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        ...


class VirtualHand:
    """Records clicks; no OS side effects. The honest default for a demo."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.clicks: list[tuple[int, int]] = []

    def click(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        px = int(round(x_norm * (self.width - 1)))
        py = int(round(y_norm * (self.height - 1)))
        self.clicks.append((px, py))
        return (px, py)


@dataclasses.dataclass
class LoopStep:
    """One turn of the loop, with the evidence that produced its outcome."""

    turn: int
    query: str
    category: str
    action: str
    presence: float
    point_norm: tuple[float, float]
    allowed: bool
    gate_reason: str
    clicked: Optional[tuple[int, int]] = None
    visual_delta: Optional[float] = None
    verified: Optional[bool] = None
    note: str = ""


def to_tensor(frame: np.ndarray, size: int) -> torch.Tensor:
    """HxWx3 uint8 frame -> normalized (3, size, size) tensor."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    im = Image.fromarray(frame).convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - 0.5) / 0.5
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()


def to_luma(frame: np.ndarray) -> np.ndarray:
    a = frame.astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)
    return a


class OneModelLoop:
    """Perceive with one model, gate with the kernel, click with a hand, verify.

    Parameters
    ----------
    presence_floor:
        The model's calibrated "the target is here" threshold. The *point*
        softmax peak saturates at 1.0 for present and absent queries alike
        (measured on the clean-room corpus), so it cannot gate; the presence
        head separates them cleanly (present median 1.00, absent median 0.00).
        The kernel's `min_peak_probability` is fed this value.
    """

    ripple_radius = 0.08  # reused by VisualRippleVerifier's delta math

    def __init__(
        self,
        model: OneModel,
        kernel: Optional[CapabilityKernel] = None,
        image_size: Optional[int] = None,
        hand: Optional[Hand] = None,
        presence_floor: float = 0.5,
        delta_threshold: float = 0.015,
    ) -> None:
        self.model = model
        self.kernel = kernel or CapabilityKernel()
        self.image_size = int(image_size or model.config.input_size)
        self.hand = hand or VirtualHand(1920, 1080)
        self.presence_floor = float(presence_floor)
        self.delta_threshold = float(delta_threshold)

    @torch.no_grad()
    def step(
        self,
        frame: np.ndarray,
        query: str,
        post_frame: Optional[np.ndarray] = None,
        turn: int = 0,
    ) -> LoopStep:
        """One turn. `post_frame` is the capture *after* the click, if any."""
        self.model.eval()
        img = to_tensor(frame, self.image_size).unsqueeze(0)
        out = self.model(img, [query])

        category = out.category_name()[0]
        action = ACTION_NAMES[int(out.action_logits.argmax(dim=-1))]
        presence = float(torch.sigmoid(out.presence_logits.squeeze(-1))[0])
        point = (float(out.point_norm()[0, 0]), float(out.point_norm()[0, 1]))
        gray = to_luma(frame)

        evidence = GroundingEvidence(
            candidate_point_norm=point,
            peak_probability=presence,
            target_present=presence >= self.presence_floor,
            active_query=query,
        )
        decision = self.kernel.authorize(op=action, query=query, evidence=evidence, frame=gray)

        clicked: Optional[tuple[int, int]] = None
        delta: Optional[float] = None
        verified: Optional[bool] = None

        if not decision.allowed:
            note = f"gate refused: {decision.reason}"
        elif action not in MUTATING_OPS:
            note = f"non-mutating op '{action}'; nothing actuated"
        else:
            clicked = self.hand.click(*point)
            if post_frame is not None:
                delta = float(
                    VisualRippleVerifier.calculate_local_photometric_delta(
                        self, point, gray, to_luma(post_frame)
                    )
                )
                verified = delta >= self.delta_threshold
                note = (
                    "screen changed at target between captures"
                    if verified
                    else "no visual change at target"
                )
            else:
                note = "clicked (no post-frame to verify)"

        return LoopStep(
            turn=turn,
            query=query,
            category=category,
            action=action,
            presence=presence,
            point_norm=point,
            allowed=decision.allowed,
            gate_reason=decision.reason,
            clicked=clicked,
            visual_delta=delta,
            verified=verified,
            note=note,
        )


def run_sequence(
    loop: OneModelLoop,
    frames: Sequence[np.ndarray],
    queries: Sequence[str],
    post_frames: Optional[Sequence[Optional[np.ndarray]]] = None,
) -> list[LoopStep]:
    """Run the loop over successive captures.

    Verification needs a capture of the *same* screen after the click. Two
    distinct screenshots have a large photometric delta no matter what the click
    did, so pairing frame `i` with frame `i+1` of a slideshow would report a
    verified change that the click never caused. Pass `post_frames` explicitly
    (aligned with `frames`) when a real post-click capture exists; otherwise
    `verified` stays `None`, which is the honest value.
    """
    steps: list[LoopStep] = []
    for i, (frame, query) in enumerate(zip(frames, queries)):
        post = post_frames[i] if post_frames is not None else None
        steps.append(loop.step(frame, query, post_frame=post, turn=i))
    return steps
