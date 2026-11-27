"""Safety invariants for the System 2 -> System 1 capability kernel.

These tests pin the *fail-closed* property that made the reasoning loop's click
step impossible to open safely. They use real frames from `datasets/ui_grounding`
and the real grounder checkpoint, so they exercise the production decision path
rather than a mock.

Run:  PYTHONPATH=. python3 -m pytest tests/test_capability_kernel.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avcs.control.capability import (
    MUTATING_OPS,
    CapabilityKernel,
    GroundingEvidence,
    frame_is_degenerate,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "datasets" / "ui_grounding" / "manifest.json"
GROUNDER_CKPT = REPO / "runs" / "weights" / "goal_grounder_best.pt"


# --------------------------------------------------------------------------- #
# Pure policy tests (no model, always runnable)
# --------------------------------------------------------------------------- #
def test_default_kernel_grants_nothing():
    k = CapabilityKernel()
    assert not k.has_any_mutating_grant
    for op in MUTATING_OPS:
        assert not k.is_granted(op), f"{op} must not be granted by default"


def test_observing_ops_never_need_a_grant():
    k = CapabilityKernel()
    for op in ("observe", "observe_target", "finish", "wait", "ask_user"):
        assert k.is_granted(op)


def test_authorize_allows_an_observing_op_with_no_evidence():
    """A "look at the screen" step must not be refused for want of a point.

    `authorize` used to run the grounding checks on every op, so an observing op
    with no candidate point came back denied -- a plain perception step read as a
    denied click. Observing ops actuate nothing, so they are allowed outright.
    """
    k = CapabilityKernel()  # no grants, no evidence, no frame
    d = k.authorize(
        op="observe_target",
        query="current screen",
        evidence=GroundingEvidence(candidate_point_norm=None, peak_probability=0.0),
    )
    assert d.allowed
    assert d.coordinate is None
    assert "observing" in d.reason


def test_mutating_op_refused_without_grant_even_with_perfect_evidence():
    k = CapabilityKernel()  # no grants
    decision = k.authorize(
        op="click",
        query="Save",
        evidence=GroundingEvidence(candidate_point_norm=(0.5, 0.5), peak_probability=1.0),
    )
    assert not decision.allowed
    assert "grant" in decision.reason


def test_grant_alone_is_not_enough_low_peak_is_refused():
    k = CapabilityKernel(granted_ops=["click"])
    decision = k.authorize(
        op="click",
        query="Save",
        evidence=GroundingEvidence(candidate_point_norm=(0.5, 0.5), peak_probability=0.30),
    )
    assert not decision.allowed
    assert "peak" in decision.reason


def test_grant_and_high_peak_is_allowed():
    k = CapabilityKernel(granted_ops=["click"])
    decision = k.authorize(
        op="click",
        query="Save",
        evidence=GroundingEvidence(candidate_point_norm=(0.5, 0.5), peak_probability=0.90),
    )
    assert decision.allowed
    assert decision.coordinate == (0.5, 0.5)


def test_allow_list_restricts_targets():
    k = CapabilityKernel(granted_ops=["click"], allowed_queries=["Launch button"])
    ok = k.authorize(
        op="click",
        query="Launch button",
        evidence=GroundingEvidence(candidate_point_norm=(0.1, 0.9), peak_probability=0.9),
    )
    bad = k.authorize(
        op="click",
        query="Delete button",
        evidence=GroundingEvidence(candidate_point_norm=(0.1, 0.9), peak_probability=0.9),
    )
    assert ok.allowed
    assert not bad.allowed
    assert "allow-list" in bad.reason


def test_allow_list_is_case_insensitive():
    k = CapabilityKernel(granted_ops=["click"], allowed_queries=["Launch Button"])
    d = k.authorize(
        op="click",
        query="launch button",
        evidence=GroundingEvidence(candidate_point_norm=(0.1, 0.9), peak_probability=0.9),
    )
    assert d.allowed


def test_saturated_presence_probability_is_not_trusted():
    """The shipped grounder returns target_present_probability == 1.0 for wrong
    queries too, so a kernel that trusted it would fail open. The kernel must
    decide on peak_probability, not on presence."""
    k = CapabilityKernel(granted_ops=["click"])
    decision = k.authorize(
        op="click",
        query="anything",
        evidence=GroundingEvidence(
            candidate_point_norm=(0.5, 0.5),
            peak_probability=0.20,               # genuinely low
            target_present_probability=1.0,      # saturated, must be ignored
            target_present=True,
        ),
    )
    assert not decision.allowed


def test_no_candidate_point_is_refused():
    k = CapabilityKernel(granted_ops=["click"])
    d = k.authorize(
        op="click",
        query="Save",
        evidence=GroundingEvidence(candidate_point_norm=None, peak_probability=0.99),
    )
    assert not d.allowed


def test_nan_peak_and_drag_peak2_are_refused():
    import math
    k = CapabilityKernel(granted_ops=["click", "drag"])
    d1 = k.authorize(
        op="click",
        query="Save",
        evidence=GroundingEvidence(candidate_point_norm=(0.5, 0.5), peak_probability=math.nan),
    )
    assert not d1.allowed
    assert "non-finite" in d1.reason

    d2 = k.authorize(
        op="drag",
        query="Item",
        evidence=GroundingEvidence(candidate_point_norm=(0.2, 0.2), peak_probability=0.95),
        arguments={"point2": (0.8, 0.8), "peak2": float("nan")},
    )
    assert not d2.allowed
    assert "non-finite" in d2.reason


# --------------------------------------------------------------------------- #
# Frame degeneracy
# --------------------------------------------------------------------------- #
def test_blank_frame_is_degenerate():
    blank = np.full((64, 64, 3), 200, dtype=np.uint8)
    degenerate, why = frame_is_degenerate(blank)
    assert degenerate
    assert "uniform" in why


def test_structured_frame_is_not_degenerate():
    frame = np.full((64, 64, 3), 40, dtype=np.uint8)
    frame[10:20, 10:50] = 220
    frame[40:50, 5:30] = 150
    degenerate, _ = frame_is_degenerate(frame)
    assert not degenerate


def test_degenerate_frame_blocks_even_with_grant():
    blank = np.full((64, 64, 3), 128, dtype=np.uint8)
    k = CapabilityKernel(granted_ops=["click"])
    d = k.authorize(
        op="click",
        query="Save",
        evidence=GroundingEvidence(candidate_point_norm=(0.5, 0.5), peak_probability=0.99),
        frame=blank,
    )
    assert not d.allowed
    assert "degenerate" in d.reason


# --------------------------------------------------------------------------- #
# Real-model integration (skipped when the gitignored checkpoint is absent)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not GROUNDER_CKPT.is_file(), reason="grounder checkpoint not present")
@pytest.mark.skipif(not MANIFEST.is_file(), reason="ui_grounding dataset not present")
def test_real_grounder_correct_query_passes_wrong_query_refused():
    """End-to-end: on a real frame, the real target is granted and a wrong query
    is refused by the capability floor."""
    from PIL import Image

    from avcs.goal_grounder import load_grounding_checkpoint

    manifest = json.loads(MANIFEST.read_text())
    frame_meta = next(f for f in manifest["frames"] if not f.get("is_negative"))
    img = np.array(Image.open(REPO / "datasets" / "ui_grounding" / frame_meta["file_path"]).convert("RGB"))
    target = frame_meta["interactive_points"][0]["target_query"]

    grounder = load_grounding_checkpoint(str(GROUNDER_CKPT), device="cpu")
    grounder.eval()
    k = CapabilityKernel(granted_ops=["click"])

    good = k.authorize_prediction(op="click", query=target, prediction=grounder.predict(img, target), frame=img)
    bad = k.authorize_prediction(op="click", query="Save button", prediction=grounder.predict(img, "Save button"), frame=img)

    assert good.allowed, f"correct query '{target}' should be granted: {good.reason}"
    assert not bad.allowed, "a wrong query must not be granted"


@pytest.mark.skipif(not GROUNDER_CKPT.is_file(), reason="grounder checkpoint not present")
def test_controller_default_is_fail_closed():
    """A controller built with no capability configuration must refuse to ground."""
    from avcs.control.controller import AVCSController
    from forge.vision.cortex.sdk import UnvalidatedGroundingError

    class _Src:
        def __init__(self):
            self._n = 0
            self._img = np.full((64, 64, 3), 40, dtype=np.uint8)
            self._img[10:20, 10:50] = 220

        def grab(self):
            self._n += 1
            return self._img

        def grab_with_sequence(self):
            self._n += 1
            return self._img, self._n

        def sequence_number(self):
            return self._n

        def close(self):
            pass

    c = AVCSController(
        source=_Src(),
        weights_path=str(GROUNDER_CKPT),
        device="cpu",
        force_virtual_motor=True,
        enable_streaming=False,
    )
    assert not c.capability.has_any_mutating_grant
    with pytest.raises(UnvalidatedGroundingError):
        c.locate("Launch button")
