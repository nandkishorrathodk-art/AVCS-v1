"""End-to-end loop tests: frame -> one model -> capability gate -> click -> verify.

The pure-policy tests use a tiny randomly-initialised model so they always run.
The integration test uses the real GPU checkpoint and real held-out frames when
both are present (both are gitignored, so it skips on a fresh checkout).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from avcs.control.capability import CapabilityKernel
from avcs.system1.loop import OneModelLoop, VirtualHand, run_sequence, to_luma
from avcs.system1.model import OneModel, OneModelConfig

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "runs" / "weights" / "system1_one_model.pt"
CORPUS = REPO / "datasets" / "cleanroom_system1_v1"


def _tiny_model():
    torch.manual_seed(0)
    return OneModel(OneModelConfig(input_size=64, width=16, latent_dim=32, grid=8))


def _blank_frame(size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), 128, dtype=np.uint8)


def _textured_frame(seed: int = 0, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def test_loop_is_fail_closed_without_a_grant():
    loop = OneModelLoop(_tiny_model(), kernel=CapabilityKernel(), image_size=64)
    step = loop.step(_textured_frame(1), "click Save")
    assert not step.allowed
    assert step.clicked is None
    assert "no capability grant" in step.gate_reason


def test_presence_floor_feeds_the_loops_own_kernel():
    # The docstring promises presence_floor becomes the kernel's
    # min_peak_probability when the loop builds the kernel itself.
    loop = OneModelLoop(_tiny_model(), image_size=64, presence_floor=0.5)
    assert loop.kernel.min_peak_probability == 0.5


def test_presence_floor_does_not_override_a_caller_supplied_kernel():
    # An explicit kernel carries its own policy; the loop must not mutate it.
    kernel = CapabilityKernel(granted_ops=["click"])
    loop = OneModelLoop(_tiny_model(), kernel=kernel, image_size=64, presence_floor=0.1)
    assert loop.kernel is kernel
    assert loop.kernel.min_peak_probability == 0.75


def test_granted_click_reaches_the_hand():
    loop = OneModelLoop(
        _tiny_model(),
        kernel=CapabilityKernel(granted_ops=["click"], allow_degenerate_frames=True, min_peak_probability=0.0),
        image_size=64,
        hand=VirtualHand(640, 480),
    )
    step = loop.step(_textured_frame(2), "click Save")
    assert step.allowed
    assert step.clicked is not None
    assert len(loop.hand.clicks) == 1


def test_degenerate_frame_is_refused_even_when_granted():
    loop = OneModelLoop(
        _tiny_model(),
        kernel=CapabilityKernel(granted_ops=["click"]),
        image_size=64,
    )
    step = loop.step(_blank_frame(), "click Save")
    assert not step.allowed
    assert step.clicked is None
    assert "degenerate" in step.gate_reason


def test_no_verification_without_a_post_frame():
    """A click with no post-click capture must not claim it was verified."""
    loop = OneModelLoop(
        _tiny_model(),
        kernel=CapabilityKernel(granted_ops=["click"], allow_degenerate_frames=True, min_peak_probability=0.0),
        image_size=64,
    )
    step = loop.step(_textured_frame(3), "click Save")
    assert step.clicked is not None
    assert step.verified is None
    assert step.visual_delta is None


def test_two_different_screens_are_not_treated_as_a_click_response():
    """Regression: pairing distinct screenshots must not read as 'verified'.

    The earlier harness verified frame i against frame i+1 of a slideshow, so a
    large inter-scene delta reported a change the click never caused.
    """
    loop = OneModelLoop(
        _tiny_model(),
        kernel=CapabilityKernel(granted_ops=["click"], allow_degenerate_frames=True, min_peak_probability=0.0),
        image_size=64,
    )
    frames = [_textured_frame(10), _textured_frame(11)]
    steps = run_sequence(loop, frames, ["click Save", "click Stop"])
    assert all(s.verified is None for s in steps)


def test_localized_change_at_target_verifies():
    loop = OneModelLoop(
        _tiny_model(),
        kernel=CapabilityKernel(granted_ops=["click"], allow_degenerate_frames=True, min_peak_probability=0.0),
        image_size=64,
    )
    frame = _textured_frame(5)
    post = frame.copy()
    post[30:38, 30:38] = 255
    step = loop.step(frame, "click Save", post_frame=post)
    assert step.clicked is not None
    assert step.visual_delta is not None and step.visual_delta > 0
    assert step.verified is True


@pytest.mark.skipif(not CKPT.is_file(), reason="one-model checkpoint not present")
@pytest.mark.skipif(not CORPUS.is_dir(), reason="clean-room corpus not present")
def test_real_model_grounds_close_and_loop_verifies():
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = OneModel(OneModelConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()

    anns = sorted(glob.glob(str(CORPUS / "annotations" / "test_heldout" / "*.json")))[:6]
    frames, queries, targets = [], [], []
    for fp in anns:
        ann = json.loads(open(fp).read())
        q = next((q for q in ann.get("queries", []) if q.get("target_presence")), None)
        if not q:
            continue
        frames.append(
            np.asarray(
                Image.open(CORPUS / "frames" / "test_heldout" / f"{ann['frame_id']}.png").convert("RGB"),
                dtype=np.uint8,
            )
        )
        queries.append(q["query"])
        targets.append(tuple(q["target_point_norm"]))

    loop = OneModelLoop(
        model,
        kernel=CapabilityKernel(granted_ops=["click"]),
        hand=VirtualHand(1920, 1080),
    )

    # Real grounding accuracy: the prediction must land near the true target.
    steps = run_sequence(loop, frames, queries)
    errs = [
        ((s.point_norm[0] - t[0]) ** 2 + (s.point_norm[1] - t[1]) ** 2) ** 0.5
        for s, t in zip(steps, targets)
    ]
    assert np.median(errs) < 0.1, f"grounding median error too high: {errs}"

    # With a synthesized response at the *true* target, a correct ground must
    # see it inside the ripple window.
    post = []
    for f, t in zip(frames, targets):
        g = f.copy()
        h, w = g.shape[:2]
        cy, cx = int(t[1] * (h - 1)), int(t[0] * (w - 1))
        g[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3] = 255
        post.append(g)
    verified = run_sequence(loop, frames, queries, post_frames=post)
    hits = [s for s in verified if s.verified]
    assert len(hits) >= len(frames) - 1, f"expected the response to be seen: {[s.verified for s in verified]}"


def test_scorecard_query_kind_parsing():
    from avcs.system1.scorecard import query_kind

    assert query_kind("click Save button") == "button"
    assert query_kind("click Notes input field") == "input_field"
    assert query_kind("click View tab") == "tab"
    assert query_kind("click icon") == "icon"


@pytest.mark.skipif(not CKPT.is_file(), reason="one-model checkpoint not present")
@pytest.mark.skipif(not CORPUS.is_dir(), reason="clean-room corpus not present")
def test_scorecard_in_bbox_metric_runs_on_real_checkpoint():
    """The scorecard's headline metric must be 'point inside the target bbox'.

    Cell-accuracy understates grounding: a point one cell off can still land
    inside a large control. Pin that the metric is computed and that a correct
    model scores well on it.
    """
    from avcs.system1.scorecard import collect, load_model

    model = load_model(str(CKPT))
    g = model.config.grid
    rows = collect(str(CORPUS), "test_heldout")
    present = [r for r in rows if r["present"] and r["target_box"] is not None]
    assert present, "no present queries with a target box"

    import torch
    from avcs.system1.loop import to_tensor

    in_box = 0
    for r in present:
        p = CORPUS / "frames" / "test_heldout" / f"{r['frame_id']}.png"
        frame = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        with torch.no_grad():
            out = model(to_tensor(frame, model.config.input_size).unsqueeze(0), [r["query"]])
        px, py = out.point_norm()[0].tolist()
        x0, y0, x1, y1 = r["target_box"]
        if x0 <= px <= x1 and y0 <= py <= y1:
            in_box += 1
    rate = in_box / len(present)
    assert rate >= 0.85, f"in-bbox rate {rate:.2f} below expectation"


def test_loop_latency_meets_a_screen_clock_at_a_realistic_size():
    """A 128px, width-64 model must beat the 60 FPS screen budget on CPU.

    Timing takes the minimum over runs, not the mean: on a loaded box the mean
    is dominated by scheduler preemption.
    """
    import time

    from avcs.system1.loop import to_tensor

    torch.manual_seed(0)
    model = OneModel(OneModelConfig(input_size=128, width=64, latent_dim=128, grid=16)).eval()
    frame = np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    x = to_tensor(frame, 128).unsqueeze(0)
    for _ in range(3):
        with torch.no_grad():
            model(x, ["click Save button"])
    times = []
    for _ in range(15):
        with torch.no_grad():
            t0 = time.perf_counter()
            model(x, ["click Save button"])
            times.append(time.perf_counter() - t0)
    ms = min(times) * 1000.0
    assert ms < 16.7, f"128px w64 took {ms:.1f} ms, above the 60 FPS budget"
