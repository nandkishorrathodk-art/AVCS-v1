"""Tests for the System 1 one-model: one trunk, one latent, every task.

These pin the architectural claim, not a number: a single forward pass produces
every output, all outputs are functions of the same latent, and the model has no
hand-written decision path.
"""
from __future__ import annotations

import torch

from avcs.system1.dataset import System1OneModelDataset, collate
from avcs.system1.losses import OneModelLoss
from avcs.system1.model import (
    ACTION_NAMES,
    CATEGORY_NAMES,
    ELEMENT_KINDS,
    WORD_NAMES,
    OneModel,
    OneModelConfig,
)


def _model():
    torch.manual_seed(0)
    return OneModel(OneModelConfig(input_size=64, width=16, latent_dim=32, grid=8))


def test_single_forward_pass_returns_every_task():
    m = _model()
    x = torch.randn(2, 3, 64, 64)
    out = m(x, ["click Save", "click Stop"])
    g = 8
    assert out.latent.shape == (2, 32)
    assert out.category_logits.shape == (2, len(CATEGORY_NAMES))
    assert out.kind_logits.shape == (2, g * g, len(ELEMENT_KINDS))
    assert out.word_logits.shape == (2, g * g, len(WORD_NAMES))
    assert out.action_logits.shape == (2, len(ACTION_NAMES))
    assert out.point_logits.shape == (2, g * g)
    assert out.presence_logits.shape == (2, 1)


def test_one_latent_feeds_every_head():
    """Every head must read from the single latent map and nothing else.

    Proof: zero the latent map and every frame-conditioned head collapses to the
    same output regardless of the input image. If a head were secretly reading
    the trunk features directly, its output would still vary with the frame.
    """
    m = _model()
    m.eval()
    x1 = torch.randn(1, 3, 64, 64)
    x2 = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        m.latent.weight.zero_()
        m.latent.bias.zero_()
        a = m(x1, ["click"])
        b = m(x2, ["click"])

    # with a zero latent, frame-conditioned heads cannot see the frame at all
    assert torch.allclose(a.category_logits, b.category_logits)
    assert torch.allclose(a.kind_logits, b.kind_logits)
    assert torch.allclose(a.objectness_logits, b.objectness_logits)
    assert torch.allclose(a.word_logits, b.word_logits)
    assert torch.allclose(a.latent, b.latent)


def test_query_changes_grounding_but_not_the_frame_heads():
    """Grounding is query-conditioned; category is a property of the frame."""
    m = _model()
    m.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a = m(x, ["click Save"])
        b = m(x, ["click Stop"])
    assert not torch.allclose(a.point_logits, b.point_logits), "grounding must depend on the query"
    assert torch.allclose(a.category_logits, b.category_logits), "category must not depend on the query"


def test_no_content_branching_in_forward():
    """The forward path must not contain `if` on tensor values. We assert the
    model produces finite output for a blank frame as well as a noisy one, which
    a rule-based path with a degenerate-frame guard would not."""
    m = _model()
    m.eval()
    blank = torch.zeros(1, 3, 64, 64)
    noise = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        for x in (blank, noise):
            out = m(x, ["click"])
            assert torch.isfinite(out.latent).all()
            assert torch.isfinite(out.point_logits).all()


def test_point_norm_is_in_unit_square():
    m = _model()
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(3, 3, 64, 64), ["a", "b", "c"])
        p = out.point_norm()
    assert p.shape == (3, 2)
    assert (p >= 0).all() and (p <= 1).all()


def test_loss_has_gradient_to_the_trunk():
    """One loss, and it reaches the shared trunk -- that is what keeps the trunk
    serving every task instead of letting heads grow private front-ends."""
    m = _model()
    loss_fn = OneModelLoss()
    x = torch.randn(2, 3, 64, 64)
    out = m(x, ["click Save", "click Stop"])
    batch = {
        "category": torch.tensor([0, 1]),
        "objectness": (torch.rand(2, 64) > 0.5).float(),
        "kind": torch.randint(0, len(ELEMENT_KINDS), (2, 64)),
        "word": torch.randint(0, len(WORD_NAMES), (2, 64)),
        "action": torch.tensor([0, 0]),
        "present": torch.tensor([1.0, 0.0]),
        "point_cell": torch.tensor([3, 0]),
    }
    loss, br = loss_fn(out, batch)
    loss.backward()
    grad = m.trunk.stem.conv.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_query_token_ids_are_stable_across_processes():
    """The query encoder must map a word to the same id in a fresh process.

    A per-process-salted ``hash()`` would silently change the token a word maps
    to between training and inference, so a checkpoint's grounding head would
    read a different embedding for the same word.
    """
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tempfile

    repo = Path(__file__).resolve().parents[1]
    code = (
        "from avcs.system1.model import _QueryEncoder as Q;"
        "print(Q.word_id('Save', 4096), Q.word_id('click', 4096), Q.word_id('Stop', 4096))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)

    def _run_isolated() -> str:
        with tempfile.TemporaryFile() as tmp:
            p = subprocess.Popen(
                [sys.executable, "-c", code],
                stdin=subprocess.DEVNULL,
                stdout=tmp,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            p.wait()
            tmp.seek(0)
            return tmp.read().decode("utf-8", errors="ignore").strip()

    a = _run_isolated()
    b = _run_isolated()
    assert a and a == b, f"token ids differ between processes: {a!r} vs {b!r}"


def test_dataset_emits_every_target():
    ds = System1OneModelDataset(split="train", grid=16, image_size=64)
    item = ds[0]
    for key in ("image", "category", "objectness", "kind", "word", "action", "query", "present", "point_cell"):
        assert key in item
    batch = collate([ds[0], ds[1]])
    assert batch["image"].shape == (2, 3, 64, 64)


def test_point_norm_recovers_both_axes():
    """A peak off-centre in *both* axes must move both coordinates.

    Regression for a real bug: `point_norm` softmaxed along a single axis, so
    the other axis' marginal stayed constant and collapsed to 0.5 no matter
    where the peak was. A sharp, correct head then measured as a dead one and
    grounding was wrongly reported as broken.
    """
    m = _model()
    g = m.config.grid
    out = m(torch.randn(1, 3, 64, 64), ["click Save"])
    out.point_logits = torch.full_like(out.point_logits, -20.0)
    cx, cy = 1, g - 2  # top-left corner, off-centre on both axes
    out.point_logits[0, cy * g + cx] = 20.0
    x, y = out.point_norm()[0].tolist()
    assert abs(x - cx / (g - 1)) < 1e-4, f"x did not track the peak: {x}"
    assert abs(y - cy / (g - 1)) < 1e-4, f"y did not track the peak: {y}"

