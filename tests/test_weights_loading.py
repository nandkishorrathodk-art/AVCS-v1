from __future__ import annotations

import torch
from torch import nn

from avcs.weights import load_avcs_weights


def test_unrelated_checkpoint_does_not_report_weights_loaded(tmp_path):
    module = nn.Linear(3, 2)
    before = {key: value.clone() for key, value in module.state_dict().items()}
    checkpoint = tmp_path / "unrelated.pt"
    torch.save({"optimizer_state": {"step": 1}}, checkpoint)

    assert load_avcs_weights(module, path=str(checkpoint)) is False
    assert all(torch.equal(before[key], value) for key, value in module.state_dict().items())


def test_partial_checkpoint_does_not_report_ready_or_mutate_model(tmp_path):
    module = nn.Linear(3, 2)
    before = {key: value.clone() for key, value in module.state_dict().items()}
    checkpoint = tmp_path / "partial.pt"
    torch.save({"weight": torch.ones_like(module.weight)}, checkpoint)

    assert load_avcs_weights(module, path=str(checkpoint), strict=False) is False
    assert all(torch.equal(before[key], value) for key, value in module.state_dict().items())


def test_strict_loading_rejects_unexpected_keys(tmp_path):
    module = nn.Linear(3, 2)
    checkpoint = tmp_path / "unexpected.pt"
    state = module.state_dict()
    torch.save({**state, "unexpected": torch.tensor(1)}, checkpoint)

    assert load_avcs_weights(module, path=str(checkpoint), strict=True) is False


def test_complete_checkpoint_loads_and_reports_ready(tmp_path):
    module = nn.Linear(3, 2)
    expected = {
        "weight": torch.full_like(module.weight, 0.25),
        "bias": torch.full_like(module.bias, -0.5),
    }
    checkpoint = tmp_path / "complete.pt"
    torch.save(expected, checkpoint)

    assert load_avcs_weights(module, path=str(checkpoint), strict=False) is True
    assert all(torch.equal(expected[key], value) for key, value in module.state_dict().items())
