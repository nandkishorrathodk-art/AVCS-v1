"""Integration and Security Tests for AVC In-Memory Encrypted Vault."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

import avc
from avc import (
    ArtificialVisualCortex,
    InvalidKeyError,
    ModelVault,
    ScreenEye,
    VaultSecurityError,
    default_weights_path,
    load_cortex,
)


def test_avc_vault_encrypt_and_decrypt(tmp_path: Path):
    """Test encrypting and decrypting full 1.68M parameter AVC cortex model."""
    pt_path = default_weights_path()
    if not pt_path.exists():
        pytest.skip("avcs_cortex_weights.pt not found")

    vault_path = tmp_path / "avcs_cortex_weights.avcs"
    test_key = "avc_production_key_2026"

    # Encrypt
    ModelVault.encrypt_checkpoint(pt_path, vault_path, api_key=test_key)
    assert vault_path.exists()
    assert vault_path.stat().st_size > 0

    # Decrypt in RAM
    decrypted_dict = ModelVault.decrypt_to_ram(vault_path, api_key=test_key)
    assert isinstance(decrypted_dict, dict)
    original_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "state_dict" in original_dict:
        original_dict = original_dict["state_dict"]

    assert len(decrypted_dict) == len(original_dict)
    for k in list(original_dict.keys())[:10]:
        torch.testing.assert_close(decrypted_dict[k], original_dict[k])


def test_avc_load_cortex_from_vault(tmp_path: Path):
    """Test load_cortex instantiating directly from encrypted .avcs vault."""
    pt_path = default_weights_path()
    if not pt_path.exists():
        pytest.skip("avcs_cortex_weights.pt not found")

    vault_path = tmp_path / "cortex_vault.avcs"
    test_key = "avc_auth_key_xyz"
    ModelVault.encrypt_checkpoint(pt_path, vault_path, api_key=test_key)

    # 1. Invalid key fails closed
    with pytest.raises(InvalidKeyError):
        load_cortex(weights_path=vault_path, api_key="invalid_key")

    # 2. Missing key fails closed
    with pytest.raises(InvalidKeyError):
        load_cortex(weights_path=vault_path, api_key=None)

    # 3. Valid key loads model and runs perception
    cortex = load_cortex(weights_path=vault_path, api_key=test_key)
    assert isinstance(cortex, ArtificialVisualCortex)

    synth_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    state = cortex.perceive(synth_frame)
    assert state.primary_category in avc.UI_CATEGORIES
    assert state.inference_ms > 0.0


def test_screen_eye_with_vault(tmp_path: Path):
    """Test ScreenEye SDK initializing from encrypted .avcs vault."""
    pt_path = default_weights_path()
    if not pt_path.exists():
        pytest.skip("avcs_cortex_weights.pt not found")

    vault_path = tmp_path / "eye_vault.avcs"
    test_key = "eye_key_789"
    ModelVault.encrypt_checkpoint(pt_path, vault_path, api_key=test_key)

    with ScreenEye(weights_path=vault_path, api_key=test_key) as eye:
        assert eye.weights_loaded is True
        synth_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
        state = eye.perceive(synth_frame)
        assert state.primary_category in avc.UI_CATEGORIES
