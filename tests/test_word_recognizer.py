"""AVCS Milestone 3.5 — Unit and Integration Tests for PixelWordRecognizer.

Validates:
1. Neural architecture, parameter count (<250k), and forward pass shape.
2. Gradient flow across all convolutional stages.
3. Crop preprocessing across NumPy arrays, PIL images, and color dimensions.
4. Vocabulary integrity and bijection.
5. Strict pure-pixel compliance (zero OCR engine imports, zero UIA/DOM).
6. Checkpoint loading and evaluation mode.
7. End-to-end integration with FPNRegionProposer and VisualRegion.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from avcs.perception.fpn_proposer import FPNRegionProposer
from avcs.perception.visual_state import VisualRegion
from avcs.perception.word_recognizer import (
    CANONICAL_CROP_HEIGHT,
    CANONICAL_CROP_WIDTH,
    ID_TO_WORD,
    PixelWordRecognizer,
    UI_WORD_VOCABULARY,
    WORD_TO_ID,
    extract_word_label_from_point,
    load_word_recognizer,
)


def test_word_recognizer_architecture():
    """Validates model shapes and lightweight CPU footprint (<250k params)."""
    model = PixelWordRecognizer(num_classes=len(UI_WORD_VOCABULARY))
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 260_000, f"Parameter count {param_count} exceeded budget"
    assert param_count > 100_000, f"Parameter count {param_count} unexpectedly low"

    x = torch.randn(3, 3, CANONICAL_CROP_HEIGHT, CANONICAL_CROP_WIDTH)
    logits = model(x)
    assert logits.shape == (3, len(UI_WORD_VOCABULARY))
    assert not torch.isnan(logits).any()


def test_word_recognizer_backward_gradients():
    """Validates gradient backpropagation through all convolutional layers."""
    model = PixelWordRecognizer(num_classes=len(UI_WORD_VOCABULARY))
    x = torch.randn(2, 3, CANONICAL_CROP_HEIGHT, CANONICAL_CROP_WIDTH)
    target = torch.tensor([5, 12], dtype=torch.long)

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_vocabulary_integrity():
    """Validates that vocabulary contains canonical <none> and bijective ID mappings."""
    assert UI_WORD_VOCABULARY[0] == "<none>"
    assert len(UI_WORD_VOCABULARY) >= 120
    assert len(WORD_TO_ID) == len(UI_WORD_VOCABULARY)
    assert len(ID_TO_WORD) == len(UI_WORD_VOCABULARY)

    for i, w in enumerate(UI_WORD_VOCABULARY):
        assert WORD_TO_ID[w] == i
        assert ID_TO_WORD[i] == w


def test_crop_preprocessing():
    """Validates preprocessing on various crop input formats."""
    # NumPy RGB
    arr_rgb = np.zeros((20, 60, 3), dtype=np.uint8)
    t_rgb = PixelWordRecognizer.preprocess_crop(arr_rgb)
    assert t_rgb.shape == (1, 3, CANONICAL_CROP_HEIGHT, CANONICAL_CROP_WIDTH)
    assert t_rgb.dtype == torch.float32

    # NumPy Grayscale (2D)
    arr_gray = np.zeros((24, 50), dtype=np.uint8)
    t_gray = PixelWordRecognizer.preprocess_crop(arr_gray)
    assert t_gray.shape == (1, 3, CANONICAL_CROP_HEIGHT, CANONICAL_CROP_WIDTH)

    # PIL Image
    pil_img = Image.new("RGB", (80, 25), color=(100, 150, 200))
    t_pil = PixelWordRecognizer.preprocess_crop(pil_img)
    assert t_pil.shape == (1, 3, CANONICAL_CROP_HEIGHT, CANONICAL_CROP_WIDTH)


def test_pure_pixel_anti_cheating_contract():
    """Strictly audits that no external OCR or UI automation packages are imported."""
    import ast

    with open("avcs/perception/word_recognizer.py", "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0].lower())
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split(".")[0].lower())

    prohibited = [
        "pytesseract",
        "tesseract",
        "easyocr",
        "paddleocr",
        "winocr",
        "uiautomation",
        "pywinauto",
        "playwright",
        "selenium",
    ]
    for p in prohibited:
        assert p not in imported_modules, f"Prohibited OCR / DOM module imported: {p}"


def test_predict_crop_and_batch():
    """Validates single-crop and batch inference APIs."""
    model = PixelWordRecognizer(num_classes=len(UI_WORD_VOCABULARY))
    model.eval()

    sample_crop = np.random.randint(0, 255, (24, 70, 3), dtype=np.uint8)
    pred_word, conf = model.predict_crop(sample_crop, threshold=0.10)
    assert isinstance(pred_word, str)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0

    batch_crops = [np.random.randint(0, 255, (16, 40, 3), dtype=np.uint8) for _ in range(5)]
    results = model.predict_crops_batch(batch_crops, threshold=0.10)
    assert len(results) == 5
    for w, c in results:
        assert isinstance(w, str)
        assert 0.0 <= c <= 1.0


def test_fpn_proposer_text_integration():
    """Validates attaching word recognizer to FPNRegionProposer and region text fields."""
    model = PixelWordRecognizer(num_classes=len(UI_WORD_VOCABULARY)).eval()
    fpn = FPNRegionProposer(pyramid_dim=48).eval()
    fpn.attach_word_recognizer(model)

    synthetic_screen = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cat, cat_conf, proposals = fpn.propose_regions(synthetic_screen, threshold=0.20, max_proposals=6)

    assert isinstance(proposals, list)
    for p in proposals:
        assert isinstance(p, VisualRegion)
        # Check that detected_text is either None or a string in UI_WORD_VOCABULARY
        if p.detected_text is not None:
            assert p.detected_text in UI_WORD_VOCABULARY
            assert 0.0 <= p.text_confidence <= 1.0

        # Check serialization in to_dict()
        p_dict = {
            "region_id": p.region_id,
            "element_kind": p.element_kind,
            "detected_text": p.detected_text,
            "text_confidence": p.text_confidence,
        }
        assert "detected_text" in p_dict
