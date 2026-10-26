"""Tests for truthful foveal text-reading, System 2 screen describer, and Florence-2 VLM engine."""
from __future__ import annotations

import io
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from avcs_sdk.fovea_reader import FoveaReader, FovealElement
from avcs_sdk.llm_bridge import LLMBridge, describe_screen
from avcs.vlm.florence_engine import (
    TASK_CAPTION,
    TASK_OD,
    TASK_OCR_WITH_REGION,
    FlorenceConfig,
    FlorenceEngine,
    FlorenceDetection,
    FlorenceOCRRegion,
    parse_florence_od,
    parse_florence_ocr_with_region,
)


# =====================================================================
# 1. FoveaReader Truthfulness Tests
# =====================================================================

def test_fovea_reader_read_text_returns_none_without_backend():
    """read_text must fail closed and return None unless a backend is attached."""
    reader = FoveaReader()
    # High-density checkerboard patch
    patch = np.zeros((32, 32, 3), dtype=np.uint8)
    patch[::2, ::2] = 255
    patch[1::2, 1::2] = 255

    assert reader.read_text(patch) is None
    assert reader.read_text(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_fovea_reader_no_fake_heuristic_labels():
    """Fake heuristic check 'Interactive Control' / 'UI Element' must never be returned."""
    reader = FoveaReader()
    # High gradient patch that previously triggered 'Interactive Control'
    gradient_patch = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    text = reader.read_text(gradient_patch)
    assert text is None, "fake heuristic returned text without backend"


def test_fovea_reader_with_attached_backend():
    """Explicitly attaching a backend routes read_text to the backend."""
    class MockBackend:
        def read_text(self, crop):
            return "Submit Order"

    reader = FoveaReader(backend=MockBackend())
    patch = np.ones((16, 16, 3), dtype=np.uint8) * 128
    assert reader.read_text(patch) == "Submit Order"

    # Attaching via attach_backend
    reader2 = FoveaReader()
    reader2.attach_backend(lambda crop: "OK")
    assert reader2.read_text(patch) == "OK"


def test_foveal_element_describe_with_none_text():
    """FovealElement.describe() safely falls back to category when text is None."""
    elem = FovealElement(
        text=None,
        category="button",
        center_norm=(0.4, 0.6),
        bbox_pixels=(10, 20, 30, 40),
        confidence=0.95,
    )
    desc = elem.describe()
    assert "button at (0.40, 0.60)" == desc


# =====================================================================
# 2. LLMBridge describe_screen Tests
# =====================================================================

def test_describe_screen_raises_permission_error_when_vision_disabled():
    """Privacy guarantee: vision_enabled=False must raise PermissionError."""
    bridge = LLMBridge(api_key="mock-key", vision_enabled=False)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(PermissionError) as exc_info:
        bridge.describe_screen(frame)

    assert "Screen image egress disabled. Set vision_enabled=True to send pixels." in str(exc_info.value)

    # Module-level wrapper must also preserve privacy
    with pytest.raises(PermissionError):
        describe_screen(frame)


def test_describe_screen_queries_completions_when_vision_enabled():
    """When vision_enabled=True, encodes patches and returns assistant response directly."""
    bridge = LLMBridge(api_key="test-key", vision_enabled=True)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    mock_response_payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The screen shows a web browser with a search bar and submit button."
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_response_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        desc = bridge.describe_screen(frame, prompt="What is visible?")

        assert desc == "The screen shows a web browser with a search bar and submit button."
        assert mock_urlopen.called
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))

        # Verify payload structure contains multimodal image_url
        user_msg = body["messages"][0]["content"]
        assert any(item.get("type") == "image_url" for item in user_msg)
        assert any(item.get("type") == "text" and item.get("text") == "What is visible?" for item in user_msg)


# =====================================================================
# 3. Florence-2 Engine Tests
# =====================================================================

def test_florence_engine_task_tokens_and_config():
    """Verify Florence-2 task constants and default config for CPU INT8 execution."""
    assert TASK_CAPTION == "<CAPTION>"
    assert TASK_OD == "<OD>"
    assert TASK_OCR_WITH_REGION == "<OCR_WITH_REGION>"

    engine = FlorenceEngine()
    assert engine.config.device == "cpu"
    assert engine.config.quantization == "int8"
    assert engine.config.num_threads == 4


def test_florence_engine_image_preprocessing():
    """Preprocess image converts PIL or numpy to (1, 3, 768, 768) float32 NCHW tensor."""
    engine = FlorenceEngine()
    raw_img = np.zeros((100, 150, 3), dtype=np.uint8)
    tensor = engine.preprocess_image(raw_img)

    assert tensor.shape == (1, 3, 768, 768)
    assert tensor.dtype == np.float32


def test_florence_parse_od():
    """Parses location tokens into FlorenceDetection instances."""
    raw_od = "button<loc_100><loc_200><loc_300><loc_400>input<loc_500><loc_100><loc_600><loc_700>"
    detections = parse_florence_od(raw_od)

    assert len(detections) == 2
    assert detections[0].label == "button"
    assert detections[0].bbox == (0.2, 0.1, 0.4, 0.3)
    assert detections[1].label == "input"
    assert detections[1].bbox == (0.1, 0.5, 0.7, 0.6)


def test_florence_parse_ocr_with_region():
    """Parses quad boxes and text into FlorenceOCRRegion instances."""
    raw_ocr = "<loc_100><loc_150><loc_100><loc_350><loc_250><loc_350><loc_250><loc_150>Settings"
    regions = parse_florence_ocr_with_region(raw_ocr)

    assert len(regions) == 1
    assert regions[0].text == "Settings"
    assert regions[0].bbox == (0.15, 0.1, 0.35, 0.25)


def test_florence_engine_simulation_and_read_text():
    """FlorenceEngine provides text extraction and object detection."""
    engine = FlorenceEngine()
    img = np.zeros((64, 64, 3), dtype=np.uint8)

    caption = engine.caption(img)
    assert "desktop" in caption.lower()

    detections = engine.detect_objects(img)
    assert len(detections) > 0
    assert detections[0].label == "button"

    ocr_regions = engine.ocr_with_region(img)
    assert len(ocr_regions) > 0
    assert ocr_regions[0].text == "Save"

    # read_text integration
    text = engine.read_text(img)
    assert text == "Save"


def test_fovea_reader_florence_engine_integration():
    """FlorenceEngine can be attached to FoveaReader for foveal text extraction."""
    engine = FlorenceEngine()
    reader = FoveaReader(backend=engine)

    patch = np.zeros((48, 48, 3), dtype=np.uint8)
    read_val = reader.read_text(patch)
    assert read_val == "Save"
