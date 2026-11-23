"""Unit tests for Step 5.1 Zero-Shot Evaluator."""
import pytest
from benchmarks.eval_qwen_showdown import (
    parse_prediction,
    evaluate_prediction,
    wilson_interval,
    ShowdownSample,
    load_showdown_samples,
)


def test_parse_prediction_canonical_json_absolute_pixels():
    # Native 1440x900 macOS display
    raw = '{"action": "left_click", "coordinate": [1136, 202]}'
    coord = parse_prediction(raw, 1440, 900)
    assert coord == (1136, 202)


def test_parse_prediction_qwen25_native_box_absolute():
    # Qwen2.5-VL outputs bounding box tokens in absolute pixels: (y1, x1, y2, x2)
    raw = '<|box_start|>(202, 1136, 225, 1162)<|box_end|>'
    # Center: cx = (1136 + 1162)/2 = 1149, cy = (202 + 225)/2 = 213.5 -> 214
    coord = parse_prediction(raw, 1440, 900)
    assert coord == (1149, 214)


def test_parse_prediction_normalized_floats_fallback():
    # If a model outputs normalized floats in [0.0, 1.0]
    raw = '{"coordinate": [0.5, 0.25]}'
    coord = parse_prediction(raw, 1280, 800)
    assert coord == (640, 200)


def test_evaluate_prediction_strict_0px_inside_bbox():
    sample = ShowdownSample(
        id="s1",
        image_rel_path="dummy.webp",
        instruction="Click button",
        x1=100, y1=200, x2=200, y2=250,
        width=1280, height=800,
    )
    # Target center: (150, 225) -> strictly inside bbox
    hit_strict, hit_15px, dist = evaluate_prediction(sample, (150, 225), diagnostic_tolerance_px=15.0)
    assert hit_strict is True
    assert hit_15px is True
    assert dist == 0.0


def test_evaluate_prediction_10px_near_miss_strict_false():
    sample = ShowdownSample(
        id="s1",
        image_rel_path="dummy.webp",
        instruction="Click button",
        x1=100, y1=200, x2=200, y2=250,
        width=1280, height=800,
    )
    # 10px below bbox bottom (y=260) -> outside bbox (strict 0px is FALSE, diagnostic 15px is TRUE)
    hit_strict, hit_15px, dist = evaluate_prediction(sample, (150, 260), diagnostic_tolerance_px=15.0)
    assert hit_strict is False
    assert hit_15px is True
    assert dist == 10.0


def test_evaluate_prediction_50px_miss():
    sample = ShowdownSample(
        id="s1",
        image_rel_path="dummy.webp",
        instruction="Click button",
        x1=100, y1=200, x2=200, y2=250,
        width=1280, height=800,
    )
    # 50px away -> both False
    hit_strict, hit_15px, dist = evaluate_prediction(sample, (150, 300), diagnostic_tolerance_px=15.0)
    assert hit_strict is False
    assert hit_15px is False
    assert dist == 50.0


def test_load_downloaded_showdown_samples():
    samples = load_showdown_samples(r"C:\Users\nandk\AVCS-v1\data\showdown-clicks-dev\data.csv")
    assert len(samples) == 557
    assert samples[0].id == "9bfd79b4abe532b0"
    assert samples[0].width == 1470
    assert samples[0].height == 956


def test_wilson_interval_bounds():
    low, high = wilson_interval(100, 557)
    assert 0.0 < low < high < 100.0
    assert 14.0 < low < 16.0
    assert 20.0 < high < 22.0
