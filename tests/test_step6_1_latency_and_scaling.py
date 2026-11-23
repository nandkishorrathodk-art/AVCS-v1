"""Unit & Integration Tests for Step 6.1: FP16/WebSocket/smart_resize/Turbo-JPEG Upgrades.

Verifies:
1. Qwen2.5-VL smart_resize aspect-ratio divisibility (factor=28) and pixel limits.
2. Coordinate rescaling math: Eliminates the 40% vs 76% coordinate drift on Showdown-Clicks.
3. parse_prediction backward compatibility and dual-canvas rescaling.
4. Turbo JPEG compression: <15ms warm latency and <35KB payload for 1280x800 frames.
5. Assistant prefill: deterministic JSON emission without markdown prose or hallucinations.
6. Server endpoints: HTTP POST /act and persistent WebSocket /ws/act streaming.
7. VLAClient thin-client loop, binary framing, URL routing, and multi-step streaming.
"""
from __future__ import annotations

import base64
import io
import json
import math
import time
from typing import Any, Dict, Tuple
import unittest

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from benchmarks.eval_qwen_showdown import (
    parse_prediction,
    rescale_coordinates_to_native,
    smart_resize,
    evaluate_prediction,
    ShowdownSample,
)
from avcs.runtime.turbo_jpeg import decode_turbo_jpeg, encode_turbo_jpeg
from avcs.runtime.server import (
    ASSISTANT_PREFILL,
    app,
    run_inference_with_prefill,
    set_inference_handler,
)
from avcs.runtime.vla_client import VLAClient


class TestSmartResizeAndCoordinateRescaling(unittest.TestCase):
    """Verifies Qwen2.5-VL smart_resize math and resolution mapping."""

    def test_smart_resize_factor_divisibility_and_bounds(self):
        # 1. Showdown-Clicks standard 1470x956
        h1, w1 = smart_resize(956, 1470, factor=28, max_pixels=14 * 14 * 4 * 1280)
        self.assertEqual(h1 % 28, 0, f"Height {h1} must be divisible by 28")
        self.assertEqual(w1 % 28, 0, f"Width {w1} must be divisible by 28")
        self.assertLessEqual(h1 * w1, 14 * 14 * 4 * 1280, "Total pixels must satisfy max_pixels ceiling")
        self.assertEqual((h1, w1), (784, 1232))

        # 2. Standard 1080p (1920x1080)
        h2, w2 = smart_resize(1080, 1920, factor=28, max_pixels=14 * 14 * 4 * 1280)
        self.assertEqual(h2 % 28, 0)
        self.assertEqual(w2 % 28, 0)
        self.assertLessEqual(h2 * w2, 14 * 14 * 4 * 1280)

        # 3. Canonical 1280x800 with 1024000 max_pixels (used in train_vla_qwen)
        h3, w3 = smart_resize(800, 1280, factor=28, max_pixels=1280 * 800)
        self.assertEqual(h3 % 28, 0)
        self.assertEqual(w3 % 28, 0)
        self.assertLessEqual(h3 * w3, 1280 * 800)

        # 4. Small image scaled to min_pixels
        h4, w4 = smart_resize(40, 40, factor=28, min_pixels=56 * 56)
        self.assertGreaterEqual(h4 * w4, 56 * 56)
        self.assertEqual(h4 % 28, 0)
        self.assertEqual(w4 % 28, 0)

        # 5. Extreme aspect ratio raises ValueError
        with self.assertRaises(ValueError):
            smart_resize(10, 2500)

    def test_rescale_coordinates_to_native_drift_elimination(self):
        """Demonstrates fixing the 40% vs 76% coordinate drift on showdown-clicks-dev."""
        raw_w, raw_h = 1470, 956
        resized_h, resized_w = smart_resize(raw_h, raw_w)  # (784, 1232)

        # Native button at (735, 478) with a 30px bounding box
        target_bbox = (720, 460, 750, 495)
        sample = ShowdownSample(
            id="s_test",
            image_rel_path="dummy.webp",
            instruction="Click search",
            x1=target_bbox[0],
            y1=target_bbox[1],
            x2=target_bbox[2],
            y2=target_bbox[3],
            width=raw_w,
            height=raw_h,
        )

        # In resized 1232x784 space, the model predicts the exact button center:
        pred_x_resized = int(round(735 * (resized_w / raw_w)))  # 616
        pred_y_resized = int(round(478 * (resized_h / raw_h)))  # 392

        # SCENARIO A: Old bug (evaluating resized coordinate directly against native bbox)
        unscaled_pred = (pred_x_resized, pred_y_resized)
        hit_strict_old, hit_15px_old, dist_old = evaluate_prediction(sample, unscaled_pred)
        self.assertFalse(hit_strict_old, "Unscaled prediction should fail strict 0px")
        self.assertGreater(dist_old, 50.0, "Drift error should be substantial (>50px)")

        # SCENARIO B: Fixed (rescaling back to native raw_w, raw_h)
        scaled_pred = rescale_coordinates_to_native(
            (pred_x_resized, pred_y_resized),
            raw_w=raw_w,
            raw_h=raw_h,
            resized_w=resized_w,
            resized_h=resized_h,
        )
        self.assertEqual(scaled_pred, (735, 478))
        hit_strict_fixed, hit_15px_fixed, dist_fixed = evaluate_prediction(sample, scaled_pred)
        self.assertTrue(hit_strict_fixed, "Rescaled prediction must land strictly inside bbox (0px error)")
        self.assertTrue(hit_15px_fixed)
        self.assertEqual(dist_fixed, 0.0)

    def test_rescale_coordinates_boundary_clamping(self):
        raw_w, raw_h = 1280, 800
        resized_w, resized_h = 640, 400

        # Out-of-bounds coordinates clamped
        coord = rescale_coordinates_to_native((700, 450), raw_w, raw_h, resized_w, resized_h)
        self.assertEqual(coord, (1279, 799))

        # None coordinate returns None
        self.assertIsNone(rescale_coordinates_to_native(None, raw_w, raw_h, resized_w, resized_h))

    def test_parse_prediction_formats_with_rescaling(self):
        raw_w, raw_h = 1470, 956
        resized_w, resized_h = 1232, 784

        # 1. Canonical JSON format
        raw_json = '{"action": "left_click", "coordinate": [616, 392]}'
        p1 = parse_prediction(raw_json, canvas_w=resized_w, canvas_h=resized_h, raw_w=raw_w, raw_h=raw_h)
        self.assertEqual(p1, (735, 478))

        # 2. Qwen box tokens
        raw_box = '<|box_start|>(382, 606, 402, 626)<|box_end|>'  # cx=616, cy=392
        p2 = parse_prediction(raw_box, canvas_w=resized_w, canvas_h=resized_h, raw_w=raw_w, raw_h=raw_h)
        self.assertEqual(p2, (735, 478))

        # 3. Normalized float format [0.5, 0.5]
        raw_norm = '{"coordinate": [0.5, 0.5]}'
        p3 = parse_prediction(raw_norm, canvas_w=resized_w, canvas_h=resized_h, raw_w=raw_w, raw_h=raw_h)
        self.assertEqual(p3, (735, 478))

        # 4. Backward compatibility: if raw_w and raw_h are None, behaves as 1:1
        p4 = parse_prediction('{"coordinate": [100, 200]}', canvas_w=1280, canvas_h=800)
        self.assertEqual(p4, (100, 200))

    def test_parse_prediction_boundary_integers_not_normalized(self):
        """Verifies integer boundary pixels [1, 1], [0, 1], [1, 0] are NOT wrongly treated as 100% normalized screen coordinates."""
        # Test on 1280x800 canvas with 1:1 scaling
        p_1_1 = parse_prediction('{"coordinate": [1, 1]}', canvas_w=1280, canvas_h=800)
        self.assertEqual(p_1_1, (1, 1), f"Pixel (1, 1) must remain (1, 1) and not get catapulted across screen: got {p_1_1}")

        p_1_0 = parse_prediction('{"coordinate": [1, 0]}', canvas_w=1280, canvas_h=800)
        self.assertEqual(p_1_0, (1, 0), f"Pixel (1, 0) must remain (1, 0): got {p_1_0}")

        p_0_1 = parse_prediction('{"coordinate": [0, 1]}', canvas_w=1280, canvas_h=800)
        self.assertEqual(p_0_1, (0, 1), f"Pixel (0, 1) must remain (0, 1): got {p_0_1}")

        p_0_0 = parse_prediction('{"coordinate": [0, 0]}', canvas_w=1280, canvas_h=800)
        self.assertEqual(p_0_0, (0, 0))

        # Explicit floats [1.0, 1.0] are treated as normalized coordinates
        p_float_1 = parse_prediction('{"coordinate": [1.0, 1.0]}', canvas_w=1280, canvas_h=800)
        self.assertEqual(p_float_1, (1279, 799))

    def test_parse_prediction_truncated_brackets_resilience(self):
        """Verifies parsing recovers coordinates even if token budget cuts off before closing ']'."""
        # Truncated canonical JSON coordinate
        raw_trunc1 = '{"action": "left_click", "coordinate": [346, 785'
        p1 = parse_prediction(raw_trunc1, canvas_w=1280, canvas_h=800)
        self.assertEqual(p1, (346, 785))

        # Truncated plain bracket pair
        raw_trunc2 = '[346, 785'
        p2 = parse_prediction(raw_trunc2, canvas_w=1280, canvas_h=800)
        self.assertEqual(p2, (346, 785))


class TestTurboJPEGAndAssistantPrefill(unittest.TestCase):
    """Verifies low-latency Turbo JPEG compression and Assistant Prefill parsing."""

    def test_turbo_jpeg_encode_speed_and_payload_budget(self):
        # Create synthetic 1280x800 desktop canvas
        frame = np.full((800, 1280, 3), 245, dtype=np.uint8)
        frame[750:, :] = 40  # Taskbar
        frame[100:600, 200:900] = 255  # Window
        frame[100:130, 200:900] = 60  # Titlebar

        # Warmup
        _ = encode_turbo_jpeg(frame, quality=75)

        t0 = time.time()
        jpeg_bytes = encode_turbo_jpeg(frame, quality=75)
        enc_ms = (time.time() - t0) * 1000.0

        # Verification 1: Header magic bytes
        self.assertEqual(jpeg_bytes[:2], b"\xff\xd8", "Must be valid JPEG magic header")
        self.assertEqual(jpeg_bytes[-2:], b"\xff\xd9", "Must end with JPEG EOI marker")

        # Verification 2: Payload budget < 35KB
        payload_kb = len(jpeg_bytes) / 1024.0
        self.assertLess(payload_kb, 35.0, f"Payload must be < 35KB, got {payload_kb:.1f} KB")

        # Verification 3: Encoding speed < 25ms warm
        self.assertLess(enc_ms, 25.0, f"Encoding must be < 25ms, got {enc_ms:.2f} ms")

        # Verification 4: Decoding fidelity
        decoded = decode_turbo_jpeg(jpeg_bytes)
        self.assertEqual(decoded.size, (1280, 800))
        self.assertEqual(decoded.mode, "RGB")

    def test_assistant_prefill_invariants(self):
        # Prefill token string
        self.assertEqual(ASSISTANT_PREFILL, '{"action": "left_click", "coordinate": [')

        # Model continuation of only 6 tokens: "346, 785]}"
        continuation = "346, 785]}"
        reconstructed = ASSISTANT_PREFILL + continuation

        data = json.loads(reconstructed)
        self.assertEqual(data["action"], "left_click")
        self.assertEqual(data["coordinate"], [346, 785])


class TestVLAServerHTTPAndWebSocketEndpoints(unittest.TestCase):
    """Verifies FastAPI HTTP /act and WebSocket /ws/act endpoints using TestClient."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Create a sample test frame
        cls.test_arr = np.full((800, 1280, 3), 200, dtype=np.uint8)
        cls.test_arr[760:, :] = 35
        cls.jpeg_bytes = encode_turbo_jpeg(cls.test_arr, quality=75)
        cls.b64_str = base64.b64encode(cls.jpeg_bytes).decode("utf-8")

    def tearDown(self):
        set_inference_handler(None)

    def test_http_act_success(self):
        payload = {
            "instruction": "Click taskbar search",
            "image_base64": self.b64_str,
            "width": 1280,
            "height": 800,
        }
        resp = self.client.post("/act", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["action"], "left_click")
        self.assertIsInstance(data["coordinate"], list)
        self.assertEqual(len(data["coordinate"]), 2)
        self.assertIn("latency_ms", data)
        self.assertIn(ASSISTANT_PREFILL, data["raw_response"])

    def test_http_act_invalid_base64_error(self):
        payload = {
            "instruction": "Click here",
            "image_base64": "not_valid_base64!@@#",
            "width": 1280,
            "height": 800,
        }
        resp = self.client.post("/act", json=payload)
        self.assertEqual(resp.status_code, 400)

    def test_websocket_act_binary_streaming(self):
        with self.client.websocket_connect("/ws/act?instruction=Click+Start&width=1280&height=800") as ws:
            # Send raw binary JPEG bytes
            ws.send_bytes(self.jpeg_bytes)
            data = ws.receive_json()

            self.assertEqual(data["action"], "left_click")
            self.assertIsInstance(data["coordinate"], list)
            self.assertEqual(len(data["coordinate"]), 2)
            self.assertGreaterEqual(data["latency_ms"], 0.0)

    def test_websocket_act_framed_packet_with_header(self):
        with self.client.websocket_connect("/ws/act") as ws:
            header = json.dumps({"instruction": "Click taskbar", "width": 1280, "height": 800}).encode("utf-8")
            packet = len(header).to_bytes(4, byteorder="big") + header + self.jpeg_bytes

            ws.send_bytes(packet)
            data = ws.receive_json()
            self.assertEqual(data["action"], "left_click")
            self.assertIsNotNone(data["coordinate"])

    def test_websocket_persistent_multi_step_streaming(self):
        """Verifies connection persistence over multiple consecutive steps."""
        with self.client.websocket_connect("/ws/act?width=1280&height=800") as ws:
            for step in range(5):
                t0 = time.time()
                ws.send_bytes(self.jpeg_bytes)
                data = ws.receive_json()
                dt = (time.time() - t0) * 1000.0

                self.assertEqual(data["action"], "left_click")
                self.assertIsNotNone(data["coordinate"])
                self.assertLess(dt, 100.0, f"Step {step} local roundtrip should be <100ms")

    def test_custom_inference_handler_in_server(self):
        called = []

        def mock_predictor(image: Image.Image, instruction: str, w: int, h: int) -> Dict[str, Any]:
            called.append((image.size, instruction, w, h))
            return {
                "action": "left_click",
                "coordinate": [346, 785],
                "raw_response": f'{ASSISTANT_PREFILL}346, 785]}}',
                "latency_ms": 1.25,
            }

        set_inference_handler(mock_predictor)

        with self.client.websocket_connect("/ws/act?instruction=Find+Search&width=1280&height=800") as ws:
            ws.send_bytes(self.jpeg_bytes)
            data = ws.receive_json()

            self.assertEqual(len(called), 1)
            self.assertEqual(called[0][1], "Find Search")
            self.assertEqual(data["coordinate"], [346, 785])
            self.assertEqual(data["latency_ms"], 1.25)


class TestVLAClient(unittest.TestCase):
    """Verifies thin client URL resolution and routing."""

    def test_url_resolution(self):
        client = VLAClient(server_url="http://localhost:8000")
        ws_url = client._resolve_ws_url("Click Icon", 1280, 800)
        self.assertTrue(ws_url.startswith("ws://localhost:8000/ws/act"))
        self.assertIn("width=1280", ws_url)
        self.assertIn("instruction=Click+Icon", ws_url)

        http_url = client._resolve_http_url()
        self.assertEqual(http_url, "http://localhost:8000/act")

    def test_url_resolution_https_to_wss(self):
        client = VLAClient(server_url="https://cloud.avcs.ai")
        ws_url = client._resolve_ws_url()
        self.assertTrue(ws_url.startswith("wss://cloud.avcs.ai/ws/act"))
        http_url = client._resolve_http_url()
        self.assertEqual(http_url, "https://cloud.avcs.ai/act")

    def test_url_resolution_with_existing_endpoint_suffixes(self):
        """Verifies URLs already containing /act or /ws/act do not produce corrupted paths like /act/ws/act."""
        # Case 1: Base configured with /act
        client1 = VLAClient(server_url="http://localhost:8000/act")
        ws_url1 = client1._resolve_ws_url()
        http_url1 = client1._resolve_http_url()
        self.assertTrue(ws_url1.startswith("ws://localhost:8000/ws/act"))
        self.assertNotIn("/act/ws/act", ws_url1)
        self.assertEqual(http_url1, "http://localhost:8000/act")

        # Case 2: Base configured with /ws/act
        client2 = VLAClient(server_url="ws://localhost:8000/ws/act")
        ws_url2 = client2._resolve_ws_url()
        http_url2 = client2._resolve_http_url()
        self.assertTrue(ws_url2.startswith("ws://localhost:8000/ws/act"))
        self.assertEqual(http_url2, "http://localhost:8000/act")

        # Case 3: HTTPS remote Cloudflare URL ending with /ws/act/
        client3 = VLAClient(server_url="https://tunnel.example.com/ws/act/")
        self.assertTrue(client3._resolve_ws_url().startswith("wss://tunnel.example.com/ws/act"))
        self.assertEqual(client3._resolve_http_url(), "https://tunnel.example.com/act")


class TestServerPrefillAndCLI(unittest.TestCase):
    """Verifies server assistant prefill adaptation and Showdown evaluator CLI."""

    def test_dynamic_action_intent_prefill(self):
        from avcs.runtime.server import get_assistant_prefill

        # Left click (default)
        act1, p1 = get_assistant_prefill("Click on the search box")
        self.assertEqual(act1, "left_click")
        self.assertIn('"action": "left_click"', p1)

        # Right click
        act2, p2 = get_assistant_prefill("Right click on the taskbar")
        self.assertEqual(act2, "right_click")
        self.assertIn('"action": "right_click"', p2)

        # Double click
        act3, p3 = get_assistant_prefill("Double click the folder on desktop")
        self.assertEqual(act3, "double_click")
        self.assertIn('"action": "double_click"', p3)

    def test_showdown_benchmark_evaluator_cli_run(self):
        import os
        import subprocess
        import sys
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                sys.executable,
                "benchmarks/eval_qwen_showdown.py",
                "--limit", "1",
                "--output", tmp_path,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd="C:\\Users\\nandk\\AVCS-v1")
            self.assertEqual(res.returncode, 0, f"CLI runner failed: {res.stderr}")
            self.assertIn("Primary (Strict 0px)", res.stderr + res.stdout)
            self.assertTrue(os.path.exists(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
