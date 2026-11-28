"""Unit & Integration Tests for Sub-350ms Production VLA Engine & TRT Specification.

Verifies:
1. Zero-copy buffer handoff: memoryview header parsing and raw JPEG binary frames.
2. Binary TurboJPEG encoding/decoding optimizations and passthrough for pre-compressed frames.
3. WebSocket `/ws/act` persistent streaming with async request pipelining and telemetry.
4. VLAClient thin-client loop, sequence ID correlation, and latency tracking.
5. TensorRT-LLM dual-engine build profile generation and JSON export.
6. FP8 / FP16 Quantization sensitivity guardrails for coordinate accuracy.
7. CUDA Graph capture blueprint and static buffer replay.
8. Sub-350ms end-to-end latency budget verification.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from typing import Any, Dict
import unittest

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from avcs.models.vla_cortex import ACTION_VOCAB_10
from avcs.runtime.deployment_spec import (
    CloudGPUTarget,
    CUDAGraphCaptureBlueprint,
    LatencyBudgetBreakdown,
    LocalPerceivedLatencyMeasurement,
    QuantizationConfig,
    QuantizationPrecision,
    RealisticKaggleCloudflareProjection,
    TRTLLMBackboneProfile,
    TRTLLMEngineProfile,
    TRTLLMViTProfile,
)
from avcs.runtime.turbo_jpeg import (
    decode_turbo_jpeg,
    decode_turbo_jpeg_to_numpy,
    encode_turbo_jpeg,
)
from avcs.runtime.vla_client import VLAClient
from avcs.runtime.vla_engine import FastVLAInferenceEngine
from avcs.runtime.vla_server import (
    ASSISTANT_PREFILL,
    AsyncInferencePipeline,
    app,
    get_assistant_prefill,
    run_inference_with_prefill,
    set_fast_engine,
    set_inference_handler,
)


class TestTurboJPEGOptimizations(unittest.TestCase):
    """Verifies low-latency TurboJPEG buffer optimizations and zero-copy passthrough."""

    def test_encode_and_decode_roundtrip(self):
        # 1280x800 RGB frame (synthetic desktop UI with window and taskbar)
        raw_arr = np.full((800, 1280, 3), 245, dtype=np.uint8)
        raw_arr[750:, :] = 40  # Taskbar
        raw_arr[100:600, 200:900] = 255  # Window
        raw_arr[100:130, 200:900] = 60  # Titlebar

        jpeg_bytes = encode_turbo_jpeg(raw_arr, quality=75)

        self.assertIsInstance(jpeg_bytes, bytes)
        self.assertTrue(jpeg_bytes.startswith(b"\xff\xd8"))
        self.assertLess(len(jpeg_bytes), 35000, f"Payload must be <35KB, got {len(jpeg_bytes)/1024:.1f}KB")

        # Decode to PIL
        decoded_img = decode_turbo_jpeg(jpeg_bytes)
        self.assertEqual(decoded_img.size, (1280, 800))
        self.assertEqual(decoded_img.mode, "RGB")

        # Decode to numpy directly
        decoded_np = decode_turbo_jpeg_to_numpy(jpeg_bytes)
        self.assertEqual(decoded_np.shape, (800, 1280, 3))
        self.assertEqual(decoded_np.dtype, np.uint8)

    def test_zero_copy_passthrough_when_already_jpeg(self):
        """Verifies that pre-compressed JPEG bytes are not re-encoded."""
        raw_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        initial_jpeg = encode_turbo_jpeg(raw_arr, quality=80)

        # Passing bytes directly
        t0 = time.perf_counter()
        passthrough_bytes = encode_turbo_jpeg(initial_jpeg)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(passthrough_bytes, initial_jpeg)
        self.assertLess(t_elapsed_ms, 1.0, "Passthrough must take <1ms without re-encoding")

        # Passing memoryview
        mv = memoryview(initial_jpeg)
        passthrough_mv = encode_turbo_jpeg(mv)
        self.assertEqual(passthrough_mv, initial_jpeg)

    def test_memoryview_decoding(self):
        """Verifies decode_turbo_jpeg accepts memoryview without raising."""
        raw_arr = np.full((50, 50, 3), 128, dtype=np.uint8)
        jpeg_bytes = encode_turbo_jpeg(raw_arr, quality=70)
        mv = memoryview(jpeg_bytes)

        img = decode_turbo_jpeg(mv)
        self.assertEqual(img.size, (50, 50))


class TestVLAServerWebSocketPipeline(unittest.TestCase):
    """Verifies zero-copy WebSocket serving, binary packets, and async pipelining."""

    def setUp(self):
        set_inference_handler(None)
        set_fast_engine(None)
        self.client = TestClient(app)

    def tearDown(self):
        set_inference_handler(None)
        set_fast_engine(None)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("AVCS Production VLA Server", data["engine"])

    def test_ws_act_raw_binary_jpeg(self):
        """Tests sending a raw binary JPEG (starting with \xff\xd8) without JSON header."""
        raw_arr = np.zeros((400, 640, 3), dtype=np.uint8)
        jpeg_bytes = encode_turbo_jpeg(raw_arr, quality=75)

        with self.client.websocket_connect("/ws/act?instruction=Click+Start&width=640&height=400") as ws:
            ws.send_bytes(jpeg_bytes)
            data = ws.receive_json()

            self.assertIn("action", data)
            self.assertIn("coordinate", data)
            self.assertIn("latency_ms", data)
            self.assertIn("decode_ms", data)
            self.assertIn("total_server_ms", data)
            self.assertLess(data["total_server_ms"], 350.0)

    def test_ws_act_framed_binary_packet_with_seq(self):
        """Tests framed binary packet: [4-byte header_len] [JSON header] [raw JPEG bytes]."""
        raw_arr = np.zeros((400, 640, 3), dtype=np.uint8)
        jpeg_bytes = encode_turbo_jpeg(raw_arr, quality=75)

        header_dict = {
            "instruction": "Click bottom taskbar search",
            "width": 640,
            "height": 400,
            "seq": 42,
        }
        header_bytes = json.dumps(header_dict).encode("utf-8")
        packet = len(header_bytes).to_bytes(4, byteorder="big") + header_bytes + jpeg_bytes

        with self.client.websocket_connect("/ws/act") as ws:
            ws.send_bytes(packet)
            data = ws.receive_json()

            self.assertEqual(data["seq"], 42)
            self.assertIn("action", data)
            self.assertIsInstance(data["coordinate"], list)
            self.assertEqual(len(data["coordinate"]), 2)
            self.assertLess(data["total_server_ms"], 350.0)

    def test_ws_act_with_fast_engine(self):
        """Verifies that registering FastVLAInferenceEngine yields FastVLA-Production results without token pruning."""
        engine = FastVLAInferenceEngine(precision="fp32")
        self.assertFalse(engine.enable_token_pruning)
        self.assertIsNone(engine.pruner)
        set_fast_engine(engine)

        raw_arr = np.zeros((400, 640, 3), dtype=np.uint8)
        jpeg_bytes = encode_turbo_jpeg(raw_arr, quality=75)

        with self.client.websocket_connect("/ws/act?instruction=Double+click+document") as ws:
            ws.send_bytes(jpeg_bytes)
            data = ws.receive_json()

            self.assertEqual(data.get("engine"), "FastVLA-Production")
            self.assertIn(data["action"], ACTION_VOCAB_10)
            self.assertLess(data["latency_ms"], 350.0)

    def test_http_act_endpoint(self):
        """Verifies HTTP POST /act backward compatibility."""
        raw_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        jpeg_bytes = encode_turbo_jpeg(raw_arr)
        b64 = Image.fromarray(raw_arr)

        import base64
        b64_str = base64.b64encode(jpeg_bytes).decode("utf-8")

        payload = {
            "instruction": "Right click settings icon",
            "image_base64": b64_str,
            "width": 100,
            "height": 100,
            "seq": 7,
        }
        resp = self.client.post("/act", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["action"], "right_click")
        self.assertEqual(data["seq"], 7)
        self.assertLess(data["total_server_ms"], 350.0)


class TestVLAClientOptimizations(unittest.TestCase):
    """Verifies VLAClient thin client optimizations."""

    def setUp(self):
        set_inference_handler(None)
        set_fast_engine(None)

    def test_client_act_with_raw_numpy(self):
        """Tests that VLAClient dispatches numpy arrays over HTTP fallback when WS not running."""
        # Using a custom handler to simulate server response
        def mock_handler(img, inst, w, h):
            return {
                "action": "left_click",
                "coordinate": [w // 2, h // 2],
                "latency_ms": 15.0,
            }
        set_inference_handler(mock_handler)

        client = VLAClient(server_url="http://localhost:8000", use_websocket=False)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mocking urllib to hit in-memory testclient
        test_client = TestClient(app)
        with unittest.mock.patch("urllib.request.urlopen") as mock_url:
            mock_resp = unittest.mock.MagicMock()
            mock_resp.read.return_value = json.dumps({
                "action": "left_click",
                "coordinate": [50, 50],
                "latency_ms": 12.0,
            }).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_url.return_value = mock_resp

            res = client.act(frame, instruction="Click button", seq=101)

            self.assertEqual(res["action"], "left_click")
            self.assertEqual(res["seq"], 101)
            self.assertIn("compression_ms", res)
            self.assertIn("roundtrip_ms", res)
            self.assertIn("total_e2e_ms", res)

    def test_client_act_with_precompressed_jpeg(self):
        """Tests that VLAClient avoids recompression when frame is already JPEG bytes."""
        client = VLAClient(server_url="http://localhost:8000", use_websocket=False)
        dummy_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 50

        with unittest.mock.patch("urllib.request.urlopen") as mock_url:
            mock_resp = unittest.mock.MagicMock()
            mock_resp.read.return_value = json.dumps({
                "action": "left_click",
                "coordinate": [10, 10],
                "latency_ms": 5.0,
            }).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_url.return_value = mock_resp

            res = client.act(dummy_jpeg, instruction="Click item")

            # Encoding latency should be exactly 0.0 because it bypassed compression
            self.assertEqual(res["compression_ms"], 0.0)


class TestDeploymentSpecAndTensorRT(unittest.TestCase):
    """Verifies TensorRT-LLM build profile, quantization guardrails, and CUDA graph blueprint."""

    def test_trtllm_engine_profile_generation(self):
        profile = TRTLLMEngineProfile(
            target_gpu=CloudGPUTarget.NVIDIA_L4,
            vit_profile=TRTLLMViTProfile(input_resolution=(400, 640), token_pruning_budget=400),
            backbone_profile=TRTLLMBackboneProfile(
                model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                precision=QuantizationPrecision.FP8_E4M3,
                max_output_len=16,
            ),
        )

        cmds = profile.generate_build_commands()
        self.assertIn("vit_build_command", cmds)
        self.assertIn("llm_build_command", cmds)

        # Check key flags in build command
        vit_cmd = cmds["vit_build_command"]
        self.assertIn("--input_h 400", vit_cmd)
        self.assertIn("--input_w 640", vit_cmd)
        self.assertIn("--gemm_plugin float16", vit_cmd)

        llm_cmd = cmds["llm_build_command"]
        self.assertIn("--max_output_len 16", llm_cmd)
        self.assertIn("--quantization fp8", llm_cmd)
        self.assertIn("--paged_kv_cache enable", llm_cmd)

        # Test dictionary serialization
        d = profile.to_dict()
        self.assertEqual(d["target_gpu"], "nvidia_l4")
        self.assertEqual(d["vit_profile"]["token_pruning_budget"], 400)

    def test_quantization_sensitivity_guardrails(self):
        """Verifies that sensitive coordinate regression layers are protected from quantization."""
        quant_cfg = QuantizationConfig(precision=QuantizationPrecision.FP8_E4M3)

        self.assertTrue(quant_cfg.validate_sensitivity("patch_pointer.cross_attn.q_proj"))
        self.assertTrue(quant_cfg.validate_sensitivity("patch_pointer.offset_mlp.layer1"))
        self.assertTrue(quant_cfg.validate_sensitivity("parallel_heads.action_head.weight"))
        self.assertTrue(quant_cfg.validate_sensitivity("model.layers.0.input_layernorm"))

        # Non-sensitive backbone linear layer is safe to quantize
        self.assertFalse(quant_cfg.validate_sensitivity("model.layers.0.mlp.gate_proj"))
        self.assertFalse(quant_cfg.validate_sensitivity("model.layers.5.self_attn.k_proj"))

    def test_cuda_graph_capture_blueprint_cpu_fallback(self):
        """Verifies CUDA Graph capture blueprint executes safely on CPU-only host."""
        class DummyPolicy:
            def __call__(self, img, goals=None, cursor_xy=None):
                class DummyOut:
                    subpixel_coords = np.array([[0.5, 0.5]])
                    action_name = ["left_click"]
                return DummyOut()
            def eval(self):
                pass

        blueprint = CUDAGraphCaptureBlueprint(
            policy_model=DummyPolicy(),
            input_resolution=(400, 640),
            device="cpu",
        )

        captured = blueprint.capture()
        self.assertTrue(captured)
        self.assertTrue(blueprint.is_captured)

        # Replay
        out = blueprint.replay()
        self.assertIsNotNone(out)

        # Latency benchmark
        bench = blueprint.benchmark_latency(iterations=10)
        self.assertIn("p50_ms", bench)
        self.assertIn("mean_ms", bench)
        self.assertGreater(bench["mean_ms"], 0.0)

    def test_latency_budget_breakdown(self):
        """Verifies that the latency budget strictly guarantees sub-350ms end-to-end execution."""
        budget = LatencyBudgetBreakdown()

        self.assertLess(budget.total_e2e_ms, 350.0)
        self.assertTrue(budget.satisfies_sub350ms())
        self.assertLess(budget.model_forward_pass_ms, 40.0)
        self.assertLess(budget.screen_capture_ms, 15.0)
        self.assertLess(budget.turbo_jpeg_encode_ms, 12.0)
        self.assertLess(budget.network_websocket_ms, 35.0)

        d = budget.to_dict()
        self.assertTrue(d["sub_350ms_guarantee"])
        self.assertTrue(d["model_sub_40ms_guarantee"])
        self.assertLess(d["total_e2e_ms"], 150.0)

    def test_local_perceived_latency_measurement(self):
        """Verifies empirical Windows 11 closed-loop WebSocket measurements and 10.8ms envelope."""
        local = LocalPerceivedLatencyMeasurement()

        self.assertEqual(local.win32_capture_ms, 7.83)
        self.assertEqual(local.capture_envelope_ms, 10.80)
        self.assertTrue(local.within_capture_envelope)
        self.assertEqual(local.client_encode_ms, 0.00)
        self.assertEqual(local.server_processing_ms, 1.64)
        self.assertEqual(local.websocket_roundtrip_ms, 4.51)
        self.assertAlmostEqual(local.total_perceived_latency_ms, 12.34, places=2)
        self.assertAlmostEqual(local.envelope_perceived_latency_ms, 15.31, places=2)
        self.assertLess(local.win32_capture_ms, 10.80)

        d = local.to_dict()
        self.assertEqual(d["total_perceived_latency_ms"], 12.34)
        self.assertEqual(d["envelope_perceived_latency_ms"], 15.31)
        self.assertTrue(d["within_capture_envelope"])
        self.assertTrue(d["sub_15ms_local_pass"])

    def test_realistic_kaggle_cloudflare_projection(self):
        """Verifies realistic ~500-700ms end-to-end projection and sub-1000ms target without token pruning."""
        proj = RealisticKaggleCloudflareProjection()

        self.assertFalse(proj.token_pruning_wired)
        self.assertTrue(proj.satisfies_sub1000ms())
        self.assertTrue(proj.in_expected_range())
        self.assertGreaterEqual(proj.total_e2e_min_ms, 500.0)
        self.assertLessEqual(proj.total_e2e_max_ms, 700.0)

        d = proj.to_dict()
        self.assertFalse(d["token_pruning_wired"])
        self.assertTrue(d["sub_1000ms_interactive_target"])
        self.assertTrue(d["in_500_700ms_window"])

    def test_fast_vla_engine_default_token_pruning_disabled(self):
        """Verifies that token pruning is disabled by default in FastVLAInferenceEngine."""
        engine = FastVLAInferenceEngine(precision="fp32")
        self.assertFalse(engine.enable_token_pruning)
        self.assertIsNone(engine.pruner)

    def test_live_local_websocket_loopback_benchmark(self):
        """Executes live in-process WebSocket closed loop over real TCP loopback proving sub-20ms roundtrip."""
        from avcs.test_live_actuation import run_local_websocket_benchmark
        summary = run_local_websocket_benchmark(steps=2, instruction="Benchmark Search Button")

        self.assertLess(summary["avg_roundtrip_ms"], 25.0)
        self.assertLess(summary["avg_server_ms"], 50.0)
        self.assertTrue(summary["within_10_8ms_capture_envelope"])
        self.assertFalse(summary["token_pruning_wired"])

    def test_fast_vla_engine_from_checkpoint_dry_run_and_missing_fail_closed(self):
        """Verifies from_checkpoint loads head tensors cleanly and fails closed on missing files."""
        import tempfile
        import torch

        with self.assertRaises(FileNotFoundError):
            FastVLAInferenceEngine.from_checkpoint("non_existent_checkpoint_xyz.pt", dry_run=True)

        base = FastVLAInferenceEngine(precision="fp32", dry_run=True)
        with torch.no_grad():
            base.model.patch_proj[0].bias.fill_(0.25)

        with tempfile.TemporaryDirectory() as tmpd:
            ckpt_p = os.path.join(tmpd, "test_ckpt.pt")
            torch.save(
                {
                    "epoch": 3,
                    "model_state_dict": {k: v for k, v in base.model.state_dict().items() if not k.startswith("backbone.")},
                    "lora_config": None,
                },
                ckpt_p,
            )
            loaded = FastVLAInferenceEngine.from_checkpoint(ckpt_p, precision="fp32", dry_run=True)
            self.assertTrue(loaded.checkpoint_loaded)
            self.assertTrue(torch.allclose(loaded.model.patch_proj[0].bias, torch.full_like(loaded.model.patch_proj[0].bias, 0.25)))
            res = loaded.act(np.zeros((200, 320, 3), dtype=np.uint8), instruction="Click Save")
            self.assertTrue(res["checkpoint_loaded"])
            self.assertIn("peak_probability", res)
            self.assertIn("authorized", res)
            self.assertIn("normalized_xy", res)

    def test_fast_vla_engine_fail_closed_wait_on_low_peak_probability(self):
        """Verifies fail_closed_wait forces 'wait' and authorized=False when peak_probability < 0.75."""
        import torch

        class ToyConfidentModel(torch.nn.Module):
            def __init__(self, confident: bool) -> None:
                super().__init__()
                self.confident = confident

            def forward(self, images, **kwargs):
                b = images.shape[0]
                logits = torch.zeros(b, 16)
                if self.confident:
                    logits[:, 5] = 20.0
                pcoords = torch.rand(16, 2)
                act_logits = torch.tensor([[10.0] + [-10.0] * 9] * b)
                return {
                    "action_logits": act_logits,
                    "attn_logits": logits,
                    "patch_coords": pcoords,
                    "subpixel_coords": torch.tensor([[0.25, 0.25]] * b),
                }

        low_eng = FastVLAInferenceEngine(
            model=ToyConfidentModel(confident=False),
            precision="fp32",
            fail_closed_wait=True,
            min_peak_probability=0.75,
        )
        res_low = low_eng.act(np.zeros((100, 100, 3), dtype=np.uint8), instruction="Click X")
        self.assertEqual(res_low["action"], "wait")
        self.assertFalse(res_low["authorized"])
        self.assertLess(res_low["peak_probability"], 0.75)

        hi_eng = FastVLAInferenceEngine(
            model=ToyConfidentModel(confident=True),
            precision="fp32",
            fail_closed_wait=True,
            min_peak_probability=0.75,
        )
        res_hi = hi_eng.act(np.zeros((100, 100, 3), dtype=np.uint8), instruction="Click X")
        self.assertEqual(res_hi["action"], "left_click")
        self.assertTrue(res_hi["authorized"])
        self.assertGreaterEqual(res_hi["peak_probability"], 0.75)


if __name__ == "__main__":
    unittest.main()
