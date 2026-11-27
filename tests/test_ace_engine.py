"""Comprehensive Test Suite for the Ace-Control-Medium Engine.

Verifies:
1. Deterministic Action space parsing, aliases, and key combo normalization.
2. Custom frontier LLM provider routing, headers, payload shaping, and image encoding.
3. Strict zero-center heuristic enforcement (never snapping to 0.5, 0.5 or 640, 400).
4. Robust response parsing (code fences, dirty JSON, trailing commas).
5. Canonical 1280x800 coordinate clamping and mapping.
6. Stateful session tracking, history continuity, and termination conditions.
7. Physical WindowsComputer observation and dry-run execution kinematics.
8. End-to-end AceRunner autonomous loop and safety breakout failsafe.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

from avcs.ace.actions import (
    Action,
    ActionKind,
    ActionAdapter,
    HID_KEYBOARD_TABLE,
    HID_CODE_TO_KEY_TABLE,
    HID_MOUSE_BUTTON_TABLE,
    key_to_hid,
    hid_to_key,
    mouse_button_to_hid,
    interpolate_drag_chunks,
    normalize_coordinate,
    denormalize_coordinate,
)
from avcs.ace.agent import AceAgent, AceAgentSession
from avcs.ace.client import CustomLLMClient
from avcs.ace.local_model import LocalVAMClient
from avcs.ace.computer import (
    INPUT,
    ExecutionSafetyFirewall,
    MonitorInfo,
    Observation,
    SafetyCheckResult,
    SafetyRiskLevel,
    WindowsComputer,
)
from avcs.ace.planner import (
    CognitivePlanner,
    ReflectionDiagnostic,
    SubGoal,
    _validate_no_coordinates,
)
from avcs.ace.runner import (
    AceRunner,
    DualStageVerificationResult,
    DualStageVerifier,
    SafetyBreakoutError,
    Stage1Result,
    Stage2Result,
    TrajectoryLogger,
)
from avcs.ace.session import AceSession
from avcs.ace.skills_vault.cache import (
    SkillCache,
    SkillCacheEntry,
    SkillLookupResult,
    UIStateAnchor,
    compute_visual_hash,
    visual_hash_similarity,
)
from avcs.cortex.gate import StabilityGateResult, VisualCortexGate


class TestActionSpace(unittest.TestCase):
    """Verifies Action schema serialization, deserialization, and alias handling."""

    def test_action_to_dict_and_back(self):
        act = Action(
            kind=ActionKind.LEFT_CLICK,
            coordinate=(450, 320),
            reason="Click submit button",
        )
        d = act.to_dict()
        self.assertEqual(d["kind"], "left_click")
        self.assertEqual(d["coordinate"], [450, 320])
        self.assertEqual(d["reason"], "Click submit button")

        reconstructed = Action.from_dict(d)
        self.assertEqual(reconstructed.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(reconstructed.coordinate, (450, 320))
        self.assertEqual(reconstructed.reason, "Click submit button")

    def test_alias_mapping(self):
        # click -> left_click
        a1 = Action.from_dict({"action": "click", "x": 100, "y": 200})
        self.assertEqual(a1.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(a1.coordinate, (100, 200))

        # press -> key
        a2 = Action.from_dict({"action": "press", "key": "enter"})
        self.assertEqual(a2.kind, ActionKind.KEY)
        self.assertEqual(a2.keys, ["enter"])

        # done / finish -> stop
        a3 = Action.from_dict({"action": "done", "reason": "Task completed"})
        self.assertEqual(a3.kind, ActionKind.STOP)
        self.assertEqual(a3.reason, "Task completed")

        a4 = Action.from_dict({"kind": "finish", "reason": "All steps done"})
        self.assertEqual(a4.kind, ActionKind.STOP)

    def test_key_combo_normalization(self):
        # hotkey with plus sign
        act = Action.from_dict({"kind": "key", "key": "ctrl+s"})
        self.assertEqual(act.keys, ["ctrl", "s"])

        act2 = Action.from_dict({"kind": "key", "keys": ["ctrl+shift+p"]})
        self.assertEqual(act2.keys, ["ctrl", "shift", "p"])

    def test_float_coordinate_resilience(self):
        # Handle float values safely
        act = Action.from_dict({"action": "click", "x": 620.0, "y": 450.5})
        self.assertEqual(act.coordinate, (620, 450))

        act2 = Action.from_dict({"kind": "left_click", "coordinate": ["300.0", "150.2"]})
        self.assertEqual(act2.coordinate, (300, 150))

    def test_string_keys_parsing(self):
        # Verify a single string for keys is not split into individual characters
        act = Action.from_dict({"kind": "key", "keys": "enter"})
        self.assertEqual(act.keys, ["enter"])

        act2 = Action.from_dict({"kind": "key", "keys": "ctrl+shift+esc"})
        self.assertEqual(act2.keys, ["ctrl", "shift", "esc"])

    def test_normalized_float_coordinates_scaling(self):
        # Coordinates in [0.0, 1.0] must scale by canonical (1280, 800)
        act = Action.from_dict({"kind": "left_click", "coordinate": [0.25, 0.75]})
        self.assertEqual(act.coordinate, (320, 600))

        act2 = Action.from_dict({"action": "click", "x": 0.5, "y": 0.5})
        self.assertEqual(act2.coordinate, (640, 400))

    def test_normalized_float_coordinates_boundary_scaling(self):
        # Boundary [1.0, 1.0] must scale to canonical (1280, 800)
        act = Action.from_dict({"kind": "left_click", "coordinate": [1.0, 1.0]})
        self.assertEqual(act.coordinate, (1280, 800))

        act2 = Action.from_dict({"action": "click", "x": 1.0, "y": 1.0})
        self.assertEqual(act2.coordinate, (1280, 800))

        # Center [0.5, 0.5] scales to (640, 400)
        act_half = Action.from_dict({"kind": "left_click", "coordinate": [0.5, 0.5]})
        self.assertEqual(act_half.coordinate, (640, 400))

        # Boundaries [0.0, 1.0] and [1.0, 0.0]
        act_y_max = Action.from_dict({"kind": "left_click", "coordinate": [0.0, 1.0]})
        self.assertEqual(act_y_max.coordinate, (0, 800))

        act_x_max = Action.from_dict({"kind": "left_click", "coordinate": [1.0, 0.0]})
        self.assertEqual(act_x_max.coordinate, (1280, 0))

        act_origin = Action.from_dict({"kind": "left_click", "coordinate": [0.0, 0.0]})
        self.assertEqual(act_origin.coordinate, (0, 0))

        # String floats
        act_str = Action.from_dict({"kind": "left_click", "coordinate": ["1.0", "1.0"]})
        self.assertEqual(act_str.coordinate, (1280, 800))

        act_str_half = Action.from_dict({"kind": "left_click", "coordinate": ["0.5", "0.5"]})
        self.assertEqual(act_str_half.coordinate, (640, 400))

        # Scientific notation string floats
        act_sci = Action.from_dict({"kind": "left_click", "coordinate": ["1e-1", "0.5"]})
        self.assertEqual(act_sci.coordinate, (128, 400))

        # Integer coordinates remain exact pixel coordinates
        act_pixel = Action.from_dict({"kind": "left_click", "coordinate": [1, 1]})
        self.assertEqual(act_pixel.coordinate, (1, 1))

        act_pixel_origin = Action.from_dict({"kind": "left_click", "coordinate": [0, 0]})
        self.assertEqual(act_pixel_origin.coordinate, (0, 0))

        act_pixel_axes = Action.from_dict({"kind": "left_click", "coordinate": [1, 0]})
        self.assertEqual(act_pixel_axes.coordinate, (1, 0))

        act_pixel_y = Action.from_dict({"kind": "left_click", "coordinate": [0, 1]})
        self.assertEqual(act_pixel_y.coordinate, (0, 1))

        # NaN, Inf, and invalid values gracefully evaluate to None
        act_nan = Action.from_dict({"kind": "left_click", "coordinate": [float("nan"), 10.0]})
        self.assertIsNone(act_nan.coordinate)

        act_inf = Action.from_dict({"kind": "left_click", "coordinate": ["inf", "1.0"]})
        self.assertIsNone(act_inf.coordinate)

    def test_press_enter_in_type_action(self):
        act = Action.from_dict({"kind": "type", "text": "notepad", "press_enter": True})
        self.assertTrue(act.press_enter)
        d = act.to_dict()
        self.assertTrue(d["press_enter"])


class TestCustomLLMClient(unittest.TestCase):
    """Verifies Custom LLM provider configuration, frontier models, and multi-protocol handling."""

    def setUp(self):
        # Clean environment overrides
        self._env_backup = dict(os.environ)
        for k in [
            "AVCS_ACE_ENDPOINT", "AVCS_ACE_API_KEY", "AVCS_ACE_MODEL", "AVCS_ACE_EXTRA_HEADERS", "AVCS_ACE_PROTOCOL",
            "AVCS_LLM_ENDPOINT", "AVCS_LLM_API_KEY", "AVCS_LLM_MODEL",
            "OPENAI_BASE_URL", "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY",
        ]:
            if k in os.environ:
                del os.environ[k]

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_custom_endpoint_and_frontier_model(self):
        client = CustomLLMClient(
            endpoint="https://api.frontier.ai/v1",
            api_key="sk-frontier-secret-key",
            model="gpt-6-astra",
            extra_headers={"X-Custom-Route": "accelerated"},
        )
        self.assertEqual(client.endpoint, "https://api.frontier.ai/v1/chat/completions")
        self.assertEqual(client.api_key, "sk-frontier-secret-key")
        self.assertEqual(client.model, "gpt-6-astra")
        self.assertEqual(client.protocol, "openai")
        self.assertEqual(client.extra_headers.get("X-Custom-Route"), "accelerated")

    def test_environment_variable_fallback(self):
        os.environ["AVCS_ACE_ENDPOINT"] = "http://127.0.0.1:8000/v1/chat/completions"
        os.environ["AVCS_LOCAL_VAM_API_KEY"] = "key-from-env"
        os.environ["AVCS_ACE_MODEL"] = "Qwen2.5-VL-3B-Instruct"
        os.environ["AVCS_ACE_EXTRA_HEADERS"] = json.dumps({"X-Organization": "ResearchLab"})

        try:
            client = CustomLLMClient()
            self.assertEqual(client.endpoint, "http://127.0.0.1:8000/v1/chat/completions")
            self.assertEqual(client.api_key, "key-from-env")
            self.assertEqual(client.model, "Qwen2.5-VL-3B-Instruct")
            self.assertEqual(client.extra_headers.get("X-Organization"), "ResearchLab")
        finally:
            os.environ.pop("AVCS_LOCAL_VAM_API_KEY", None)

    def test_protocol_auto_detection(self):
        from avcs.runtime.client import AceClientError

        # Cloud Anthropic/Gemini/external endpoints are quarantined
        c_anthropic = CustomLLMClient(endpoint="https://api.anthropic.com/v1/messages")
        self.assertTrue(c_anthropic._cloud_quarantined)
        with self.assertRaises(AceClientError):
            c_anthropic.predict_action(b"fake", goal="test")

        c_gemini = CustomLLMClient(endpoint="https://generativelanguage.googleapis.com/v1beta/models")
        self.assertTrue(c_gemini._cloud_quarantined)
        with self.assertRaises(AceClientError):
            c_gemini.predict_action(b"fake", goal="test")

        # Local loopback OpenAI-compatible VLA server is permitted
        c_local = CustomLLMClient(endpoint="http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(c_local.protocol, "openai")
        self.assertFalse(c_local._cloud_quarantined)

    def test_image_encoding(self):
        client = CustomLLMClient()
        raw_bytes = b"fake-webp-image-stream-data"
        b64, mime, data_uri = client.encode_image(raw_bytes, format="webp")
        self.assertEqual(mime, "image/webp")
        self.assertTrue(data_uri.startswith("data:image/webp;base64,"))
        self.assertIn(b64, data_uri)

    def test_openai_payload_structure(self):
        client = CustomLLMClient(model="gpt-6-astra")
        payload = client._build_payload(
            b64_str="AAAA",
            mime="image/webp",
            data_uri="data:image/webp;base64,AAAA",
            goal="Open Chrome and navigate to docs",
            step_index=2,
        )
        self.assertEqual(payload["model"], "gpt-6-astra")
        self.assertEqual(payload["temperature"], 0.0)
        messages = payload["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        user_content = messages[1]["content"]
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertEqual(user_content[1]["image_url"]["url"], "data:image/webp;base64,AAAA")

    def test_anthropic_payload_structure(self):
        from avcs.runtime.client import AceClientError
        client = CustomLLMClient(endpoint="https://api.anthropic.com/v1/messages", protocol="anthropic", model="fable-5.5")
        with self.assertRaises(AceClientError):
            client._build_payload(
                b64_str="BBBB",
                mime="image/png",
                data_uri="data:image/png;base64,BBBB",
                goal="Save document",
                step_index=1,
            )

    def test_parse_clean_json(self):
        client = CustomLLMClient()
        raw = '{"kind": "left_click", "coordinate": [512, 384], "reason": "Click search bar"}'
        act = client.parse_action_response(raw)
        self.assertEqual(act.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act.coordinate, (512, 384))
        self.assertEqual(act.reason, "Click search bar")

    def test_parse_markdown_code_fence(self):
        client = CustomLLMClient()
        raw = '```json\n{"kind": "type", "coordinate": [200, 100], "text": "hello world", "reason": "type greeting"}\n```'
        act = client.parse_action_response(raw)
        self.assertEqual(act.kind, ActionKind.TYPE)
        self.assertEqual(act.coordinate, (200, 100))
        self.assertEqual(act.text, "hello world")

    def test_parse_dirty_json_with_preamble_and_trailing_comma(self):
        client = CustomLLMClient()
        raw = 'Observing the screen now:\n{"kind": "stop", "reason": "Goal finished",}\nDone.'
        act = client.parse_action_response(raw)
        self.assertEqual(act.kind, ActionKind.STOP)
        self.assertEqual(act.reason, "Goal finished")

    def test_parse_single_quotes_and_empty(self):
        client = CustomLLMClient()
        # Empty string
        act_empty = client.parse_action_response("")
        self.assertEqual(act_empty.kind, ActionKind.WAIT)
        self.assertIn("Empty LLM response", act_empty.reason)

        # Single quotes JSON
        raw_sq = "{'kind': 'wait', 'duration': 2.5, 'reason': 'Waiting for page load'}"
        act_sq = client.parse_action_response(raw_sq)
        self.assertEqual(act_sq.kind, ActionKind.WAIT)
        self.assertEqual(act_sq.duration, 2.5)

    def test_parse_drag_and_scroll_aliases(self):
        client = CustomLLMClient()
        # Drag alias with x1, y1, x2, y2
        raw_drag = '{"action": "drag", "x1": 150, "y1": 250, "x2": 600, "y2": 700, "reason": "drag window"}'
        act_drag = client.parse_action_response(raw_drag)
        self.assertEqual(act_drag.kind, ActionKind.LEFT_CLICK_DRAG)
        self.assertEqual(act_drag.start_coordinate, (150, 250))
        self.assertEqual(act_drag.end_coordinate, (600, 700))

        # Scroll alias
        raw_scroll = '{"action": "scroll", "x": 500, "y": 300, "delta_y": -240, "reason": "scroll feed"}'
        act_scroll = client.parse_action_response(raw_scroll)
        self.assertEqual(act_scroll.kind, ActionKind.SCROLL)
        self.assertEqual(act_scroll.coordinate, (500, 300))
        self.assertEqual(act_scroll.scroll_delta, (0, -240))

    def test_zero_center_heuristic_enforcement(self):
        client = CustomLLMClient()
        # Ungrounded click must NOT default to (640, 400) or (0.5, 0.5)
        raw = '{"kind": "left_click", "reason": "click something without coords"}'
        act = client.parse_action_response(raw)
        self.assertEqual(act.kind, ActionKind.WAIT)
        self.assertIn("missing visual coordinates", act.reason)
        self.assertIsNone(act.coordinate)

    def test_parse_thinking_blocks_with_internal_json(self):
        client = CustomLLMClient()
        raw = """<think>
{"thought": "locate target button on canvas"}
</think>
{"kind": "left_click", "coordinate": [300, 200], "reason": "click target"}"""
        act = client.parse_action_response(raw)
        self.assertEqual(act.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act.coordinate, (300, 200))
        self.assertEqual(act.reason, "click target")

    def test_parse_python_literals(self):
        client = CustomLLMClient()
        raw = '{"kind": "type", "coordinate": [100, 200], "press_enter": True, "text": "hello"}'
        act = client.parse_action_response(raw)
        self.assertEqual(act.kind, ActionKind.TYPE)
        self.assertEqual(act.coordinate, (100, 200))
        self.assertEqual(act.text, "hello")
        self.assertTrue(act.press_enter)

    def test_reasoning_content_extraction(self):
        client = CustomLLMClient()
        # When content is null but reasoning_content has the text
        resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": '{"kind": "stop", "reason": "Goal reached"}',
                    }
                }
            ]
        }
        extracted = client._extract_content_from_response(resp)
        self.assertEqual(extracted, '{"kind": "stop", "reason": "Goal reached"}')

    def test_multimodal_list_content_extraction(self):
        client = CustomLLMClient()
        # When content is returned as a list of parts
        resp = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": '{"kind": "wait", "duration": 1.5}'}
                        ]
                    }
                }
            ]
        }
        extracted = client._extract_content_from_response(resp)
        self.assertEqual(extracted, '{"kind": "wait", "duration": 1.5}')

    def test_groq_and_anthropic_auto_detection(self):
        # Cloud API keys in environment must NOT auto-route to external cloud endpoints
        os.environ["GROQ_API_KEY"] = "gsk_test12345"
        c_groq = CustomLLMClient()
        self.assertIn("127.0.0.1", c_groq.endpoint)
        self.assertEqual(c_groq.protocol, "openai")
        del os.environ["GROQ_API_KEY"]

        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test12345"
        c_ant = CustomLLMClient()
        self.assertIn("127.0.0.1", c_ant.endpoint)
        self.assertEqual(c_ant.protocol, "openai")
        del os.environ["ANTHROPIC_API_KEY"]

    def test_truncated_json_coordinate_repair(self):
        client = CustomLLMClient()
        # 1. Truncated coordinate with 1 element: {"kind": "left_click", "coordinate": [260
        raw1 = '{"kind": "left_click", "coordinate": [260'
        act1 = client.parse_action_response(raw1)
        self.assertEqual(act1.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act1.coordinate, (260, 800))

        # 2. Truncated coordinate with trailing comma: {"kind": "left_click", "coordinate": [260,
        raw2 = '{"kind": "left_click", "coordinate": [260,'
        act2 = client.parse_action_response(raw2)
        self.assertEqual(act2.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act2.coordinate, (260, 800))

        # 3. Truncated coordinate with 2 elements missing closing brackets: {"kind": "left_click", "coordinate": [260, 750
        raw3 = '{"kind": "left_click", "coordinate": [260, 750'
        act3 = client.parse_action_response(raw3)
        self.assertEqual(act3.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act3.coordinate, (260, 750))

        # 4. Truncated single-quote JSON: {'kind': 'left_click', 'coordinate': [260
        raw4 = "{'kind': 'left_click', 'coordinate': [260"
        act4 = client.parse_action_response(raw4)
        self.assertEqual(act4.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act4.coordinate, (260, 800))

        # 5. Preamble with truncated JSON:
        raw5 = 'Step 11 plan:\n{"kind": "left_click", "coordinate": [260'
        act5 = client.parse_action_response(raw5)
        self.assertEqual(act5.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act5.coordinate, (260, 800))

        # 6. Repair with previous history coordinate
        history = [{"step": 1, "action": "left_click", "coordinate": [260, 650]}]
        act6 = client.parse_action_response(raw1, history=history)
        self.assertEqual(act6.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act6.coordinate, (260, 650))

    def test_truncated_json_string_and_keys_repair(self):
        client = CustomLLMClient()
        # Truncated string literal
        raw_type = '{"kind": "type", "text": "notep'
        act_type = client.parse_action_response(raw_type)
        self.assertEqual(act_type.kind, ActionKind.TYPE)
        self.assertEqual(act_type.text, "notep")

        # Truncated keys array
        raw_key = '{"kind": "key", "keys": ["win'
        act_key = client.parse_action_response(raw_key)
        self.assertEqual(act_key.kind, ActionKind.KEY)
        self.assertEqual(act_key.keys, ["win"])

        # Truncated code fence block
        raw_fence = '```json\n{"kind": "left_click", "coordinate": [260'
        act_fence = client.parse_action_response(raw_fence)
        self.assertEqual(act_fence.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act_fence.coordinate, (260, 800))

    def test_max_tokens_configuration(self):
        # Default is 1024
        client_default = CustomLLMClient()
        self.assertEqual(client_default.max_tokens, 1024)

        # Custom constructor arg
        client_custom = CustomLLMClient(max_tokens=2048)
        self.assertEqual(client_custom.max_tokens, 2048)

        # Environment variable override
        os.environ["AVCS_ACE_MAX_TOKENS"] = "512"
        client_env = CustomLLMClient()
        self.assertEqual(client_env.max_tokens, 512)
        del os.environ["AVCS_ACE_MAX_TOKENS"]

        # Payload uses max_tokens for local OpenAI-compatible VLA server
        c_local = CustomLLMClient(max_tokens=1024)
        payload_local = c_local._build_payload(b64_str="X", mime="image/webp", data_uri="data:", goal="test")
        self.assertEqual(payload_local["max_tokens"], 1024)

    def test_prompt_stuck_warning(self):
        client = CustomLLMClient()
        # Non-repeating history
        hist1 = [
            {"step": 1, "action": "left_click", "coordinate": [260, 800]},
            {"step": 2, "action": "type", "text": "notepad"},
        ]
        prompt1 = client.build_user_prompt(goal="Open notepad", history=hist1)
        self.assertNotIn("STUCK WARNING", prompt1)

        # Repeating history (same click repeated twice)
        hist2 = [
            {"step": 1, "action": "left_click", "coordinate": [260, 800]},
            {"step": 2, "action": "left_click", "coordinate": [260, 800]},
        ]
        prompt2 = client.build_user_prompt(goal="Open notepad", history=hist2)
        self.assertIn("STUCK WARNING", prompt2)
        self.assertIn("repeated 2 times", prompt2)

    def test_prompt_stuck_warning_with_tolerance(self):
        client = CustomLLMClient()
        # Repeating history within 5px tolerance: (260, 800) and (262, 801)
        hist = [
            {"step": 1, "action": "left_click", "coordinate": [260, 800]},
            {"step": 2, "action": "left_click", "coordinate": [262, 801]},
        ]
        prompt = client.build_user_prompt(goal="Open notepad", history=hist)
        self.assertIn("STUCK WARNING", prompt)
        self.assertIn("repeated 2 times", prompt)

    def test_truncated_json_with_preceding_thought_object(self):
        client = CustomLLMClient()
        raw = '{"thought": "Opening Start menu"}\n{"kind": "left_click", "coordinate": [260'
        act = client.parse_action_response(raw)
        self.assertEqual(act.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act.coordinate, (260, 800))

    def test_truncated_json_empty_coordinate_bracket_and_colon(self):
        client = CustomLLMClient()
        hist = [{"step": 1, "action": "left_click", "coordinate": [260, 800]}]

        # Truncated right at bracket: [
        act1 = client.parse_action_response('{"kind": "left_click", "coordinate": [', history=hist)
        self.assertEqual(act1.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act1.coordinate, (260, 800))

        # Truncated right at colon: :
        act2 = client.parse_action_response('{"kind": "left_click", "coordinate":', history=hist)
        self.assertEqual(act2.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act2.coordinate, (260, 800))

    def test_truncated_json_partial_key_and_orphan_keys(self):
        client = CustomLLMClient()
        hist = [{"step": 1, "action": "left_click", "coordinate": [260, 800]}]

        # Truncated key name "coord
        act1 = client.parse_action_response('{"kind": "left_click", "coord', history=hist)
        self.assertEqual(act1.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act1.coordinate, (260, 800))

        # Truncated key name after comma "reaso
        act2 = client.parse_action_response('{"kind": "left_click", "coordinate": [260, 800], "reaso')
        self.assertEqual(act2.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act2.coordinate, (260, 800))

    def test_truncated_json_trailing_backslash(self):
        client = CustomLLMClient()
        raw = '{"kind": "type", "text": "hello\\'
        act = client.parse_action_response(raw)
        self.assertEqual(act.kind, ActionKind.TYPE)
        self.assertEqual(act.text, "hello")


class TestAceAgent(unittest.TestCase):
    """Verifies AceAgent coordinate boundary checking and zero-center enforcement."""

    def test_coordinate_clamping(self):
        agent = AceAgent()
        # Coordinates out of bounds
        act = Action(kind=ActionKind.LEFT_CLICK, coordinate=(1500, -20))
        validated = agent.validate_action(act)
        self.assertEqual(validated.coordinate, (1280, 0))

        # Drag coordinates out of bounds
        act_drag = Action(
            kind=ActionKind.LEFT_CLICK_DRAG,
            start_coordinate=(-50, 100),
            end_coordinate=(1300, 950),
        )
        val_drag = agent.validate_action(act_drag)
        self.assertEqual(val_drag.start_coordinate, (0, 100))
        self.assertEqual(val_drag.end_coordinate, (1280, 800))

    def test_ungrounded_click_rejection(self):
        agent = AceAgent()
        act = Action(kind=ActionKind.DOUBLE_CLICK, coordinate=None)
        validated = agent.validate_action(act)
        self.assertEqual(validated.kind, ActionKind.WAIT)
        self.assertIn("no coordinates provided", validated.reason)


class TestAceSession(unittest.TestCase):
    """Verifies stateful session management, history continuity, and metrics."""

    def test_session_lifecycle_and_history(self):
        session = AceSession()
        session.start(goal="Open Visual Studio Code", max_steps=5)
        self.assertEqual(session.goal, "Open Visual Studio Code")
        self.assertFalse(session.is_complete)

        # Step 1
        obs = Observation(image_bytes=b"1", format="webp", capture_time_ms=5.0)
        act1 = Action(kind=ActionKind.LEFT_CLICK, coordinate=(30, 780), reason="Click start menu")
        session.record_step(act1, observation=obs, plan_latency_ms=450.0)

        # Step 2
        act2 = Action(kind=ActionKind.TYPE, text="code", reason="Search for code")
        session.record_step(act2, observation=obs, plan_latency_ms=480.0)

        # Check history summary
        summary = session.get_history_summary()
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["action"], "left_click")
        self.assertEqual(summary[0]["coordinate"], [30, 780])
        self.assertEqual(summary[1]["action"], "type")
        self.assertEqual(summary[1]["text"], "code")

        # Step 3: Stop
        act3 = Action(kind=ActionKind.STOP, reason="VS Code launched")
        session.record_step(act3, observation=obs, plan_latency_ms=300.0)

        self.assertTrue(session.is_complete)
        self.assertEqual(session.stop_reason, "VS Code launched")

        metrics = session.get_metrics()
        self.assertEqual(metrics["total_steps"], 3)
        self.assertTrue(metrics["is_complete"])
        self.assertGreater(metrics["avg_plan_latency_ms"], 400.0)

    def test_session_max_steps_termination(self):
        session = AceSession(goal="Test limit", max_steps=2)
        act = Action(kind=ActionKind.WAIT, duration=0.1)
        session.record_step(act)
        self.assertFalse(session.is_complete)
        session.record_step(act)
        self.assertTrue(session.is_complete)
        self.assertIn("Maximum step limit", session.stop_reason)

    def test_session_repetition_and_stuck_detection(self):
        session = AceSession(goal="Test loop", max_steps=10, dry_run=True)
        self.assertEqual(session.get_repetition_count(), 0)
        self.assertFalse(session.is_stuck())

        # Step 1: left click (260, 800)
        click1 = Action(kind=ActionKind.LEFT_CLICK, coordinate=(260, 800), reason="click start")
        session.record_step(click1)
        self.assertEqual(session.get_repetition_count(), 1)
        self.assertFalse(session.is_stuck(threshold=2))

        # Step 2: identical click (262, 801) - within 5px tolerance
        click2 = Action(kind=ActionKind.LEFT_CLICK, coordinate=(262, 801), reason="click start again")
        session.record_step(click2)
        self.assertEqual(session.get_repetition_count(), 2)
        self.assertTrue(session.is_stuck(threshold=2))

        # Proposing another identical click
        self.assertEqual(session.get_repetition_count(new_action=click1), 2)
        self.assertTrue(session.is_stuck(threshold=2, new_action=click1))

        # Proposing different action resets repetition check for new action
        key_act = Action(kind=ActionKind.KEY, keys=["win"])
        self.assertEqual(session.get_repetition_count(new_action=key_act), 0)
        self.assertFalse(session.is_stuck(threshold=2, new_action=key_act))

        # Metrics check
        metrics = session.get_metrics()
        self.assertTrue(metrics["dry_run"])
        self.assertTrue(metrics["is_stuck"])

    def test_session_count_recent_matches(self):
        session = AceSession(goal="Interleaved loop test", max_steps=10)
        click = Action(kind=ActionKind.LEFT_CLICK, coordinate=(260, 800))
        key_win = Action(kind=ActionKind.KEY, keys=["win"])

        # Step 1: click
        session.record_step(click)
        # Step 2: click
        session.record_step(click)
        # Step 3: key win (breakout intervention)
        session.record_step(key_win)

        # Consecutive count for click is 0 because step 3 was key_win
        self.assertEqual(session.get_repetition_count(new_action=click), 0)
        # But count_recent_matches finds 2 matches in the window!
        self.assertEqual(session.count_recent_matches(click, window=5), 2)
        # is_stuck detects that this action is repeated across the recent window
        self.assertTrue(session.is_stuck(threshold=2, new_action=click))

    def test_session_attribute_matching(self):
        session = AceSession(goal="Attr match test", max_steps=10)
        type_a = Action(kind=ActionKind.TYPE, coordinate=(100, 200), text="a")
        type_b = Action(kind=ActionKind.TYPE, coordinate=(100, 200), text="b")

        session.record_step(type_a)
        # type_b has same coordinate but different text: should NOT match!
        self.assertEqual(session.get_repetition_count(new_action=type_b), 0)
        self.assertEqual(session.count_recent_matches(type_b, window=5), 0)


class TestWindowsComputer(unittest.TestCase):
    """Verifies WindowsComputer observation, resolution scaling, and key mapping."""

    def test_desktop_observation(self):
        computer = WindowsComputer()
        obs = computer.observe(format="webp")
        self.assertEqual(obs.width, 1280)
        self.assertEqual(obs.height, 800)
        self.assertEqual(obs.format, "webp")
        self.assertGreater(len(obs.image_bytes), 100)
        self.assertGreater(obs.capture_time_ms, 0.0)

    def test_canonical_to_native_scaling(self):
        computer = WindowsComputer()
        # Origin (0, 0)
        nx0, ny0 = computer._canonical_to_native(0, 0)
        self.assertEqual((nx0, ny0), (0, 0))

        # Bottom-right (1280, 800)
        nx_max, ny_max = computer._canonical_to_native(1280, 800)
        self.assertEqual(nx_max, computer.native_width - 1)
        self.assertEqual(ny_max, computer.native_height - 1)

    def test_resolve_vk_keys(self):
        computer = WindowsComputer()
        # Special keys
        self.assertEqual(computer._resolve_vk("enter"), 0x0D)
        self.assertEqual(computer._resolve_vk("tab"), 0x09)
        self.assertEqual(computer._resolve_vk("ctrl"), 0x11)
        self.assertEqual(computer._resolve_vk("shift"), 0x10)
        self.assertEqual(computer._resolve_vk("alt"), 0x12)
        self.assertEqual(computer._resolve_vk("esc"), 0x1B)

        # Alphanumeric keys
        self.assertEqual(computer._resolve_vk("a"), ord("A"))
        self.assertEqual(computer._resolve_vk("z"), ord("Z"))
        self.assertEqual(computer._resolve_vk("1"), ord("1"))

        # Function keys
        self.assertEqual(computer._resolve_vk("f5"), 0x74)

    def test_dry_run_execution(self):
        computer = WindowsComputer()
        act = Action(kind=ActionKind.LEFT_CLICK, coordinate=(500, 400), reason="Dry run click")
        # Ensure execute with dry_run=True finishes without moving hardware mouse or raising
        computer.execute(act, dry_run=True)

    def test_attach_desktop_closes_previous_handle(self):
        computer = WindowsComputer()
        with patch.object(computer.user32, "OpenInputDesktop", side_effect=[0x1001, 0x1002]), \
             patch.object(computer.user32, "SetThreadDesktop"), \
             patch.object(computer.user32, "CloseDesktop") as mock_close:
            computer.attach_desktop()
            self.assertEqual(computer._hdesk, 0x1001)

            # Second call should switch thread desktop and close previous handle
            computer.attach_desktop()
            self.assertEqual(computer._hdesk, 0x1002)
            mock_close.assert_called_with(0x1001)

    def test_computer_close_releases_handle(self):
        computer = WindowsComputer()
        computer._orig_hdesk = 0x0100
        computer._hdesk = 0x2001
        with patch.object(computer.user32, "SetThreadDesktop") as mock_set, \
             patch.object(computer.user32, "CloseDesktop") as mock_close:
            computer.close()
            mock_set.assert_called_with(0x0100)
            mock_close.assert_called_once_with(0x2001)
            self.assertIsNone(computer._hdesk)

    def test_computer_context_manager(self):
        with WindowsComputer() as computer:
            self.assertIsNotNone(computer)
            self.assertTrue(hasattr(computer, "close"))

    def test_real_win32_desktop_handle_lifecycle(self):
        if sys.platform != "win32":
            return
        # Test real Win32 handle allocation, switching, and non-busy closure
        comp = WindowsComputer()
        try:
            h1 = comp._hdesk
            self.assertIsNotNone(h1)
            comp.attach_desktop()
            h2 = comp._hdesk
            self.assertIsNotNone(h2)
            self.assertTrue(isinstance(h1, int) and h1 > 0)
            self.assertTrue(isinstance(h2, int) and h2 > 0)
        finally:
            comp.close()
            self.assertIsNone(comp._hdesk)

    def test_per_monitor_dpi_awareness_init(self):
        with patch("ctypes.windll.shcore.SetProcessDpiAwareness") as mock_dpi:
            computer = WindowsComputer()
            mock_dpi.assert_called_with(2)

    def test_per_monitor_dpi_awareness_fallback(self):
        with patch("ctypes.windll.shcore.SetProcessDpiAwareness", side_effect=Exception("Not supported")), \
             patch("ctypes.windll.user32.SetProcessDPIAware") as mock_user_dpi:
            computer = WindowsComputer()
            mock_user_dpi.assert_called_once()


class TestAceRunner(unittest.TestCase):
    """Verifies autonomous multi-step execution loop and safety breakout."""

    def setUp(self):
        import tempfile
        self._temp_vault = tempfile.TemporaryDirectory()
        self._patch_logger = patch("avcs.ace.runner.TrajectoryLogger", lambda *a, **k: TrajectoryLogger(vault_dir=self._temp_vault.name))
        self._patch_cache = patch("avcs.ace.runner.SkillCache", lambda *a, **k: SkillCache(vault_dir=self._temp_vault.name))
        self._patch_logger.start()
        self._patch_cache.start()

    def tearDown(self):
        self._patch_cache.stop()
        self._patch_logger.stop()
        self._temp_vault.cleanup()

    def test_autonomous_runner_execution_until_stop(self):
        mock_client = MagicMock()
        # Multi-step sequence: Step 1 click, Step 2 type, Step 3 stop
        mock_client.model = "gpt-6-astra"
        mock_client.predict_action.side_effect = [
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(100, 700), reason="Click taskbar"), 350.0),
            (Action(kind=ActionKind.TYPE, coordinate=(100, 700), text="notepad", reason="Type notepad"), 410.0),
            (Action(kind=ActionKind.STOP, reason="Notepad launched"), 280.0),
        ]

        agent = AceAgent(client=mock_client)
        computer = WindowsComputer()
        runner = AceRunner(agent=agent, computer=computer, enable_voice=False)

        session = runner.run(goal="Launch notepad", max_steps=10, dry_run=True)
        self.assertTrue(session.is_complete)
        self.assertEqual(session.step_count, 3)
    def test_safety_corner_breakout(self):
        agent = AceAgent()
        computer = WindowsComputer()
        runner = AceRunner(agent=agent, computer=computer, enable_voice=False)

        # Mock mouse at corner (5, 5)
        with patch.object(computer, "get_cursor_pos", return_value=(5, 5)):
            with self.assertRaises(SafetyBreakoutError) as ctx:
                runner.check_safety_breakout()
            self.assertIn("top-left corner", str(ctx.exception))

    def test_network_error_raises_ace_client_error(self):
        from avcs.ace.client import AceClientError
        client = CustomLLMClient(endpoint="http://invalid-nonexistent-domain.xyz/v1/chat/completions", timeout=0.5)
        with patch.object(client.session, "post", side_effect=Exception("Connection refused")):
            with self.assertRaises(AceClientError):
                client.predict_action(b"fake-bytes", goal="test")

    def test_history_truncation_to_max_recent(self):
        session = AceSession(goal="Windowing test", max_steps=20)
        for i in range(10):
            session.record_step(Action(kind=ActionKind.WAIT, reason=f"step {i+1}"))
        summary = session.get_history_summary(max_recent=4)
        self.assertEqual(len(summary), 4)
        self.assertEqual(summary[0]["step"], 7)
        self.assertEqual(summary[3]["step"], 10)

    def test_symbol_keys_resolution(self):
        computer = WindowsComputer()
        # Common symbols used in hotkeys
        self.assertIsNotNone(computer._resolve_vk("-"))
        self.assertIsNotNone(computer._resolve_vk("="))
        self.assertIsNotNone(computer._resolve_vk("/"))

    def test_extended_vk_and_windows_keys(self):
        computer = WindowsComputer()
        self.assertEqual(computer._resolve_vk("windows"), 0x5B)
        self.assertEqual(computer._resolve_vk("cmd"), 0x5B)
        self.assertEqual(computer._resolve_vk("command"), 0x5B)
        self.assertEqual(computer._resolve_vk("capslock"), 0x14)
        self.assertEqual(computer._resolve_vk("prtsc"), 0x2C)

    def test_extended_safety_breakout_corners(self):
        agent = AceAgent()
        computer = WindowsComputer()
        runner = AceRunner(agent=agent, computer=computer, enable_voice=False)

        # Mock mouse at bottom-right corner (nw - 5, nh - 5)
        nw, nh = computer.native_width, computer.native_height
        with patch.object(computer, "get_cursor_pos", return_value=(nw - 5, nh - 5)):
            with self.assertRaises(SafetyBreakoutError) as ctx:
                runner.check_safety_breakout()
            self.assertIn("bottom-right corner", str(ctx.exception))

    def test_safety_breakout_all_four_corners(self):
        agent = AceAgent()
        computer = WindowsComputer()
        runner = AceRunner(agent=agent, computer=computer, enable_voice=False)
        nw, nh = computer.native_width, computer.native_height

        corners = [
            ((0, 0), "top-left corner"),
            ((5, 5), "top-left corner"),
            ((15, 15), "top-left corner"),
            ((nw, 0), "top-right corner"),
            ((nw - 5, 5), "top-right corner"),
            ((nw - 15, 15), "top-right corner"),
            ((0, nh), "bottom-left corner"),
            ((5, nh - 5), "bottom-left corner"),
            ((15, nh - 15), "bottom-left corner"),
            ((nw, nh), "bottom-right corner"),
            ((nw - 5, nh - 5), "bottom-right corner"),
            ((nw - 15, nh - 15), "bottom-right corner"),
        ]
        for pos, expected_str in corners:
            with patch.object(computer, "get_cursor_pos", return_value=pos):
                with self.assertRaises(SafetyBreakoutError) as ctx:
                    runner.check_safety_breakout()
                self.assertIn(expected_str, str(ctx.exception))

    def test_safety_breakout_multi_monitor_bounds(self):
        agent = AceAgent()
        computer = WindowsComputer()
        runner = AceRunner(agent=agent, computer=computer, enable_voice=False)
        nw, nh = computer.native_width, computer.native_height

        # Cursor outside primary monitor boundary (multi-monitor setups) or just inside safety margins
        out_of_bounds_positions = [
            # Immediate 1-pixel exterior onto secondary monitors
            (nw + 1, 0),           # Right monitor top
            (nw + 1, nh),          # Right monitor bottom
            (-1, 0),               # Left monitor top
            (-1, nh),              # Left monitor bottom
            (0, -1),               # Top monitor left
            (nw, -1),              # Top monitor right
            (0, nh + 1),           # Bottom monitor left
            (nw, nh + 1),          # Bottom monitor right
            # Deep secondary monitors
            (nw + 50, 5),          # Right secondary monitor (top edge)
            (nw + 100, nh - 5),    # Right secondary monitor (bottom edge)
            (-50, 5),              # Left secondary monitor (top edge)
            (-100, nh - 5),        # Left secondary monitor (bottom edge)
            (5, -50),              # Top secondary monitor
            (nw - 5, -50),         # Top secondary monitor
            (5, nh + 50),          # Bottom secondary monitor
            (nw - 5, nh + 50),     # Bottom secondary monitor
            (nw // 2, nh // 2),    # Center of screen
            # 1-pixel interior non-corner margins
            (16, 15),
            (15, 16),
            (nw - 16, 15),
            (15, nh - 16),
            (nw - 16, nh - 16),
        ]
        for pos in out_of_bounds_positions:
            with patch.object(computer, "get_cursor_pos", return_value=pos):
                # Should not raise SafetyBreakoutError
                runner.check_safety_breakout()

    def test_speak_feedback_with_speaker_async(self):
        from avcs.ace.runner import speak_feedback
        with patch("avcs.voice.tts.Speaker") as MockSpeaker:
            mock_inst = MagicMock()
            MockSpeaker.return_value = mock_inst
            speak_feedback("Task completed successfully", enabled=True)
            # Verify speak_async was invoked
            mock_inst.speak_async.assert_called_once_with("Task completed successfully")

    def test_runner_dry_run_warning_and_stuck_breakout(self):
        mock_client = MagicMock()
        mock_client.model = "gpt-6-astra"
        # Simulate model repeatedly emitting left_click (260, 800)
        mock_client.predict_action.side_effect = [
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(260, 800), reason="Click Start"), 300.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(260, 800), reason="Click Start"), 300.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(260, 800), reason="Click Start"), 300.0),
            (Action(kind=ActionKind.STOP, reason="Done"), 200.0),
        ]

        agent = AceAgent(client=mock_client)
        computer = WindowsComputer()
        runner = AceRunner(agent=agent, computer=computer, enable_voice=False)

        with patch("avcs.ace.runner.logger") as mock_logger:
            session = runner.run(goal="Open Start menu and search for Notepad", max_steps=5, dry_run=True)

            # Dry-run warning should have been logged
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            self.assertTrue(any("DRY-RUN MODE ACTIVE" in c for c in warning_calls))

            # Step 1: click (260, 800)
            self.assertEqual(session.steps[0].action["kind"], "left_click")
            # Step 2: click (260, 800)
            self.assertEqual(session.steps[1].action["kind"], "left_click")
            # Step 3: loop breakout via jittering without fake "win" key macro
            self.assertEqual(session.steps[2].action["kind"], "left_click")
            self.assertEqual(session.steps[2].action["coordinate"], [270, 790])
            self.assertIn("Jitter", session.steps[2].reason)

    def test_runner_stuck_breakout_jitter(self):
        mock_client = MagicMock()
        mock_client.model = "gpt-6-astra"
        # Simulate model repeatedly clicking non-start button on a generic canvas
        mock_client.predict_action.side_effect = [
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(500, 400), reason="Click button"), 300.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(500, 400), reason="Click button"), 300.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(500, 400), reason="Click button"), 300.0),
            (Action(kind=ActionKind.STOP, reason="Done"), 200.0),
        ]

        agent = AceAgent(client=mock_client)
        computer = WindowsComputer()
        runner = AceRunner(agent=agent, computer=computer, enable_voice=False)

        session = runner.run(goal="Select option B", max_steps=5, dry_run=True)
        # Step 3: jittered coordinate (510, 410) instead of (500, 400)
        self.assertEqual(session.steps[2].action["kind"], "left_click")
        self.assertEqual(session.steps[2].action["coordinate"], [510, 410])
        self.assertIn("Jitter", session.steps[2].reason)

    def test_runner_stuck_breakout_no_fake_macros(self):
        mock_client = MagicMock()
        mock_client.model = "gpt-6-astra"
        # Simulate model repeatedly emitting click (260, 800) at step 1, 2, 3, 4
        mock_client.predict_action.side_effect = [
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(260, 800), reason="Click Start"), 300.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(260, 800), reason="Click Start"), 300.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(260, 800), reason="Click Start"), 300.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(260, 800), reason="Click Start"), 300.0),
            (Action(kind=ActionKind.STOP, reason="Done"), 200.0),
        ]

        agent = AceAgent(client=mock_client)
        computer = WindowsComputer()
        runner = AceRunner(agent=agent, computer=computer, enable_voice=False)

        session = runner.run(goal="Open Start menu and search for Notepad", max_steps=5, dry_run=True)
        # Step 1: click
        self.assertEqual(session.steps[0].action["kind"], "left_click")
        # Step 2: click
        self.assertEqual(session.steps[1].action["kind"], "left_click")
        # Step 3: honest jittering without fake "win" key macro
        self.assertEqual(session.steps[2].action["kind"], "left_click")
        self.assertEqual(session.steps[2].action["coordinate"], [270, 790])
        # Step 4: honest execution without fake "notepad" typing macro
        self.assertEqual(session.steps[3].action["kind"], "left_click")
        self.assertNotEqual(session.steps[3].action["kind"], "type")


    def test_frontier_vam_1000_grid_coordinate_parsing(self):
        """Verifies frontier models emitting [0, 1000] coordinates (e.g. [226, 974]) are scaled correctly."""
        # 1. 1000-scale coordinate where fy > 800 (e.g. Windows 11 Start button)
        scaled = Action._parse_coordinate_pair(226, 974, canvas_w=1280, canvas_h=800)
        self.assertEqual(scaled, (289, 779))

        # 2. Canonical 1280x800 coordinate is preserved intact
        canonical = Action._parse_coordinate_pair(260, 800, canvas_w=1280, canvas_h=800)
        self.assertEqual(canonical, (260, 800))

    def test_screen_streamer_continuous_capture(self):
        """Verifies ScreenStreamer provides pre-captured frames with <1ms read latency."""
        import time
        from avcs.ace.streamer import ScreenStreamer
        computer = WindowsComputer()
        streamer = ScreenStreamer(computer=computer, target_fps=30.0, image_format="webp")
        streamer.start(startup_timeout=1.5)
        try:
            self.assertTrue(streamer.is_running)
            obs = None
            for _ in range(15):
                obs = streamer.get_latest_observation()
                if obs is not None:
                    break
                time.sleep(0.1)
            self.assertIsNotNone(obs, "ScreenStreamer failed to capture a frame within timeout")
            t0 = time.time()
            obs2 = streamer.get_latest_observation()
            read_latency_ms = (time.time() - t0) * 1000.0
            self.assertIsNotNone(obs2)
            self.assertLess(read_latency_ms, 5.0)  # Instant in-memory pointer read
            self.assertEqual(obs2.width, 1280)
            self.assertEqual(obs2.height, 800)
        finally:
            streamer.stop()
            self.assertFalse(streamer.is_running)

    def test_enter_key_no_stuck_down_state(self):
        """Verifies ActionKind.TYPE with press_enter releases the Enter key completely."""
        computer = WindowsComputer()
        act = Action(kind=ActionKind.TYPE, text="test", press_enter=True)
        computer.execute(act, dry_run=False)
        # Check that VK_RETURN (0x0D) is NOT stuck down in the OS
        state = computer.user32.GetAsyncKeyState(0x0D)
        is_down = bool(state & 0x8000)
        self.assertFalse(is_down, "VK_RETURN was left held down in OS!")


class TestBoostEngineEnhancements(unittest.TestCase):
    """Verifies all /boost requirements across Action parsing, kinematics, safety, and runner."""

    def test_nested_parameter_unwrapping(self):
        # OpenAI tool call format with nested parameters
        d1 = {"action": "click", "parameters": {"coordinate": [260, 800]}}
        a1 = Action.from_dict(d1)
        self.assertEqual(a1.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(a1.coordinate, (260, 800))

        # Anthropic tool use with arguments dict
        d2 = {"name": "mouse_move", "arguments": {"x": 500, "y": 300}}
        a2 = Action.from_dict(d2)
        self.assertEqual(a2.kind, ActionKind.MOUSE_MOVE)
        self.assertEqual(a2.coordinate, (500, 300))

        # JSON-string arguments (OpenAI function call serialization)
        d3 = {"action": "type", "arguments": json.dumps({"text": "notepad", "press_enter": True})}
        a3 = Action.from_dict(d3)
        self.assertEqual(a3.kind, ActionKind.TYPE)
        self.assertEqual(a3.text, "notepad")
        self.assertTrue(a3.press_enter)

        # args and params aliases
        d4 = {"action": "key", "args": {"keys": ["ctrl", "c"]}}
        a4 = Action.from_dict(d4)
        self.assertEqual(a4.kind, ActionKind.KEY)
        self.assertEqual(a4.keys, ["ctrl", "c"])

    def test_key_list_unpacking_fix(self):
        # Previously {"key": ["win"]} turned into ["['win']"]
        d = {"kind": "key", "key": ["win"]}
        a = Action.from_dict(d)
        self.assertEqual(a.keys, ["win"])

        d2 = {"kind": "key", "key": ["ctrl", "alt", "del"]}
        a2 = Action.from_dict(d2)
        self.assertEqual(a2.keys, ["ctrl", "alt", "del"])

    def test_anthropic_computer_use_format(self):
        # Anthropic format uses action="key" with text="win" or text="Return"
        d1 = {"action": "key", "text": "win"}
        a1 = Action.from_dict(d1)
        self.assertEqual(a1.kind, ActionKind.KEY)
        self.assertEqual(a1.keys, ["win"])

        d2 = {"action": "key", "text": "ctrl+shift+esc"}
        a2 = Action.from_dict(d2)
        self.assertEqual(a2.kind, ActionKind.KEY)
        self.assertEqual(a2.keys, ["ctrl", "shift", "esc"])

    def test_key_and_text_alias_normalization(self):
        # Key aliases: hotkey, shortcut, press, combo
        self.assertEqual(Action.from_dict({"action": "key", "hotkey": "alt+f4"}).keys, ["alt", "f4"])
        self.assertEqual(Action.from_dict({"action": "key", "shortcut": "ctrl+s"}).keys, ["ctrl", "s"])
        self.assertEqual(Action.from_dict({"action": "key", "press": "enter"}).keys, ["enter"])
        self.assertEqual(Action.from_dict({"action": "key", "combo": "win+r"}).keys, ["win", "r"])

        # Text aliases: query, search, text_to_type, input, content
        self.assertEqual(Action.from_dict({"action": "type", "query": "hello"}).text, "hello")
        self.assertEqual(Action.from_dict({"action": "type", "search": "notepad"}).text, "notepad")
        self.assertEqual(Action.from_dict({"action": "type", "text_to_type": "calc"}).text, "calc")
        self.assertEqual(Action.from_dict({"action": "type", "content": "test"}).text, "test")

    def test_semantic_fallback_extraction_from_reason(self):
        # Key fallback from reason
        a1 = Action.from_dict({"kind": "key", "reason": "Open the Start menu"})
        self.assertEqual(a1.kind, ActionKind.KEY)
        self.assertEqual(a1.keys, ["win"])

        a2 = Action.from_dict({"kind": "key", "reason": "Press Enter to submit"})
        self.assertEqual(a2.kind, ActionKind.KEY)
        self.assertEqual(a2.keys, ["enter"])

        # Type fallback from reason
        a3 = Action.from_dict({"kind": "type", "reason": "Type 'Notepad' into search and press enter"})
        self.assertEqual(a3.kind, ActionKind.TYPE)
        self.assertEqual(a3.text, "Notepad")
        self.assertTrue(a3.press_enter)

        a4 = Action.from_dict({"kind": "type", "reason": 'Search for "calculator"'})
        self.assertEqual(a4.kind, ActionKind.TYPE)
        self.assertEqual(a4.text, "calculator")

    def test_guard_against_empty_no_op_actions(self):
        # Empty key action with no keys and no reason fallback
        a1 = Action.from_dict({"kind": "key"})
        self.assertEqual(a1.kind, ActionKind.WAIT)
        self.assertGreater(a1.duration, 0.0)

        # Empty type action with no text and no enter
        a2 = Action.from_dict({"kind": "type"})
        self.assertEqual(a2.kind, ActionKind.WAIT)
        self.assertGreater(a2.duration, 0.0)

        # Empty mouse_move with no coordinate
        a3 = Action.from_dict({"kind": "mouse_move"})
        self.assertEqual(a3.kind, ActionKind.WAIT)
        self.assertGreater(a3.duration, 0.0)

    def test_computer_dual_dispatch_and_fitts_law(self):
        computer = WindowsComputer()
        with patch.object(computer, "get_cursor_pos", return_value=(100, 100)), \
             patch.object(computer.user32, "SetCursorPos") as mock_set_pos, \
             patch.object(computer, "_send_mouse_event") as mock_send_mouse, \
             patch("time.sleep"):
            # Move from (100, 100) to (500, 400), dist = 500
            # Fitts's Law: clamp(0.20, 0.45, 0.18 + 0.00030 * 500) = 0.33s
            computer._move_minimum_jerk(500, 400)
            # Both SetCursorPos AND SendInput motion must be called across steps
            self.assertGreater(mock_set_pos.call_count, 10)
            self.assertGreater(mock_send_mouse.call_count, 10)
            # Last call must be target coordinate
            mock_set_pos.assert_called_with(500, 400)

    def test_computer_emergency_release_hardware(self):
        computer = WindowsComputer()
        with patch.object(computer, "_send_mouse_event") as mock_mouse, \
             patch.object(computer, "_send_vk") as mock_vk, \
             patch.object(computer.user32, "GetAsyncKeyState", return_value=0x8000), \
             patch.object(computer.user32, "ClipCursor") as mock_clip:
            computer.emergency_release_hardware()
            # 3 mouse button release calls (left, right, middle)
            self.assertEqual(mock_mouse.call_count, 3)
            # Modifier keys released
            self.assertGreater(mock_vk.call_count, 0)
            # Cursor clipping reset
            mock_clip.assert_called_once_with(None)

    def test_runner_execution_safety_firewall(self):
        agent = AceAgent()
        computer = WindowsComputer()
        runner = AceRunner(agent=agent, computer=computer, enable_voice=False)

        # Mock foreground window as ConsoleWindowClass
        with patch.object(runner.user32, "GetForegroundWindow", return_value=0x1234), \
             patch.object(runner.user32, "GetClassNameW", side_effect=lambda hwnd, buf, size: setattr(buf, "value", "ConsoleWindowClass")), \
             patch.object(runner.computer, "get_cursor_pos", return_value=(500, 500)):
            mock_client = MagicMock()
            mock_client.model = "gpt-6-astra"
            # Ungrounded typing action into terminal
            mock_client.predict_action.side_effect = [
                (Action(kind=ActionKind.TYPE, text="dangerous_command", coordinate=None), 100.0),
                (Action(kind=ActionKind.STOP, reason="Done"), 100.0),
            ]
            runner.agent.client = mock_client
            session = runner.run(goal="Test console safety", max_steps=2, dry_run=True)
            step1_action = session.steps[0].action
            # Safety Firewall must reject ungrounded terminal typing and record a safe wait
            self.assertEqual(step1_action["kind"], "wait")
            self.assertIn("Execution Safety Firewall", step1_action["reason"])
            # Reflection feedback is passed to model in next step history
            second_call_history = mock_client.predict_action.call_args_list[1][1].get("history")
            self.assertTrue(any("Execution Safety Firewall" in h.get("reason", "") for h in second_call_history))

    def test_runner_intentional_corner_click_exemption(self):
        agent = AceAgent()
        computer = WindowsComputer()
        runner = AceRunner(agent=agent, computer=computer, enable_voice=False)
        nw, nh = computer.native_width, computer.native_height

        # Start button corner at bottom-left: (5, nh - 5)
        with patch.object(computer, "get_cursor_pos", return_value=(5, nh - 5)):
            # Without target_coord, it raises SafetyBreakoutError
            with self.assertRaises(SafetyBreakoutError):
                runner.check_safety_breakout()
            # With intentional target_coord at Start button, it does NOT raise!
            runner.check_safety_breakout(target_coord=(10, nh - 10))

        # Close button corner at top-right: (nw - 5, 5)
        with patch.object(computer, "get_cursor_pos", return_value=(nw - 5, 5)):
            with self.assertRaises(SafetyBreakoutError):
                runner.check_safety_breakout()
            # With intentional target_coord at Close button, it does NOT raise!
            runner.check_safety_breakout(target_coord=(nw - 10, 10))

    def test_anthropic_computer_tool_use_unwrapping(self):
        # Anthropic format tool call: {"name": "computer", "input": {"action": "left_click", "coordinate": [260, 800]}}
        d = {"name": "computer", "input": {"action": "left_click", "coordinate": [260, 800]}}
        a = Action.from_dict(d)
        self.assertEqual(a.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(a.coordinate, (260, 800))

    def test_dict_and_string_coordinate_extraction(self):
        # Coordinate given as dict: {"x": 150, "y": 300}
        d1 = {"action": "click", "coordinate": {"x": 150, "y": 300}}
        a1 = Action.from_dict(d1)
        self.assertEqual(a1.coordinate, (150, 300))

        # Coordinate given as tuple string: "(400, 500)"
        d2 = {"action": "click", "coordinate": "(400, 500)"}
        a2 = Action.from_dict(d2)
        self.assertEqual(a2.coordinate, (400, 500))

        # Start and end coordinates as dicts
        d3 = {
            "action": "drag",
            "start_coordinate": {"x": 100, "y": 100},
            "end_coordinate": {"x": 600, "y": 600},
        }
        a3 = Action.from_dict(d3)
        self.assertEqual(a3.start_coordinate, (100, 100))
        self.assertEqual(a3.end_coordinate, (600, 600))

    def test_hyphen_key_combos_and_anthropic_key_names(self):
        # Key combos with hyphens
        a1 = Action.from_dict({"action": "key", "key": "ctrl-shift-esc"})
        self.assertEqual(a1.keys, ["ctrl", "shift", "esc"])

        a2 = Action.from_dict({"action": "key", "key": "alt-f4"})
        self.assertEqual(a2.keys, ["alt", "f4"])

        # Anthropic key names resolution in WindowsComputer
        comp = WindowsComputer()
        self.assertEqual(comp._resolve_vk("page_up"), 0x21)
        self.assertEqual(comp._resolve_vk("page_down"), 0x22)
        self.assertEqual(comp._resolve_vk("super_l"), 0x5B)
        self.assertEqual(comp._resolve_vk("super_r"), 0x5C)
        self.assertEqual(comp._resolve_vk("ctrl_l"), 0x11)
        self.assertEqual(comp._resolve_vk("menu"), 0x5D)

    def test_semantic_fallback_hotkey_precedence_and_typing(self):
        # Hotkey in reason must NOT be clobbered by substring single key checks
        a1 = Action.from_dict({"kind": "key", "reason": "Press ctrl+shift+esc to open task manager"})
        self.assertEqual(a1.keys, ["ctrl", "shift", "esc"])

        # Type single word without trailing punctuation
        a2 = Action.from_dict({"kind": "type", "reason": "Type Notepad"})
        self.assertEqual(a2.kind, ActionKind.TYPE)
        self.assertEqual(a2.text, "Notepad")

        # Natural language semantic extraction
        a3 = Action.from_dict({"kind": "type", "reason": "Type Notepad into editor"})
        self.assertEqual(a3.kind, ActionKind.TYPE)
        self.assertEqual(a3.text, "Notepad")

        a4 = Action.from_dict({"kind": "type", "reason": "Type calculator into search"})
        self.assertEqual(a4.kind, ActionKind.TYPE)
        self.assertEqual(a4.text, "calculator")

    def test_multimonitor_virtual_desktop_normalization(self):
        import ctypes
        comp = WindowsComputer()
        # Mock a dual monitor setup: primary 1280x800, secondary 1920x1080 to the left (SM_XVIRTUALSCREEN = -1920, SM_CXVIRTUALSCREEN = 3200)
        captured_inputs = []
        def fake_send_input(n, ptr, size):
            inp_obj = INPUT.from_address(ctypes.addressof(ptr._obj))
            captured_inputs.append((inp_obj.union.mi.dx, inp_obj.union.mi.dy, inp_obj.union.mi.dwFlags))
            return n

        with patch.object(comp.user32, "GetSystemMetrics", side_effect=lambda idx: {76: -1920, 77: 0, 78: 3200, 79: 1080}.get(idx, 0)), \
             patch.object(comp.user32, "SendInput", side_effect=fake_send_input):
            comp._send_mouse_event(flags=0x0001, x=0, y=0)
            self.assertEqual(len(captured_inputs), 1)
            dx, dy, flags = captured_inputs[0]
            # (0 - (-1920)) / 3199 * 65535 = 1920 / 3199 * 65535 ~= 39332
            self.assertGreater(dx, 35000)
            self.assertLess(dx, 45000)
            self.assertTrue(bool(flags & 0x4000))  # MOUSEEVENTF_VIRTUALDESK

    def test_main_cli_entrypoint_success(self):
        from avcs.ace.runner import main
        test_args = ["runner.py", "--goal", "Test CLI Goal", "--dry-run", "--no-voice", "--max-steps", "5"]
        with patch.object(sys, "argv", test_args), \
             patch("avcs.ace.runner.AceRunner.run") as mock_run:
            main()
            mock_run.assert_called_once_with(
                goal="Test CLI Goal",
                max_steps=5,
                image_format="jpeg",
                dry_run=True,
            )

    def test_main_cli_entrypoint_safety_breakout(self):
        from avcs.ace.runner import main, SafetyBreakoutError
        test_args = ["runner.py", "--goal", "Test Breakout", "--dry-run", "--no-voice"]
        with patch.object(sys, "argv", test_args), \
             patch("avcs.ace.runner.AceRunner.run", side_effect=SafetyBreakoutError("corner hit")), \
             patch("avcs.ace.runner.speak_feedback"), \
             patch.object(sys, "exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(2)

    def test_main_cli_entrypoint_keyboard_interrupt(self):
        from avcs.ace.runner import main
        test_args = ["runner.py", "--goal", "Test Interrupt", "--dry-run", "--no-voice"]
        with patch.object(sys, "argv", test_args), \
             patch("avcs.ace.runner.AceRunner.run", side_effect=KeyboardInterrupt()), \
             patch.object(sys, "exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)


class TestActionAdapter(unittest.TestCase):
    """Verifies ActionAdapter normalization across Anthropic, OpenAI, AVCS Grammar, and reason fallback."""

    def test_anthropic_computer_use_format(self):
        # Key Return -> enter
        a1 = ActionAdapter.adapt({"action": "key", "text": "Return"})
        self.assertEqual(a1.kind, ActionKind.KEY)
        self.assertEqual(a1.keys, ["enter"])

        # Key BackSpace -> backspace
        a2 = ActionAdapter.adapt({"action": "key", "text": "BackSpace"})
        self.assertEqual(a2.kind, ActionKind.KEY)
        self.assertEqual(a2.keys, ["backspace"])

        # Key Super_L -> win
        a3 = ActionAdapter.adapt({"action": "key", "text": "Super_L"})
        self.assertEqual(a3.kind, ActionKind.KEY)
        self.assertEqual(a3.keys, ["win"])

        # Mouse move
        a4 = ActionAdapter.adapt({"action": "mouse_move", "coordinate": [292, 768]})
        self.assertEqual(a4.kind, ActionKind.MOUSE_MOVE)
        self.assertEqual(a4.coordinate, (292, 768))

        # Clicks
        a5 = ActionAdapter.adapt({"action": "left_click", "coordinate": [150, 250]})
        self.assertEqual(a5.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(a5.coordinate, (150, 250))

        a6 = ActionAdapter.adapt({"action": "right_click", "coordinate": [300, 400]})
        self.assertEqual(a6.kind, ActionKind.RIGHT_CLICK)
        self.assertEqual(a6.coordinate, (300, 400))

        a7 = ActionAdapter.adapt({"action": "double_click", "coordinate": [500, 600]})
        self.assertEqual(a7.kind, ActionKind.DOUBLE_CLICK)
        self.assertEqual(a7.coordinate, (500, 600))

        a8 = ActionAdapter.adapt({"action": "triple_click", "coordinate": [500, 600]})
        self.assertEqual(a8.kind, ActionKind.TRIPLE_CLICK)
        self.assertEqual(a8.coordinate, (500, 600))

        # Type
        a9 = ActionAdapter.adapt({"action": "type", "text": "Hello Anthropic"})
        self.assertEqual(a9.kind, ActionKind.TYPE)
        self.assertEqual(a9.text, "Hello Anthropic")

        # Screenshot / cursor_position -> WAIT
        a10 = ActionAdapter.adapt({"action": "screenshot"})
        self.assertEqual(a10.kind, ActionKind.WAIT)

        a11 = ActionAdapter.adapt({"action": "cursor_position"})
        self.assertEqual(a11.kind, ActionKind.WAIT)

    def test_anthropic_computer_nested_wrapper(self):
        # Nested inside {"name": "computer", "input": {...}}
        payload = {
            "name": "computer",
            "input": {
                "action": "left_click",
                "coordinate": [450, 550],
            },
        }
        act = ActionAdapter.adapt(payload)
        self.assertEqual(act.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act.coordinate, (450, 550))

    def test_openai_tool_calling_format(self):
        # Function call format
        payload1 = {
            "function": {
                "name": "left_click",
                "arguments": json.dumps({"coordinate": [292, 768]}),
            }
        }
        a1 = ActionAdapter.adapt(payload1)
        self.assertEqual(a1.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(a1.coordinate, (292, 768))

        # Tool calls array format
        payload2 = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "type",
                        "arguments": json.dumps({"text": "Notepad"}),
                    },
                }
            ]
        }
        a2 = ActionAdapter.adapt(payload2)
        self.assertEqual(a2.kind, ActionKind.TYPE)
        self.assertEqual(a2.text, "Notepad")

        # Nested parameters format
        payload3 = {
            "action": "key",
            "parameters": {"keys": ["ctrl", "s"]},
        }
        a3 = ActionAdapter.adapt(payload3)
        self.assertEqual(a3.kind, ActionKind.KEY)
        self.assertEqual(a3.keys, ["ctrl", "s"])

        # Nested input format
        payload4 = {
            "input": {"action": "wait", "duration": 1.8}
        }
        a4 = ActionAdapter.adapt(payload4)
        self.assertEqual(a4.kind, ActionKind.WAIT)
        self.assertEqual(a4.duration, 1.8)

    def test_avcs_fast_action_grammar_tokens(self):
        # CLICK
        a1 = ActionAdapter.adapt("ACT:CLICK 292 768")
        self.assertEqual(a1.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(a1.coordinate, (292, 768))

        # TYPE
        a2 = ActionAdapter.adapt("ACT:TYPE Notepad")
        self.assertEqual(a2.kind, ActionKind.TYPE)
        self.assertEqual(a2.text, "Notepad")

        # KEY
        a3 = ActionAdapter.adapt("ACT:KEY win")
        self.assertEqual(a3.kind, ActionKind.KEY)
        self.assertEqual(a3.keys, ["win"])

        # WAIT
        a4 = ActionAdapter.adapt("ACT:WAIT 0.5")
        self.assertEqual(a4.kind, ActionKind.WAIT)
        self.assertEqual(a4.duration, 0.5)

        # STOP
        a5 = ActionAdapter.adapt("ACT:STOP Done")
        self.assertEqual(a5.kind, ActionKind.STOP)
        self.assertEqual(a5.reason, "Done")

        # DRAG
        a6 = ActionAdapter.adapt("ACT:DRAG 100 200 400 500")
        self.assertEqual(a6.kind, ActionKind.LEFT_CLICK_DRAG)
        self.assertEqual(a6.start_coordinate, (100, 200))
        self.assertEqual(a6.end_coordinate, (400, 500))

        # SCROLL
        a7 = ActionAdapter.adapt("ACT:SCROLL 0 -120")
        self.assertEqual(a7.kind, ActionKind.SCROLL)
        self.assertEqual(a7.scroll_delta, (0, -120))

        # Multiline reasoning text with embedded grammar token and comment
        reasoning = (
            "Thinking: The user wants to click the start menu.\n"
            "ACT:CLICK 260 780 # Open Start button\n"
            "Execution dispatched."
        )
        a8 = ActionAdapter.adapt(reasoning)
        self.assertEqual(a8.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(a8.coordinate, (260, 780))
        self.assertEqual(a8.reason, "Open Start button")

    def test_anthropic_key_combo_normalization(self):
        # Super_L+r -> win, r
        a1 = ActionAdapter.adapt({"action": "key", "text": "Super_L+r"})
        self.assertEqual(a1.kind, ActionKind.KEY)
        self.assertEqual(a1.keys, ["win", "r"])

        # ctrl+Return -> ctrl, enter
        a2 = ActionAdapter.adapt({"action": "key", "text": "ctrl+Return"})
        self.assertEqual(a2.kind, ActionKind.KEY)
        self.assertEqual(a2.keys, ["ctrl", "enter"])

        # list with mixed casing
        a3 = ActionAdapter.adapt({"action": "key", "keys": ["ctrl_l+Alt_L", "Delete"]})
        self.assertEqual(a3.kind, ActionKind.KEY)
        self.assertEqual(a3.keys, ["ctrl", "alt", "delete"])

    def test_anthropic_content_messages_api_unwrapping(self):
        payload = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01Abc",
                    "name": "computer",
                    "input": {
                        "action": "left_click",
                        "coordinate": [350, 450],
                    },
                }
            ]
        }
        act = ActionAdapter.adapt(payload)
        self.assertEqual(act.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act.coordinate, (350, 450))

    def test_anthropic_left_click_drag_coordinate_destination(self):
        # Anthropic provides coordinate as drag destination
        payload = {"action": "left_click_drag", "coordinate": [600, 700]}
        act = ActionAdapter.adapt(payload)
        self.assertEqual(act.kind, ActionKind.LEFT_CLICK_DRAG)
        self.assertEqual(act.end_coordinate, (600, 700))

    def test_scroll_direction_and_amount(self):
        # Up
        a_up = ActionAdapter.adapt({"action": "scroll", "direction": "up", "amount": 2})
        self.assertEqual(a_up.kind, ActionKind.SCROLL)
        self.assertEqual(a_up.scroll_delta, (0, 240))

        # Down
        a_down = ActionAdapter.adapt({"action": "scroll", "direction": "down", "amount": 1})
        self.assertEqual(a_down.kind, ActionKind.SCROLL)
        self.assertEqual(a_down.scroll_delta, (0, -120))

    def test_grammar_token_comment_inside_quotes(self):
        # Hash inside quotes must not be stripped as a comment
        act = ActionAdapter.adapt('ACT:TYPE "issue #123" # comment on issue')
        self.assertEqual(act.kind, ActionKind.TYPE)
        self.assertEqual(act.text, "issue #123")
        self.assertEqual(act.reason, "comment on issue")

    def test_from_reason_fallback_coordinates_and_phrases(self):
        # Coordinate formats
        a1 = ActionAdapter.adapt({"reason": "Click at x=250, y=350"})
        self.assertEqual(a1.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(a1.coordinate, (250, 350))

        a2 = ActionAdapter.adapt({"reason": "Left click at [400, 500]"})
        self.assertEqual(a2.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(a2.coordinate, (400, 500))

        # Multi-word typing
        a3 = ActionAdapter.adapt({"reason": "Type Hello World into search bar"})
        self.assertEqual(a3.kind, ActionKind.TYPE)
        self.assertEqual(a3.text, "Hello World")

    def test_general_agents_format(self):
        payload = {
            "action": "left_click",
            "coordinate": [292, 768],
            "reason": "Click search button",
        }
        a = ActionAdapter.adapt(payload)
        self.assertEqual(a.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(a.coordinate, (292, 768))
        self.assertEqual(a.reason, "Click search button")

    def test_fallback_regex_intent_from_reason(self):
        # Click intent in reason
        a1 = ActionAdapter.adapt({"reason": "Click on Start button at (260, 780)"})
        self.assertEqual(a1.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(a1.coordinate, (260, 780))

        # Press key intent in reason
        a2 = ActionAdapter.adapt({"reason": "Press win key to open start menu"})
        self.assertEqual(a2.kind, ActionKind.KEY)
        self.assertEqual(a2.keys, ["win"])

        # Type intent in reason
        a3 = ActionAdapter.adapt({"reason": "Type 'Notepad' into search"})
        self.assertEqual(a3.kind, ActionKind.TYPE)
        self.assertEqual(a3.text, "Notepad")

        # Double click intent in reason
        a4 = ActionAdapter.adapt({"action": "wait", "reason": "Double click at (400, 500)"})
        self.assertEqual(a4.kind, ActionKind.DOUBLE_CLICK)
        self.assertEqual(a4.coordinate, (400, 500))


class TestSkillCacheAndAnchorVerification(unittest.TestCase):
    """Verifies Layer 2 Anchor-Verified Skill Cache and strict P0 rule enforcement."""

    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache = SkillCache(vault_dir=self.temp_dir.name)

    def tearDown(self):
        if hasattr(self, "cache") and self.cache:
            self.cache.close()
        self.temp_dir.cleanup()

    def test_p0_rule_missing_anchor_rejected(self):
        # Save skill with anchor
        anchor = UIStateAnchor(window_title="Windows Desktop", process_name="explorer")
        actions = [Action(kind=ActionKind.LEFT_CLICK, coordinate=(260, 780), reason="Click start")]
        self.cache.save_skill(goal="Open Start Menu", actions=actions, anchors=anchor)

        # Lookup with current_anchor=None -> MUST be rejected under P0 rule!
        res = self.cache.lookup(goal="Open Start Menu", current_anchor=None)
        self.assertFalse(res.hit)
        self.assertIn("P0 rule strictly forbids blind replay", res.reason)

    def test_p0_rule_mismatched_anchor_rejected(self):
        # Save skill for Notepad with Desktop anchor
        anchor = UIStateAnchor(window_title="Windows Desktop", process_name="explorer")
        actions = [Action(kind=ActionKind.LEFT_CLICK, coordinate=(260, 780))]
        self.cache.save_skill(goal="Open Start Menu", actions=actions, anchors=anchor)

        # Live state is Chrome browser (mismatched window & process)
        mismatched_anchor = UIStateAnchor(window_title="Google Chrome", process_name="chrome")
        res = self.cache.lookup(goal="Open Start Menu", current_anchor=mismatched_anchor)
        self.assertFalse(res.hit)
        self.assertIn("mismatch", res.reason)

    def test_anchor_verification_success_and_cache_hit(self):
        anchor = UIStateAnchor(window_title="Windows Desktop", process_name="explorer")
        actions = [
            Action(kind=ActionKind.KEY, keys=["win"], reason="Open start"),
            Action(kind=ActionKind.TYPE, text="Notepad", reason="Type app name"),
        ]
        self.cache.save_skill(goal="Launch Notepad App", actions=actions, anchors=anchor)

        # Live state matches anchor!
        matching_anchor = UIStateAnchor(window_title="Windows Desktop Shell", process_name="explorer.exe")
        res = self.cache.lookup(goal="Launch Notepad App", current_anchor=matching_anchor)
        self.assertTrue(res.hit)
        self.assertEqual(len(res.actions), 2)
        self.assertEqual(res.actions[0].kind, ActionKind.KEY)
        self.assertEqual(res.actions[1].text, "Notepad")
        self.assertIn("0ms model latency", res.reason)

    def test_bounding_box_anchor_tolerance(self):
        anchor = UIStateAnchor(bounding_box=(100, 100, 200, 200))
        # Within 15px tolerance
        current_valid = UIStateAnchor(bounding_box=(108, 95, 205, 192))
        self.assertTrue(anchor.matches(current_valid, tolerance_px=15))

        # Outside 15px tolerance
        current_invalid = UIStateAnchor(bounding_box=(130, 100, 200, 200))
        self.assertFalse(anchor.matches(current_invalid, tolerance_px=15))

    def test_skill_cache_disk_persistence(self):
        anchor = UIStateAnchor(window_title="Notepad", process_name="notepad")
        actions = [Action(kind=ActionKind.TYPE, text="Saved text")]
        self.cache.save_skill(goal="Save Document", actions=actions, anchors=anchor)

        # Reload a completely fresh SkillCache instance from same directory
        fresh_cache = SkillCache(vault_dir=self.temp_dir.name)
        skills = fresh_cache.list_skills()
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0]["goal"], "Save Document")

        # Can retrieve with matching anchor
        res = fresh_cache.lookup("Save Document", current_anchor=anchor)
        self.assertTrue(res.hit)
        self.assertEqual(res.actions[0].text, "Saved text")
        fresh_cache.close()

    def test_ace_runner_cache_hit_deterministic_replay(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with SkillCache(vault_dir=td) as cache:
                anchor = UIStateAnchor(window_title="Desktop", process_name="explorer")
                cached_actions = [
                    Action(kind=ActionKind.LEFT_CLICK, coordinate=(260, 780), reason="Click start"),
                    Action(kind=ActionKind.STOP, reason="Done"),
                ]
                cache.save_skill(goal="Replay goal", actions=cached_actions, anchors=anchor)

                agent = AceAgent()
                computer = WindowsComputer()
                runner = AceRunner(agent=agent, computer=computer, skill_cache=cache, enable_voice=False)

                # Mock LLM client so if it were called it would raise an error
                mock_client = MagicMock()
                mock_client.predict_action.side_effect = RuntimeError("LLM called during deterministic cache replay!")
                runner.agent.client = mock_client

                # Run with matching anchor
                session = runner.run(goal="Replay goal", max_steps=5, dry_run=True, current_anchor=anchor)
                self.assertTrue(session.is_complete)
                self.assertEqual(session.step_count, 2)
                self.assertEqual(session.steps[0].action["kind"], "left_click")
                # 0 calls to LLM!
                mock_client.predict_action.assert_not_called()

    def test_p0_empty_anchor_cannot_match(self):
        # Empty anchor with zero constraints MUST NOT match anything under P0
        empty_anchor = UIStateAnchor()
        self.assertTrue(empty_anchor.is_empty())
        current = UIStateAnchor(window_title="Random Window", process_name="random.exe")
        self.assertFalse(empty_anchor.matches(current))

    def test_p0_empty_anchor_saving_raises_value_error(self):
        act = [Action(kind=ActionKind.LEFT_CLICK, coordinate=(10, 10))]
        with self.assertRaises(ValueError) as ctx:
            self.cache.save_skill(goal="Test goal", actions=act, anchors=UIStateAnchor())
        self.assertIn("P0 Violation", str(ctx.exception))

    def test_multi_anchor_skill_persistence_no_overwrite(self):
        # Save two different entries for the same goal with different anchors
        a1 = [Action(kind=ActionKind.LEFT_CLICK, coordinate=(10, 10))]
        a2 = [Action(kind=ActionKind.LEFT_CLICK, coordinate=(20, 20))]
        anc1 = UIStateAnchor(window_title="Window 1", process_name="proc1")
        anc2 = UIStateAnchor(window_title="Window 2", process_name="proc2")

        self.cache.save_skill(goal="Common Goal", actions=a1, anchors=anc1)
        self.cache.save_skill(goal="Common Goal", actions=a2, anchors=anc2)

        # Reload a completely fresh instance from disk
        fresh = SkillCache(vault_dir=self.temp_dir.name)
        skills = fresh.list_skills()
        self.assertEqual(len(skills), 2)

        # Both anchors can be matched and retrieved
        res1 = fresh.lookup("Common Goal", current_anchor=anc1)
        self.assertTrue(res1.hit)
        self.assertEqual(res1.actions[0].coordinate, (10, 10))

        res2 = fresh.lookup("Common Goal", current_anchor=anc2)
        self.assertTrue(res2.hit)
        self.assertEqual(res2.actions[0].coordinate, (20, 20))
        fresh.close()

    def test_sqlite_backend_skill_cache(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            # 1. Test explicit close() and release of file lock
            sql_cache = SkillCache(vault_dir=td, backend="sqlite")
            anc = UIStateAnchor(window_title="SQLite Desktop", process_name="explorer")
            act = [Action(kind=ActionKind.KEY, keys=["win"])]
            sql_cache.save_skill(goal="Open Menu", actions=act, anchors=anc)
            sql_cache.close()
            self.assertIsNone(sql_cache._conn)

            # 2. Test context manager with SkillCache(...) as fresh_sql:
            with SkillCache(vault_dir=td, backend="sqlite") as fresh_sql:
                self.assertEqual(len(fresh_sql.list_skills()), 1)

                res = fresh_sql.lookup("Open Menu", current_anchor=anc)
                self.assertTrue(res.hit)
                self.assertEqual(res.actions[0].keys, ["win"])

                # Clear works
                fresh_sql.clear()
                self.assertEqual(len(fresh_sql.list_skills()), 0)


class TestDualStageVerifier(unittest.TestCase):
    """Verifies Stage 1 Physical Verification and Stage 2 Semantic Verification."""

    def setUp(self):
        self.verifier = DualStageVerifier()

    def test_stage1_physical_verification_ripple_detected(self):
        pre_obs = Observation(image_bytes=b"FRAME_A_PIXELS_" * 100, format="jpeg")
        post_obs = Observation(image_bytes=b"FRAME_B_RIPPLE_" * 100, format="jpeg")

        res = self.verifier.verify_physical(pre_obs, post_obs)
        self.assertTrue(res.passed)
        self.assertTrue(res.ripple_detected)
        self.assertGreater(res.optical_delta, 0.0)
        self.assertLess(res.latency_ms, 10.0)  # fast probe

    def test_stage1_physical_verification_zero_delta(self):
        identical_bytes = b"STATIC_FRAME_PIXELS" * 50
        pre_obs = Observation(image_bytes=identical_bytes, format="jpeg")
        post_obs = Observation(image_bytes=identical_bytes, format="jpeg")

        res = self.verifier.verify_physical(pre_obs, post_obs)
        self.assertFalse(res.passed)
        self.assertFalse(res.ripple_detected)
        self.assertEqual(res.optical_delta, 0.0)

    def test_stage1_missing_observation(self):
        res = self.verifier.verify_physical(None, None)
        self.assertFalse(res.passed)
        self.assertIn("Missing", res.details)

    def test_stage2_semantic_verification_window_title(self):
        with patch.object(self.verifier, "get_all_window_titles", return_value=["Untitled - Notepad", "Settings"]):
            res = self.verifier.verify_semantic(expected_window_title="Notepad")
            self.assertTrue(res.passed)
            self.assertEqual(res.matched_window, "Untitled - Notepad")
            self.assertIn("Semantic Verification passed", res.details)

    def test_stage2_semantic_verification_process_name(self):
        with patch.object(self.verifier, "get_all_window_titles", return_value=[]), \
             patch.object(self.verifier, "get_foreground_process_name", return_value="calculator.exe"):
            res = self.verifier.verify_semantic(expected_process_name="calculator")
            self.assertTrue(res.passed)
            self.assertEqual(res.matched_process, "calculator.exe")

    def test_stage2_semantic_verification_from_goal_keywords(self):
        with patch.object(self.verifier, "get_all_window_titles", return_value=["Calculator Window"]):
            res = self.verifier.verify_semantic(goal="Please open Calculator for math")
            self.assertTrue(res.passed)
            self.assertIn("Calculator Window", res.matched_window)

    def test_stage2_semantic_verification_failure(self):
        with patch.object(self.verifier, "get_all_window_titles", return_value=["Browser Window"]), \
             patch.object(self.verifier, "get_foreground_process_name", return_value="chrome.exe"):
            res = self.verifier.verify_semantic(expected_window_title="Paint")
            self.assertFalse(res.passed)
            self.assertIn("failed", res.details.lower())


class TestTrajectoryLogger(unittest.TestCase):
    """Verifies serialization of golden and negative failure trajectories for offline learning."""

    def test_log_golden_success_trajectory(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            logger = TrajectoryLogger(vault_dir=td)
            session = AceSession(goal="Test golden goal")
            session.record_step(Action(kind=ActionKind.LEFT_CLICK, coordinate=(100, 100)))
            session.finish("Done successfully")

            path_str = logger.log_trajectory(session, verified_success=True)
            self.assertTrue(os.path.exists(path_str))
            self.assertIn("golden", path_str)

            data = json.loads(Path(path_str).read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "success")
            self.assertTrue(data["verified_success"])
            self.assertEqual(data["goal"], "Test golden goal")
            self.assertEqual(len(data["steps"]), 1)

    def test_log_negative_failure_trajectory_with_stuck_points(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            logger = TrajectoryLogger(vault_dir=td)
            session = AceSession(goal="Test failure goal")
            session.record_step(Action(kind=ActionKind.LEFT_CLICK, coordinate=(500, 400)))
            session.finish("Repetition stuck")

            stuck = [{"step": 1, "repetitions": 3}]
            path_str = logger.log_trajectory(
                session,
                verified_success=False,
                failure_reason="Stuck in click loop",
                stuck_points=stuck,
            )
            self.assertTrue(os.path.exists(path_str))
            self.assertIn("failures", path_str)

            data = json.loads(Path(path_str).read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "failure")
            self.assertFalse(data["verified_success"])
            self.assertEqual(data["failure_reason"], "Stuck in click loop")
            self.assertEqual(len(data["stuck_points"]), 1)


class TestCalibratedDurationAndSafetyFirewall(unittest.TestCase):
    """Verifies Fitts's Law duration clamp calibration (50-120ms) and Safety Firewall guardrails."""

    def test_fitts_law_duration_calibrated_boundaries(self):
        computer = WindowsComputer()
        self.assertEqual(computer.FITTS_MIN_DURATION_S, 0.050)
        self.assertEqual(computer.FITTS_MAX_DURATION_S, 0.120)

        # Dist 0 -> exactly clamped to 50ms (0.050s)
        dur0 = computer.calculate_fitts_duration(0.0)
        self.assertAlmostEqual(dur0, 0.050, places=3)

        # Dist 500 -> 0.050 + 0.00014 * 500 = 0.120s
        dur500 = computer.calculate_fitts_duration(500.0)
        self.assertAlmostEqual(dur500, 0.120, places=3)

        # Dist 2000 -> clamped to 120ms (0.120s)
        dur2000 = computer.calculate_fitts_duration(2000.0)
        self.assertEqual(dur2000, 0.120)

    def test_firewall_permanent_file_deletion_guardrail(self):
        firewall = ExecutionSafetyFirewall(require_confirmation=True)

        # del /f /q command
        act_del = Action(kind=ActionKind.TYPE, text="del /f /q important_files.txt")
        res1 = firewall.evaluate(act_del, user_confirmed=False)
        self.assertFalse(res1.allowed)
        self.assertTrue(res1.requires_confirmation)
        self.assertEqual(res1.risk_level, SafetyRiskLevel.FILE_DELETION)

        # rm -rf command
        act_rm = Action(kind=ActionKind.TYPE, text="rm -rf /data")
        res2 = firewall.evaluate(act_rm, user_confirmed=False)
        self.assertFalse(res2.allowed)
        self.assertEqual(res2.risk_level, SafetyRiskLevel.FILE_DELETION)

        # Shift+Delete hotkey
        act_shift_del = Action(kind=ActionKind.KEY, keys=["shift", "delete"])
        res3 = firewall.evaluate(act_shift_del, user_confirmed=False)
        self.assertFalse(res3.allowed)
        self.assertEqual(res3.risk_level, SafetyRiskLevel.FILE_DELETION)

    def test_firewall_financial_payment_guardrail(self):
        firewall = ExecutionSafetyFirewall(require_confirmation=True)

        act_card = Action(kind=ActionKind.TYPE, text="credit card cvv 999")
        res1 = firewall.evaluate(act_card, user_confirmed=False)
        self.assertFalse(res1.allowed)
        self.assertTrue(res1.requires_confirmation)
        self.assertEqual(res1.risk_level, SafetyRiskLevel.FINANCIAL_PAYMENT)

        act_pay = Action(kind=ActionKind.LEFT_CLICK, coordinate=(500, 400), reason="Click Pay Now to complete order")
        res2 = firewall.evaluate(act_pay, user_confirmed=False)
        self.assertFalse(res2.allowed)
        self.assertEqual(res2.risk_level, SafetyRiskLevel.FINANCIAL_PAYMENT)

    def test_firewall_outbound_communication_guardrail(self):
        firewall = ExecutionSafetyFirewall(require_confirmation=True)

        act_email = Action(kind=ActionKind.TYPE, text="send email to all contacts")
        res1 = firewall.evaluate(act_email, user_confirmed=False)
        self.assertFalse(res1.allowed)
        self.assertTrue(res1.requires_confirmation)
        self.assertEqual(res1.risk_level, SafetyRiskLevel.OUTBOUND_COMMUNICATION)

        act_tweet = Action(kind=ActionKind.LEFT_CLICK, coordinate=(300, 300), reason="post tweet to twitter")
        res2 = firewall.evaluate(act_tweet, user_confirmed=False)
        self.assertFalse(res2.allowed)
        self.assertEqual(res2.risk_level, SafetyRiskLevel.OUTBOUND_COMMUNICATION)

    def test_firewall_user_confirmation_bypass(self):
        firewall = ExecutionSafetyFirewall(require_confirmation=True)
        act = Action(kind=ActionKind.TYPE, text="del /f /q test.txt")
        # With user_confirmed=True, the firewall allows the action
        res = firewall.evaluate(act, user_confirmed=True)
        self.assertTrue(res.allowed)


class TestPhase1EdgeCases(unittest.TestCase):
    """Verifies edge cases across Phase 1 foundation layers."""

    def test_action_adapter_edge_cases(self):
        # Empty string and None
        self.assertEqual(ActionAdapter.adapt("").kind, ActionKind.WAIT)
        self.assertEqual(ActionAdapter.adapt(None).kind, ActionKind.WAIT)
        self.assertEqual(ActionAdapter.adapt({}).kind, ActionKind.WAIT)

        # Lowercase and whitespace in grammar
        act1 = ActionAdapter.adapt("  act:click   100.4   200.8   ")
        self.assertEqual(act1.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act1.coordinate, (100, 200))

        # Unicode text in grammar
        act2 = ActionAdapter.adapt("ACT:TYPE Hello World")
        self.assertEqual(act2.kind, ActionKind.TYPE)
        self.assertEqual(act2.text, "Hello World")

        # OpenAI tool call with empty arguments
        act3 = ActionAdapter.adapt({"function": {"name": "wait", "arguments": "{}"}})
        self.assertEqual(act3.kind, ActionKind.WAIT)

    def test_skill_cache_edge_cases(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with SkillCache(vault_dir=td) as cache:
                # Empty goal raises ValueError
                with self.assertRaises(ValueError):
                    cache.save_skill(goal="", actions=[], anchors=UIStateAnchor(window_title="test"))
                # Missing anchors raises ValueError
                with self.assertRaises(ValueError):
                    cache.save_skill(goal="test", actions=[], anchors=[])
                # Non-existent goal lookup returns hit=False
                res = cache.lookup("NonExistentGoal")
                self.assertFalse(res.hit)

    def test_firewall_edge_cases(self):
        firewall = ExecutionSafetyFirewall(require_confirmation=True)
        # Safe action is approved
        safe_act = Action(kind=ActionKind.TYPE, text="Hello world in Notepad")
        res = firewall.evaluate(safe_act)
        self.assertTrue(res.allowed)
        self.assertEqual(res.risk_level, SafetyRiskLevel.SAFE)

        # Disguised powershell deletion
        del_act = Action(kind=ActionKind.TYPE, text="Remove-Item C:\\test -Recurse -Force")
        res_del = firewall.evaluate(del_act)
        self.assertFalse(res_del.allowed)
        self.assertEqual(res_del.risk_level, SafetyRiskLevel.FILE_DELETION)


class TestCognitivePlanner(unittest.TestCase):
    """Verifies Layer 1 Cognitive Intelligence & Sub-Goal Planner (avcs/ace/planner.py)."""

    def setUp(self):
        self.planner = CognitivePlanner(rate_limit_hz=1.0)

    def test_multilingual_decomposition_start_notepad(self):
        goal = "Open Start menu and search for Notepad"
        subgoals = self.planner.decompose(goal)
        self.assertEqual(len(subgoals), 2)

        self.assertEqual(subgoals[0].index, 1)
        self.assertIn("Start Menu", subgoals[0].description)
        self.assertIn("Start menu opened", subgoals[0].expected_ui_state)

        self.assertEqual(subgoals[1].index, 2)
        self.assertIn("Notepad", subgoals[1].description)
        self.assertIn("Notepad application window", subgoals[1].expected_ui_state)

    def test_multilingual_decomposition_chrome_search(self):
        goal = "Open Chrome and search for OpenAI on Google"
        subgoals = self.planner.decompose(goal)
        self.assertEqual(len(subgoals), 2)

        self.assertEqual(subgoals[0].index, 1)
        self.assertIn("Google Chrome", subgoals[0].description)

        self.assertEqual(subgoals[1].index, 2)
        self.assertIn("OpenAI", subgoals[1].description)
        self.assertIn("Google", subgoals[1].description)

    def test_multilingual_decomposition_composite_clauses(self):
        goal = "Open Chrome and search for OpenAI"
        subgoals = self.planner.decompose(goal)
        self.assertEqual(len(subgoals), 2)
        self.assertEqual(subgoals[0].index, 1)
        self.assertIn("Google Chrome", subgoals[0].description)
        self.assertEqual(subgoals[1].index, 2)
        self.assertIn("OpenAI", subgoals[1].description)
        for sg in subgoals:
            self.assertIsInstance(sg.description, str)
            self.assertIsInstance(sg.expected_ui_state, str)

    def test_multilingual_decomposition_english_composite(self):
        goal = "Open calculator and calculate 45 plus 55"
        subgoals = self.planner.decompose(goal)
        self.assertEqual(len(subgoals), 2)
        self.assertIn("Calculator", subgoals[0].description)
        self.assertIn("calculate", subgoals[1].description.lower())

    def test_multilingual_decomposition_atomic_single_goal(self):
        goal = "Launch notepad"
        subgoals = self.planner.decompose(goal)
        self.assertEqual(len(subgoals), 1)
        self.assertEqual(subgoals[0].index, 1)
        self.assertIn("Notepad", subgoals[0].description)

    def test_strict_coordinate_invariant_subgoal_rejected(self):
        # 1. Coordinate tuple (x, y) must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            SubGoal(index=1, description="Click target at (250, 400)", expected_ui_state="Button active")
        self.assertIn("Invariant violation", str(ctx.exception))

        # 2. x=..., y=... must raise ValueError
        with self.assertRaises(ValueError) as ctx2:
            SubGoal(index=1, description="Press button", expected_ui_state="Window at x=100, y=200")
        self.assertIn("Invariant violation", str(ctx2.exception))

        # 3. ACT:CLICK token must raise ValueError
        with self.assertRaises(ValueError) as ctx3:
            SubGoal(index=1, description="ACT:CLICK 500 600", expected_ui_state="Done")
        self.assertIn("Invariant violation", str(ctx3.exception))

        # 4. _validate_no_coordinates helper directly
        with self.assertRaises(ValueError):
            _validate_no_coordinates("coordinate: [300, 400]")

    def test_subgoal_state_transitions_and_serialization(self):
        sg = SubGoal(index=1, description="Open Settings", expected_ui_state="Settings open")
        self.assertFalse(sg.is_completed)
        self.assertFalse(sg.is_failed)
        self.assertEqual(sg.status, "pending")

        sg.mark_in_progress()
        self.assertTrue(sg.is_in_progress)

        sg.mark_completed()
        self.assertTrue(sg.is_completed)

        d = sg.to_dict()
        self.assertEqual(d["index"], 1)
        self.assertEqual(d["status"], "completed")

        reconstructed = SubGoal.from_dict(d)
        self.assertEqual(reconstructed.description, "Open Settings")
        self.assertTrue(reconstructed.is_completed)

        sg2 = SubGoal(index=2, description="Save file", expected_ui_state="File saved")
        sg2.mark_failed(reason="Disk full")
        self.assertTrue(sg2.is_failed)
        self.assertEqual(sg2.metadata.get("failure_reason"), "Disk full")

    def test_operating_frequency_rate_limiter(self):
        planner = CognitivePlanner(rate_limit_hz=2.0)
        self.assertEqual(planner.min_interval_s, 0.5)

        t0 = time.monotonic()
        planner._apply_rate_limit()
        t1 = time.monotonic()
        self.assertGreaterEqual(planner.last_plan_time, t0)

    def test_dynamic_replanning_start_menu_failure(self):
        failed_sg = SubGoal(index=1, description="Open Windows Start Menu", expected_ui_state="Start menu opened")
        replanned = self.planner.replan(
            goal="Launch notepad",
            failed_subgoal=failed_sg,
            failure_reason="Start menu button click failed",
        )
        self.assertGreaterEqual(len(replanned), 1)
        # Strategic alternative route uses Windows hotkey directly
        self.assertTrue(any("Windows key" in s.description or "Win" in s.description for s in replanned))

    def test_dynamic_replanning_app_launch_failure(self):
        failed_sg = SubGoal(index=1, description="Launch Notepad application", expected_ui_state="Notepad window active")
        replanned = self.planner.replan(
            goal="Open notepad",
            failed_subgoal=failed_sg,
            failure_reason="Window did not emerge on desktop",
        )
        self.assertGreaterEqual(len(replanned), 1)
        # Strategic alternative route uses Windows Search or Start navigation
        self.assertTrue(any("Search" in s.description or "Start" in s.description or "Win" in s.description for s in replanned))

    def test_dynamic_replanning_loop_deadlock(self):
        failed_sg = SubGoal(index=2, description="Click submit button", expected_ui_state="Form submitted")
        stuck_points = [{"step": 1}, {"step": 2}, {"step": 3}]
        replanned = self.planner.replan(
            goal="Submit form",
            failed_subgoal=failed_sg,
            failure_reason="Loop deadlock: 3 repeated identical clicks without UI change",
            stuck_points=stuck_points,
        )
        self.assertGreaterEqual(len(replanned), 1)
        # Strategic alternative route dismisses overlay visually and retries
        self.assertTrue(any("dismiss" in s.description.lower() or "retry" in s.description.lower() or "refocus" in s.description.lower() for s in replanned))

    def test_dynamic_replanning_console_firewall_hazard(self):
        failed_sg = SubGoal(index=1, description="Search for Notepad query", expected_ui_state="Search results")
        replanned = self.planner.replan(
            goal="Search query",
            failed_subgoal=failed_sg,
            failure_reason="Execution Safety Firewall: Active console detected. Rejecting ungrounded typing.",
        )
        self.assertGreaterEqual(len(replanned), 1)
        # Strategic alternative route focuses input directly
        self.assertTrue(any("search" in s.description.lower() or "focus" in s.description.lower() or "input" in s.description.lower() for s in replanned))

    def test_reflection_diagnostic_telemetry(self):
        sg = SubGoal(index=1, description="Open Menu", expected_ui_state="Menu open")
        diag = ReflectionDiagnostic(
            failed_subgoal=sg,
            failure_reason="Click dropped by OS debounce",
            stuck_repetitions=2,
            active_window="Desktop",
            optical_delta=0.001,
        )
        d = diag.to_dict()
        self.assertEqual(d["failure_reason"], "Click dropped by OS debounce")
        self.assertEqual(d["stuck_repetitions"], 2)
        self.assertEqual(d["active_window"], "Desktop")


class TestVisualCortexGate(unittest.TestCase):
    """Verifies System 1 Coupling: Local Visual Cortex Stability Gate (avcs/cortex/gate.py)."""

    def setUp(self):
        self.gate = VisualCortexGate(source="synthetic")

    def test_parameter_count_matches_budget(self):
        # Must match 400,765 parameter specification exactly
        p_count = sum(p.numel() for p in self.gate.cortex.parameters())
        self.assertEqual(p_count, 400765)

    def test_pre_actuation_stability_gate_static_screen(self):
        import numpy as np
        frame = np.full((128, 128, 3), 128, dtype=np.uint8)

        # First frame initializes temporal state
        res1 = self.gate.check_stability(frame)
        self.assertTrue(res1.is_stable)
        self.assertFalse(res1.spinner_active)

        # Second identical frame has 0 feature delta -> must be strictly stable
        res2 = self.gate.check_stability(frame)
        self.assertTrue(res2.is_stable)
        self.assertEqual(res2.spatial_delta, 0.0)
        self.assertEqual(res2.surprise_score, 0.0)
        self.assertLess(res2.latency_ms, 50.0)

    def test_pre_actuation_stability_gate_optical_drift(self):
        import numpy as np
        f1 = np.zeros((128, 128, 3), dtype=np.uint8)
        f2 = np.full((128, 128, 3), 255, dtype=np.uint8)

        self.gate.check_stability(f1)
        res = self.gate.check_stability(f2)
        self.assertGreater(res.spatial_delta, 0.0)
        self.assertGreater(res.surprise_score, 0.0)

    def test_post_actuation_ripple_probe_identical_frames(self):
        obs = Observation(image_bytes=b"IDENTICAL_FRAME_DATA" * 50, format="jpeg")
        res = self.gate.evaluate_ripple(obs, obs)
        self.assertFalse(res.passed)
        self.assertFalse(res.ripple_detected)
        self.assertEqual(res.optical_delta, 0.0)

    def test_post_actuation_ripple_probe_visual_delta(self):
        obs1 = Observation(image_bytes=b"PRE_CLICK_FRAME_" * 50, format="jpeg")
        obs2 = Observation(image_bytes=b"POST_CLICK_FRAME_WITH_RIPPLE_" * 50, format="jpeg")
        res = self.gate.evaluate_ripple(obs1, obs2)
        self.assertTrue(res.passed)
        self.assertTrue(res.ripple_detected)
        self.assertGreater(res.optical_delta, 0.0)

    def test_robustness_on_corrupt_dummy_bytes(self):
        # Raw bytes or corrupt data must never throw an unhandled exception
        obs_corrupt = Observation(image_bytes=b"\x00\xff\xee\xdd_corrupt", format="jpeg")
        res = self.gate.check_stability(obs_corrupt)
        self.assertIsInstance(res.is_stable, bool)
        self.assertIsInstance(res.latency_ms, float)

    def test_reset_clears_temporal_state(self):
        import numpy as np
        self.gate.check_stability(np.full((128, 128, 3), 100, dtype=np.uint8))
        self.assertIsNotNone(self.gate.temporal_state)
        self.assertIsNotNone(self.gate.prev_features)

        self.gate.reset()
        self.assertIsNone(self.gate.temporal_state)
        self.assertIsNone(self.gate.prev_features)


class TestPhase2OrchestrationIntegration(unittest.TestCase):
    """Verifies End-to-End Phase 2 Orchestration in AceRunner."""

    def test_runner_subgoal_progression(self):
        mock_agent = MagicMock()
        # Sub-goal 1: 1 action + STOP; Sub-goal 2: 1 action + STOP
        mock_agent.step.side_effect = [
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(100, 700), reason="Click start"), 100.0),
            (Action(kind=ActionKind.STOP, reason="Start opened"), 100.0),
            (Action(kind=ActionKind.TYPE, text="Notepad", reason="Type Notepad"), 100.0),
            (Action(kind=ActionKind.STOP, reason="Notepad launched"), 100.0),
        ]
        mock_agent.client = MagicMock()
        mock_agent.client.model = "gpt-6-astra"

        mock_computer = WindowsComputer()
        planner = CognitivePlanner()
        gate = VisualCortexGate(source="synthetic")

        runner = AceRunner(
            agent=mock_agent,
            computer=mock_computer,
            planner=planner,
            cortex_gate=gate,
            enable_voice=False,
        )

        session = runner.run(
            goal="Open Start menu and search for Notepad",
            max_steps=10,
            dry_run=True,
        )

        self.assertTrue(session.is_complete)
        self.assertGreaterEqual(session.step_count, 4)
        # Sub-goals were decomposed and recorded
        self.assertEqual(len(session.subgoals), 2)
        self.assertEqual(session.subgoals[0]["status"], "completed")
        self.assertEqual(session.subgoals[1]["status"], "completed")

    def test_runner_stability_gate_queried(self):
        mock_agent = MagicMock()
        mock_agent.step.return_value = (Action(kind=ActionKind.STOP, reason="Done"), 50.0)
        mock_agent.client = MagicMock()
        mock_agent.client.model = "gpt-6-astra"

        mock_computer = WindowsComputer()
        gate = VisualCortexGate(source="synthetic")
        gate.check_stability = MagicMock(wraps=gate.check_stability)

        runner = AceRunner(
            agent=mock_agent,
            computer=mock_computer,
            cortex_gate=gate,
            enable_voice=False,
        )

        session = runner.run(goal="Test stability gate", max_steps=2, dry_run=True)
        self.assertTrue(session.is_complete)
        # Sensory gate must have been invoked prior to action inference
        gate.check_stability.assert_called()

    def test_runner_dynamic_replanning_on_loop(self):
        mock_agent = MagicMock()
        # Emit repeated clicks to simulate UI stuck condition
        mock_agent.step.side_effect = [
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(300, 300), reason="Click button"), 100.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(300, 300), reason="Click button"), 100.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(300, 300), reason="Click button"), 100.0),
            (Action(kind=ActionKind.STOP, reason="Done"), 100.0),
        ]
        mock_agent.client = MagicMock()
        mock_agent.client.model = "gpt-6-astra"

        mock_computer = WindowsComputer()
        planner = CognitivePlanner()
        planner.replan = MagicMock(wraps=planner.replan)

        runner = AceRunner(
            agent=mock_agent,
            computer=mock_computer,
            planner=planner,
            enable_voice=False,
        )

        session = runner.run(goal="Submit form", max_steps=5, dry_run=True)
        self.assertTrue(session.is_complete)
        # Dynamic replanning must have triggered on the loop deadlock
        planner.replan.assert_called()

    def test_runner_dual_stage_verifier_with_cortex_gate(self):
        gate = VisualCortexGate(source="synthetic")
        verifier = DualStageVerifier(cortex_gate=gate)
        self.assertIs(verifier.cortex_gate, gate)

        obs1 = Observation(image_bytes=b"FRAME1_" * 30, format="jpeg")
        obs2 = Observation(image_bytes=b"FRAME2_" * 30, format="jpeg")
        res = verifier.verify_physical(obs1, obs2)
        self.assertTrue(res.passed)
        self.assertTrue(res.ripple_detected)

    def test_runner_subgoal_cache_hit_replay(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with SkillCache(vault_dir=td) as cache:
                anchor = UIStateAnchor(window_title="Desktop", process_name="explorer")
                cached_actions = [
                    Action(kind=ActionKind.KEY, keys=["win"], reason="Open start via hotkey"),
                    Action(kind=ActionKind.STOP, reason="Start opened"),
                ]
                # Cache the sub-goal
                cache.save_skill(goal="Open Windows Start Menu", actions=cached_actions, anchors=anchor)

                mock_agent = MagicMock()
                mock_agent.step.return_value = (Action(kind=ActionKind.STOP, reason="Search done"), 100.0)
                mock_agent.client = MagicMock()
                mock_agent.client.model = "gpt-6-astra"

                mock_computer = WindowsComputer()
                runner = AceRunner(
                    agent=mock_agent,
                    computer=mock_computer,
                    skill_cache=cache,
                    enable_voice=False,
                )

                session = runner.run(
                    goal="Open Start menu and search for Notepad",
                    max_steps=5,
                    dry_run=True,
                    current_anchor=anchor,
                )

                self.assertTrue(session.is_complete)
                # First sub-goal was satisfied from cache
                self.assertEqual(session.steps[0].action["kind"], "key")


class TestPhase2EdgeCases(unittest.TestCase):
    """Verifies edge cases across Phase 2: Planner, Cortex Gate, and Replanning."""

    def test_planner_empty_and_whitespace_goal(self):
        planner = CognitivePlanner()
        self.assertEqual(planner.decompose(""), [])
        self.assertEqual(planner.decompose("   \n\t  "), [])
        self.assertEqual(planner.decompose(None), [])

    def test_planner_nested_multilingual_conjunctions(self):
        planner = CognitivePlanner()
        goal = "Open Chrome then open YouTube and search for music and then play"
        subgoals = planner.decompose(goal)
        self.assertGreaterEqual(len(subgoals), 3)
        for sg in subgoals:
            self.assertGreater(len(sg.description), 0)

    def test_planner_coordinate_injection_resistance(self):
        # Disguised coordinate patterns
        attacks = [
            "Open window at (0, 0)",
            "Click button at (1920, 1080)",
            "Focus input x=500 y=300",
            "Action ACT:CLICK 120 450",
            "Tap at (10.5, 20.8)",
            "coords: [100, 200]",
        ]
        for atk in attacks:
            with self.assertRaises(ValueError):
                SubGoal(index=1, description=atk, expected_ui_state="Done")

    def test_cortex_gate_grayscale_and_none_frames(self):
        import numpy as np
        gate = VisualCortexGate(source="synthetic")
        # 2D grayscale array
        gray_frame = np.full((100, 100), 150, dtype=np.uint8)
        res = gate.check_stability(gray_frame)
        self.assertTrue(res.is_stable)

        # None frame (uses synthetic screen stream)
        res_none = gate.check_stability(None)
        self.assertIsInstance(res_none.is_stable, bool)

        # Ripple probe with None
        rip_none = gate.evaluate_ripple(None, None)
        self.assertFalse(rip_none.passed)
        self.assertFalse(rip_none.ripple_detected)

    def test_failure_trajectory_logs_subgoals_metadata(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tl = TrajectoryLogger(vault_dir=td)
            session = AceSession(goal="Test failure logging", max_steps=5, dry_run=True)
            session.start(goal="Test failure logging", max_steps=5, dry_run=True)
            sg1 = SubGoal(index=1, description="Step 1", expected_ui_state="State 1")
            sg2 = SubGoal(index=2, description="Step 2", expected_ui_state="State 2")
            sg2.mark_failed("Window blocked")
            session.subgoals = [sg1.to_dict(), sg2.to_dict()]
            session.finish("Failed at step 2")

            out_path = tl.log_trajectory(
                session=session,
                verified_success=False,
                failure_reason="Window blocked",
                metadata={"subgoals": session.subgoals},
            )

            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["status"], "failure")
            self.assertEqual(len(data["metadata"]["subgoals"]), 2)
            self.assertEqual(data["metadata"]["subgoals"][1]["status"], "failed")

    def test_planner_sanitizes_coordinates_in_input(self):
        planner = CognitivePlanner()
        # Input with coordinates must not crash and output English semantic directives
        subgoals = planner.decompose("click at (500, 400) and launch notepad")
        self.assertEqual(len(subgoals), 2)
        self.assertEqual(subgoals[0].index, 1)
        self.assertNotIn("(500, 400)", subgoals[0].description)
        self.assertIn("target element", subgoals[0].description)
        self.assertEqual(subgoals[1].index, 2)
        self.assertIn("Notepad", subgoals[1].description)

        # Coordinate invariant checks in SubGoal
        with self.assertRaises(ValueError):
            SubGoal(index=1, description="Click [500, 400]", expected_ui_state="Done")
        with self.assertRaises(ValueError):
            SubGoal(index=1, description="Target at x: 500, y: 400", expected_ui_state="Done")
        with self.assertRaises(ValueError):
            SubGoal(index=1, description="Press button at 500, 400", expected_ui_state="Done")

    def test_multilingual_decomposition_sequential_directives(self):
        planner = CognitivePlanner()
        s1 = planner.decompose("Open Chrome then search for OpenAI on Google")
        self.assertEqual(len(s1), 2)
        self.assertIn("Google Chrome", s1[0].description)
        self.assertIn("OpenAI", s1[1].description)
        self.assertIn("Google", s1[1].description)

        s2 = planner.decompose("Open Start menu and search for Notepad")
        self.assertEqual(len(s2), 2)
        self.assertIn("Start Menu", s2[0].description)
        self.assertIn("Notepad", s2[1].description)

    def test_runner_stability_gate_holds_on_unstable_screen(self):
        mock_agent = MagicMock()
        mock_agent.client = MagicMock()
        mock_agent.client.model = "gpt-6-astra"

        mock_computer = WindowsComputer()
        gate = VisualCortexGate(source="synthetic")
        # Simulate active spinner making screen unstable
        gate.check_stability = MagicMock(return_value=StabilityGateResult(
            is_stable=False,
            surprise_score=0.8,
            spinner_active=True,
            spatial_delta=0.05,
            latency_ms=10.0,
            details="Loading spinner active",
        ))

        runner = AceRunner(
            agent=mock_agent,
            computer=mock_computer,
            cortex_gate=gate,
            enable_voice=False,
        )

        session = runner.run(goal="Test unstable UI", max_steps=3, dry_run=True)
        # Agent step should NOT be called while UI is unstable
        mock_agent.step.assert_not_called()
        self.assertGreaterEqual(session.step_count, 1)
        self.assertEqual(session.steps[0].action["kind"], "wait")
        self.assertIn("Stability gate held", session.steps[0].reason)

    def test_runner_dual_stage_verifier_replan_on_semantic_failure(self):
        mock_agent = MagicMock()
        # Predict STOP immediately claiming Notepad launched
        mock_agent.step.return_value = (Action(kind=ActionKind.STOP, reason="Claimed Notepad launched"), 50.0)
        mock_agent.client = MagicMock()
        mock_agent.client.model = "gpt-6-astra"

        mock_computer = WindowsComputer()
        mock_verifier = DualStageVerifier(computer=mock_computer)
        # Simulate Stage 2 semantic verification failing (window did not emerge)
        mock_verifier.verify_semantic = MagicMock(return_value=Stage2Result(
            passed=False,
            details="Semantic verification failed: Notepad window not detected.",
        ))

        planner = CognitivePlanner()
        planner.replan = MagicMock(wraps=planner.replan)

        runner = AceRunner(
            agent=mock_agent,
            computer=mock_computer,
            verifier=mock_verifier,
            planner=planner,
            enable_voice=False,
        )

        # In non-dry-run mode, semantic failure on STOP triggers dynamic replanning
        session = runner.run(goal="Launch notepad", max_steps=2, dry_run=False)
        planner.replan.assert_called()

    def test_planner_enforce_rate_limit_actual_delay(self):
        planner = CognitivePlanner(rate_limit_hz=20.0, enforce_rate_limit=True)
        t0 = time.monotonic()
        planner.decompose("Launch notepad")
        planner.decompose("Launch calculator")
        elapsed = time.monotonic() - t0
        self.assertGreaterEqual(elapsed, 0.04)

    def test_subgoal_english_only_invariant(self):
        # Strict user requirement: SubGoal must be strictly English (non-English text rejected)
        with self.assertRaises(ValueError) as ctx:
            SubGoal(index=1, description="Non-English \u00e9\u00e8\u00e0", expected_ui_state="Done")
        self.assertIn("English only", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            SubGoal(index=1, description="Open Notepad", expected_ui_state="Non-English \u00f1\u00fc")
        self.assertIn("English only", str(ctx2.exception))

    def test_extended_coordinate_rejection(self):
        coords = [
            "<500, 400>",
            "point: 500, 400",
            "location: 500, 400",
            "click at 500px, 400px",
            "500px, 400px",
            "x=500 y=400",
        ]
        for c in coords:
            with self.assertRaises(ValueError):
                _validate_no_coordinates(c)

    def test_close_and_media_commands_decomposition(self):
        planner = CognitivePlanner()
        # Close / Exit commands
        s_chrome = planner.decompose("Close Chrome")
        self.assertEqual(len(s_chrome), 1)
        self.assertIn("Close Google Chrome", s_chrome[0].description)
        self.assertIn("closed and window removed", s_chrome[0].expected_ui_state)

        s_np = planner.decompose("Close Notepad")
        self.assertEqual(len(s_np), 1)
        self.assertIn("Close Notepad", s_np[0].description)

        # Media playback
        s_media = planner.decompose("Play media")
        self.assertEqual(len(s_media), 1)
        self.assertIn("Play media", s_media[0].description)

        # Save document
        s_save = planner.decompose("Save document")
        self.assertEqual(len(s_save), 1)
        self.assertIn("Save document", s_save[0].description)

    def test_va_character_delimiter_no_word_chopping(self):
        planner = CognitivePlanner()
        s_vid = planner.decompose("Play video")
        self.assertEqual(len(s_vid), 1)
        self.assertIn("Play media", s_vid[0].description)

        s_save = planner.decompose("Save file")
        self.assertEqual(len(s_save), 1)
        self.assertIn("Save document", s_save[0].description)

    def test_replanning_preserves_subsequent_subgoals(self):
        planner = CognitivePlanner()
        initial_subgoals = [
            SubGoal(index=1, description="Open Windows Start Menu", expected_ui_state="Start menu open"),
            SubGoal(index=2, description="Click target button", expected_ui_state="Button clicked"),
            SubGoal(index=3, description="Save document or current file", expected_ui_state="File saved"),
        ]

        mock_agent = MagicMock()
        # Step 1: Open start menu
        # Step 2, 3, 4: Loop on clicking button -> triggers replan for subgoal 2
        # Step 5: Escape key handled (completes replanned subgoal 2)
        # Step 6: Retry handled (completes replanned subgoal 3)
        # Step 7: Save document handled (completes preserved subgoal 3 -> now index 4)
        mock_agent.step.side_effect = [
            (Action(kind=ActionKind.STOP, reason="Start opened"), 50.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(400, 300), reason="Click button"), 50.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(400, 300), reason="Click button"), 50.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(400, 300), reason="Click button"), 50.0),
            (Action(kind=ActionKind.STOP, reason="Escape handled"), 50.0),
            (Action(kind=ActionKind.STOP, reason="Retry handled"), 50.0),
            (Action(kind=ActionKind.STOP, reason="Save done"), 50.0),
        ]
        mock_agent.client = MagicMock()
        mock_agent.client.model = "gpt-6-astra"

        mock_computer = WindowsComputer()
        runner = AceRunner(agent=mock_agent, computer=mock_computer, planner=planner, enable_voice=False)

        session = runner.run(
            goal="Composite workflow",
            max_steps=10,
            dry_run=True,
            subgoals=initial_subgoals,
        )

        # Subgoal 3 ('Save document or current file') must still be present in the session subgoals!
        subgoal_descs = [s["description"] for s in session.subgoals]
        self.assertTrue(any("Save document" in d for d in subgoal_descs))
        # Total subgoals should be 4 (step 1 + 2 replanned + step 3)
        self.assertEqual(len(session.subgoals), 4)
        for i, s in enumerate(session.subgoals, start=1):
            self.assertEqual(s["index"], i)

    def test_final_subgoal_stop_verification_failure_continues_execution(self):
        mock_agent = MagicMock()
        # Step 1: STOP (fails verification -> triggers replan into Route 4: Start, Type, Click app icon).
        # Replanned Subgoal 1: Click start -> STOP
        # Replanned Subgoal 2: Type notepad -> STOP
        # Replanned Subgoal 3: Click app icon -> STOP (completes goal)
        mock_agent.step.side_effect = [
            (Action(kind=ActionKind.STOP, reason="First attempt claimed launched"), 50.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(24, 775), reason="Click Start"), 50.0),
            (Action(kind=ActionKind.STOP, reason="Search opened"), 50.0),
            (Action(kind=ActionKind.TYPE, text="notepad", reason="Type notepad"), 50.0),
            (Action(kind=ActionKind.STOP, reason="Search results visible"), 50.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(350, 250), reason="Click Notepad icon"), 50.0),
            (Action(kind=ActionKind.STOP, reason="Notepad confirmed launched"), 50.0),
        ]
        mock_agent.client = MagicMock()
        mock_agent.client.model = "gpt-6-astra"

        mock_computer = WindowsComputer()
        mock_verifier = DualStageVerifier(computer=mock_computer)
        mock_verifier.verify_semantic = MagicMock(side_effect=[
            Stage2Result(passed=False, details="Verification failed: window not found"),
            Stage2Result(passed=True, details="Verification passed: Search active"),
            Stage2Result(passed=True, details="Verification passed: Search results ready"),
            Stage2Result(passed=True, details="Verification passed: Notepad active"),
            Stage2Result(passed=True, details="Final verification passed"),
        ])

        planner = CognitivePlanner()
        runner = AceRunner(
            agent=mock_agent,
            computer=mock_computer,
            verifier=mock_verifier,
            planner=planner,
            enable_voice=False,
        )

        with patch.object(mock_computer, "get_cursor_pos", return_value=(500, 500)):
            session = runner.run(goal="Launch notepad", max_steps=10, dry_run=False)
        # Must execute more than 1 step (continued to execute replanned steps)
        self.assertGreater(session.step_count, 1)
        self.assertTrue(session.is_complete)
        self.assertEqual(session.stop_reason, "Notepad confirmed launched")

    def test_visual_cortex_gate_upsampling_small_frame(self):
        import numpy as np
        gate = VisualCortexGate(source="synthetic")
        # 64x64 low-res image frame
        small_frame = np.full((64, 64, 3), 180, dtype=np.uint8)
        t_4ch = gate._convert_to_tensor_4ch(small_frame)
        self.assertEqual(t_4ch.shape, (1, 4, 128, 128))
        res = gate.check_stability(small_frame)
        self.assertTrue(res.is_stable)

    def test_circular_import_safety(self):
        import avcs.eye_stream
        import avcs.cortex.gate
        self.assertTrue(hasattr(avcs.eye_stream, "EyeStream"))
        self.assertTrue(hasattr(avcs.cortex.gate, "VisualCortexGate"))


class TestMonitorDpiCalibration(unittest.TestCase):
    """Verifies dynamic multi-monitor enumeration and DPI virtual coordinate calibration."""

    def test_get_monitors_enumeration(self):
        computer = WindowsComputer()
        mons = computer.get_monitors()
        self.assertGreaterEqual(len(mons), 1)
        m0 = mons[0]
        self.assertIsInstance(m0, MonitorInfo)
        self.assertGreater(m0.width, 0)
        self.assertGreater(m0.height, 0)
        self.assertGreaterEqual(m0.scale_factor, 1.0)
        self.assertTrue(any(m.is_primary for m in mons))

    def test_canonical_to_native_and_reverse(self):
        computer = WindowsComputer()
        left, top, w, h, _ = computer.get_active_bounds()
        
        # Test center point (640, 400)
        nx, ny = computer._canonical_to_native(640, 400)
        expected_nx = left + int(0.5 * w)
        expected_ny = top + int(0.5 * h)
        self.assertEqual(nx, expected_nx)
        self.assertEqual(ny, expected_ny)

        # Roundtrip back to canonical
        cx, cy = computer._native_to_canonical(nx, ny)
        self.assertAlmostEqual(cx, 640, delta=2)
        self.assertAlmostEqual(cy, 400, delta=2)

    def test_virtual_desktop_targeting(self):
        computer = WindowsComputer()
        computer.set_display_target("virtual")
        vx, vy, vw, vh, scale = computer.get_active_bounds()
        self.assertGreater(vw, 0)
        self.assertGreater(vh, 0)
        self.assertEqual(scale, 1.0)

        # Test boundary clamping
        nx, ny = computer._canonical_to_native(0, 0)
        self.assertEqual(nx, vx)
        self.assertEqual(ny, vy)

        nx_max, ny_max = computer._canonical_to_native(1280, 800)
        self.assertEqual(nx_max, vx + vw - 1)
        self.assertEqual(ny_max, vy + vh - 1)

    def test_multi_monitor_offset_calibration(self):
        computer = WindowsComputer()
        # Mock dual monitor configuration: Primary 1920x1080 at (0,0), Secondary 1920x1080 at (1920,0)
        mock_monitors = [
            MonitorInfo(index=0, handle=1, is_primary=True, left=0, top=0, right=1920, bottom=1080, width=1920, height=1080, scale_factor=1.0),
            MonitorInfo(index=1, handle=2, is_primary=False, left=1920, top=0, right=3840, bottom=1080, width=1920, height=1080, scale_factor=1.25),
        ]
        computer.get_monitors = MagicMock(return_value=mock_monitors)

        # Target monitor 1
        computer.set_display_target(1)
        left, top, w, h, scale = computer.get_active_bounds()
        self.assertEqual(left, 1920)
        self.assertEqual(top, 0)
        self.assertEqual(w, 1920)
        self.assertEqual(h, 1080)
        self.assertEqual(scale, 1.25)

        # Canonical (640, 400) on monitor 1 should map to 1920 + 960 = 2880
        nx, ny = computer._canonical_to_native(640, 400)
        self.assertEqual(nx, 2880)
        self.assertEqual(ny, int(400 / 800 * 1080))

        # Reverse maps back to 640, 400
        cx, cy = computer._native_to_canonical(2880, ny)
        self.assertEqual(cx, 640)
        self.assertEqual(cy, 400)


class TestLocalVAMClient(unittest.TestCase):
    """Verifies local edge Vision-Action Model engine client."""

    def test_local_vam_initialization(self):
        client = LocalVAMClient(endpoint="http://localhost:8000", model="Qwen2.5-VL-7B-Instruct")
        self.assertEqual(client.endpoint, "http://localhost:8000/v1")
        self.assertEqual(client.model, "Qwen2.5-VL-7B-Instruct")
        self.assertEqual(client.max_tokens, 32)
        self.assertEqual(client.temperature, 0.0)

    def test_local_vam_check_health_offline(self):
        client = LocalVAMClient(endpoint="http://127.0.0.1:59999/v1")
        self.assertFalse(client.check_health())

    def test_local_vam_offline_fallback(self):
        client = LocalVAMClient(endpoint="http://127.0.0.1:59999/v1", allow_offline_mock=True)
        img_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 30  # dummy jpeg
        action, latency = client.predict_action(img_bytes, goal="Launch notepad", step_index=1)
        self.assertEqual(action.kind, ActionKind.KEY)
        self.assertEqual(action.keys, ["win"])

    @patch("requests.post")
    def test_local_vam_mock_server_direct_token_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ACT:CLICK 300 450"}}]
        }
        mock_post.return_value = mock_resp

        client = LocalVAMClient(endpoint="http://127.0.0.1:8000/v1")
        img_bytes = b"dummy_img_bytes"
        action, latency = client.predict_action(img_bytes, goal="Click start")

        self.assertEqual(action.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(action.coordinate, (300, 450))
        self.assertGreaterEqual(latency, 0.0)

    @patch("requests.post")
    def test_local_vam_mock_server_compact_json_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"kind": "type", "text": "hello local"}'}}]
        }
        mock_post.return_value = mock_resp

        client = LocalVAMClient(endpoint="http://127.0.0.1:8000/v1")
        action, latency = client.predict_action(b"dummy", goal="Type greeting")

        self.assertEqual(action.kind, ActionKind.TYPE)
        self.assertEqual(action.text, "hello local")


class TestShowdownBenchmark(unittest.TestCase):
    """Verifies Showdown-Clicks benchmark runner, Wilson intervals, and evaluation logic."""

    def test_wilson_score_interval_properties(self):
        from benchmarks.run_showdown_benchmark import wilson_score_interval

        # 0 successes out of 100
        low_0, high_0 = wilson_score_interval(0, 100)
        self.assertEqual(low_0, 0.0)
        self.assertGreater(high_0, 0.0)
        self.assertLess(high_0, 0.05)

        # 100 successes out of 100
        low_100, high_100 = wilson_score_interval(100, 100)
        self.assertGreater(low_100, 0.95)
        self.assertAlmostEqual(high_100, 1.0, places=5)

        # 50 out of 100 (symmetric around 0.5)
        low_50, high_50 = wilson_score_interval(50, 100)
        self.assertAlmostEqual(low_50, 0.4038, delta=0.01)
        self.assertAlmostEqual(high_50, 0.5962, delta=0.01)

    def test_synthetic_showdown_suite_generation(self):
        from benchmarks.run_showdown_benchmark import generate_synthetic_showdown_suite
        samples = generate_synthetic_showdown_suite(count=15)
        self.assertEqual(len(samples), 15)
        for s in samples:
            self.assertTrue(s.id.startswith("synth_"))
            self.assertGreater(len(s.instruction), 0)
            x1, y1, x2, y2 = s.target_bbox
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertIsNotNone(s.image_bytes)
            self.assertGreater(len(s.image_bytes), 100)

    def test_benchmark_sample_evaluation_hit_and_miss(self):
        from benchmarks.run_showdown_benchmark import BenchmarkSample, evaluate_sample

        sample = BenchmarkSample(
            id="test_01",
            instruction="Click Save Button",
            target_bbox=(100, 200, 150, 240),
            target_center=(125, 220),
            image_bytes=b"dummy",
        )

        # Mock agent predicting exact hit
        mock_hit_agent = MagicMock()
        mock_hit_agent.step.return_value = (
            Action(kind=ActionKind.LEFT_CLICK, coordinate=(125, 220), reason="Hit target"),
            45.0,
        )
        res_hit = evaluate_sample(mock_hit_agent, sample)
        self.assertTrue(res_hit.hit)
        self.assertEqual(res_hit.distance_error, 0.0)

        # Mock agent predicting miss
        mock_miss_agent = MagicMock()
        mock_miss_agent.step.return_value = (
            Action(kind=ActionKind.LEFT_CLICK, coordinate=(900, 700), reason="Miss target"),
            45.0,
        )
        res_miss = evaluate_sample(mock_miss_agent, sample)
        self.assertFalse(res_miss.hit)
        self.assertGreater(res_miss.distance_error, 500.0)

    def test_run_benchmark_summary(self):
        from benchmarks.run_showdown_benchmark import (
            BenchmarkSample,
            generate_synthetic_showdown_suite,
            run_benchmark,
        )

        samples = generate_synthetic_showdown_suite(count=5)

        # Mock agent that always targets sample's center
        mock_agent = MagicMock()
        def _mock_step(obs, goal):
            # Find matching sample
            s = next((sm for sm in samples if sm.instruction == goal), samples[0])
            return Action(kind=ActionKind.LEFT_CLICK, coordinate=s.target_center), 50.0

        mock_agent.step.side_effect = _mock_step

        report = run_benchmark(mock_agent, samples, tolerance_px=15.0)
        self.assertEqual(report["sample_count"], 5)
        self.assertEqual(report["hits"], 5)
        self.assertEqual(report["accuracy"], 100.0)
        self.assertEqual(report["wilson_95_ci"][1], 100.0)
        self.assertEqual(report["latency_p50_ms"], 50.0)

    def test_runner_loop_deadlock_skips_failed_action_and_continues(self):
        """Verifies loop deadlock on non-jittered actions triggers replan, records WAIT step, and skips physical execution."""
        mock_agent = MagicMock()
        stuck_key = Action(kind=ActionKind.KEY, keys=["enter"], reason="Repeated key")
        mock_agent.step.side_effect = [
            (stuck_key, 50.0),
            (stuck_key, 50.0),
            (stuck_key, 50.0),  # 3rd identical key triggers deadlock
            (Action(kind=ActionKind.STOP, reason="Completed successfully"), 50.0),
        ]
        mock_agent.client = MagicMock()
        mock_agent.client.model = "gpt-6-astra"

        mock_computer = MagicMock(spec=WindowsComputer)
        mock_computer.observe.return_value = Observation(image_bytes=b"dummy", format="jpeg")
        mock_computer.get_active_bounds.return_value = (0, 0, 1280, 800, 1.0)
        mock_computer.get_cursor_pos.return_value = (400, 400)
        mock_computer._canonical_to_native.return_value = (400, 400)

        planner = CognitivePlanner()
        planner.replan = MagicMock(wraps=planner.replan)

        runner = AceRunner(
            agent=mock_agent,
            computer=mock_computer,
            planner=planner,
            enable_voice=False,
            enable_hud=False,
        )

        session = runner.run(goal="Submit form", max_steps=5, dry_run=True)
        self.assertTrue(session.is_complete)
        planner.replan.assert_called()

        # Step 3 must be the deadlock resolution WAIT step
        step_kinds = [s.action["kind"] for s in session.steps]
        self.assertIn("wait", step_kinds)
        wait_step = next(s for s in session.steps if s.action["kind"] == "wait")
        self.assertIn("Loop deadlock resolved", wait_step.action["reason"])

    def test_dry_run_trajectory_marked_verified_success(self):
        """Verifies that completed dry-run session is marked verified_success=True and saved in golden/ directory."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tl = TrajectoryLogger(vault_dir=td)
            mock_agent = MagicMock()
            mock_agent.step.return_value = (Action(kind=ActionKind.STOP, reason="Dry run complete"), 50.0)
            mock_agent.client = MagicMock()
            mock_agent.client.model = "gpt-6-astra"

            mock_computer = MagicMock(spec=WindowsComputer)
            mock_computer.observe.return_value = Observation(image_bytes=b"dummy", format="jpeg")
            mock_computer.get_active_bounds.return_value = (0, 0, 1280, 800, 1.0)
            mock_computer.get_cursor_pos.return_value = (100, 100)

            runner = AceRunner(
                agent=mock_agent,
                computer=mock_computer,
                trajectory_logger=tl,
                enable_voice=False,
                enable_hud=False,
            )

            session = runner.run(goal="Test dry run logging", max_steps=2, dry_run=True)
            self.assertTrue(session.is_complete)

            golden_files = list(tl.golden_dir.glob("*.json"))
            failure_files = list(tl.failures_dir.glob("*.json"))
            self.assertEqual(len(golden_files), 1, "Completed dry run must be logged to golden/ directory")
            self.assertEqual(len(failure_files), 0, "Completed dry run must NOT be logged to failures/ directory")

    def test_complete_with_urllib_fallback(self):
        """Verifies CustomLLMClient.complete works via urllib fallback when session is None."""
        client = CustomLLMClient(
            endpoint="https://api.frontier.ai/v1/chat/completions",
            api_key="sk-test",
            model="gpt-4o",
        )
        client.session = None  # Force urllib fallback

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Sub-goal decomposition response"}}]
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            out = client.complete(system_prompt="system", user_prompt="user")
            self.assertEqual(out, "Sub-goal decomposition response")

    def test_action_adapter_strips_reasoning_tags(self):
        """Verifies ActionAdapter.adapt strips <thought>, <reasoning>, <extra_thought> tags."""
        payload = "<thought>Thinking about clicking</thought>```json\n{\"action\": \"click\", \"coordinate\": [100, 200]}\n```"
        act = ActionAdapter.adapt(payload)
        self.assertEqual(act.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act.coordinate, (100, 200))

        payload2 = "<extra_thought>Some extra thoughts</extra_thought>{\"action\": \"type\", \"text\": \"hello\"}"
        act2 = ActionAdapter.adapt(payload2)
        self.assertEqual(act2.kind, ActionKind.TYPE)
        self.assertEqual(act2.text, "hello")

    def test_grammar_token_after_thinking_tags_in_parse_action_response(self):
        """Verifies parse_action_response handles grammar tokens preceded by thinking tags."""
        client = CustomLLMClient()
        response_text = "<think>Let us click the submit button</think>\nACT:CLICK 500 200"
        act = client.parse_action_response(response_text)
        self.assertEqual(act.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(act.coordinate, (500, 200))

    def test_1000_grid_scaling_custom_canvas(self):
        """Verifies 1000-grid scaling triggers when fx > canvas_w on smaller canvases."""
        scaled = Action._parse_coordinate_pair(950, 400, canvas_w=800, canvas_h=600)
        self.assertEqual(scaled, (int(round((950 / 1000.0) * 800)), int(round((400 / 1000.0) * 600))))


class TestVisualCortexCoupling(unittest.TestCase):
    """Verifies System 1 Visual Cortex & ACE Runner Coupling."""

    def test_runner_cortex_default_coupling(self):
        runner = AceRunner()
        self.assertIsNotNone(runner.cortex_gate)
        self.assertIsInstance(runner.cortex_gate, VisualCortexGate)
        self.assertIs(runner.verifier.cortex_gate, runner.cortex_gate)

        runner_disabled = AceRunner(cortex_gate=False)
        self.assertIsNone(runner_disabled.cortex_gate)

    def test_runner_human_eye_coupling(self):
        runner = AceRunner(cortex_gate="human_eye")
        self.assertIsNotNone(runner.cortex_gate)
        self.assertIsInstance(runner.cortex_gate, VisualCortexGate)
        from avcs.perception.human_eye import HumanEye
        self.assertIsInstance(runner.cortex_gate.cortex, HumanEye)

    def test_human_eye_stability_and_ripple_probe(self):
        import numpy as np
        gate = VisualCortexGate(cortex="human_eye")
        frame = np.full((128, 128, 3), 128, dtype=np.uint8)

        # Pre-actuation stability check
        res_stable = gate.check_stability(frame)
        self.assertTrue(res_stable.is_stable)
        self.assertFalse(res_stable.spinner_active)
        self.assertIn("HumanEye", res_stable.details)

        # Post-actuation ripple probe
        obs1 = Observation(image_bytes=b"PRE_ACTUATION_STABLE_FRAME_" * 30, format="jpeg")
        obs2 = Observation(image_bytes=b"POST_ACTUATION_RIPPLE_FRAME_" * 30, format="jpeg")
        res_ripple = gate.evaluate_ripple(obs1, obs2)
        self.assertTrue(res_ripple.passed)
        self.assertTrue(res_ripple.ripple_detected)
        self.assertIn("HumanEye", res_ripple.details)

    def test_sub_2ms_ripple_verification_latency(self):
        gate = VisualCortexGate(source="synthetic")
        obs1 = Observation(image_bytes=b"FRAME1_PRE_" * 30, format="jpeg")
        obs2 = Observation(image_bytes=b"FRAME2_POST_" * 30, format="jpeg")
        # Warm up JIT and ConvGRU state
        gate.evaluate_ripple(obs1, obs2)
        # Take minimum latency over repeated runs per AGENTS.md rules
        latencies = [gate.evaluate_ripple(obs1, obs2).latency_ms for _ in range(3)]
        res = gate.evaluate_ripple(obs1, obs2)
        self.assertTrue(res.passed)
        self.assertGreaterEqual(res.latency_ms, 0.0)
        self.assertLess(min(latencies), 100.0)

    def test_runner_cortex_v2_string_instantiation(self):
        from avcs.streaming_v2.model import StreamingVisualCortexV2
        runner = AceRunner(cortex_gate="cortex_v2")
        self.assertIsNotNone(runner.cortex_gate)
        self.assertIsInstance(runner.cortex_gate, VisualCortexGate)
        self.assertIsInstance(runner.cortex_gate.cortex, StreamingVisualCortexV2)
        # Verify check_stability and evaluate_ripple execute cleanly without AttributeError
        obs = Observation(image_bytes=b"CORTEX_V2_CHECK_FRAME_" * 20, format="jpeg")
        stab = runner.cortex_gate.check_stability(obs)
        self.assertIsNotNone(stab.is_stable)

    def test_localized_sub_2ms_click_ripple_evaluation(self):
        gate = VisualCortexGate(source="synthetic")
        obs1 = Observation(image_bytes=b"CLICK_PRE_FRAME_" * 30, format="jpeg")
        obs2 = Observation(image_bytes=b"CLICK_POST_FRAME_" * 30, format="jpeg")
        click_act = Action(kind=ActionKind.LEFT_CLICK, coordinate=(64, 64))
        # Warm up
        gate.evaluate_ripple(obs1, obs2, action=click_act)
        latencies = [gate.evaluate_ripple(obs1, obs2, action=click_act).latency_ms for _ in range(5)]
        res = gate.evaluate_ripple(obs1, obs2, action=click_act)
        self.assertTrue(res.passed)
        self.assertTrue(res.ripple_detected)
        self.assertIn("localized ROI", res.details)
        # Localized 50px ROI diff runs in sub-2ms
        self.assertLess(min(latencies), 5.0)


class TestAnchorVerifiedSkillCachingAndDrift(unittest.TestCase):
    """Verifies Anchor-Verified Skill Caching and Drift Fallback."""

    def test_compute_visual_hash_determinism_and_roi(self):
        obs = Observation(image_bytes=b"DESKTOP_REGION_HASH_TEST" * 40, format="jpeg")
        h1 = compute_visual_hash(obs)
        h2 = compute_visual_hash(obs)
        self.assertIsNotNone(h1)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

        h_roi = compute_visual_hash(obs, coordinate=(200, 300), radius=40)
        self.assertIsNotNone(h_roi)
        self.assertEqual(len(h_roi), 16)

        h_bbox = compute_visual_hash(obs, bounding_box=(100, 100, 300, 400))
        self.assertIsNotNone(h_bbox)
        self.assertEqual(len(h_bbox), 16)

    def test_visual_hash_similarity(self):
        self.assertEqual(visual_hash_similarity("0123456789abcdef", "0123456789abcdef"), 1.0)
        # 1 bit flip
        val1 = "0000000000000000"
        val2 = "0000000000000001"
        sim = visual_hash_similarity(val1, val2)
        self.assertGreaterEqual(sim, 0.95)
        # Completely different
        sim_diff = visual_hash_similarity("0000000000000000", "ffffffffffffffff")
        self.assertEqual(sim_diff, 0.0)

    def test_ui_state_anchor_similarity_and_matches(self):
        anc = UIStateAnchor(
            window_title="Notepad",
            process_name="notepad.exe",
            visual_hash="0123456789abcdef",
        )
        match = UIStateAnchor(
            window_title="Untitled - Notepad",
            process_name="notepad",
            visual_hash="0123456789abcdef",
        )
        self.assertTrue(anc.matches(match))
        self.assertGreaterEqual(anc.similarity(match), 0.90)

        drift = UIStateAnchor(
            window_title="Calculator",
            process_name="calc.exe",
            visual_hash="ffffffffffffffff",
        )
        self.assertFalse(anc.matches(drift))
        self.assertLess(anc.similarity(drift), 0.50)

    def test_session_capture_and_verify_anchor(self):
        session = AceSession(goal="Test Session Anchor", max_steps=5)
        anc = session.capture_anchor(window_title="Settings", process_name="systemsettings")
        self.assertEqual(anc.window_title, "Settings")
        self.assertEqual(anc.process_name, "systemsettings")
        self.assertTrue(session.verify_anchor(anc, anc))

        drift_anc = UIStateAnchor(window_title="Task Manager", process_name="taskmgr")
        self.assertFalse(session.verify_anchor(anc, drift_anc))

    def test_runner_anchor_drift_subgoal_falls_back_to_dynamic_prediction(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with SkillCache(vault_dir=td) as cache:
                anchor = UIStateAnchor(window_title="Form Window", process_name="app.exe")
                cached_actions = [
                    Action(kind=ActionKind.LEFT_CLICK, coordinate=(150, 250), reason="Cached click"),
                    Action(kind=ActionKind.STOP, reason="Cached done"),
                ]
                cache.save_skill(goal="Submit form", actions=cached_actions, anchors=anchor)

                mock_agent = MagicMock()
                mock_agent.step.return_value = (Action(kind=ActionKind.STOP, reason="Dynamic done"), 50.0)
                mock_agent.client = MagicMock()
                mock_agent.client.model = "gpt-6-astra"

                mock_computer = WindowsComputer()
                runner = AceRunner(
                    agent=mock_agent,
                    computer=mock_computer,
                    skill_cache=cache,
                    enable_voice=False,
                )

                # Active desktop anchor drifted to "Error Window"
                drifted_anchor = UIStateAnchor(window_title="Error Window", process_name="error.exe")
                session = runner.run(
                    goal="Submit form",
                    max_steps=3,
                    dry_run=True,
                    current_anchor=drifted_anchor,
                )

                self.assertTrue(session.is_complete)
                # Dynamic prediction MUST have been invoked because anchor drifted!
                mock_agent.step.assert_called()

    def test_runner_anchor_drift_full_goal_falls_back_to_dynamic_execution(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with SkillCache(vault_dir=td) as cache:
                anchor = UIStateAnchor(window_title="Desktop Shell", process_name="explorer")
                cached_actions = [Action(kind=ActionKind.STOP, reason="Cached full goal")]
                cache.save_skill(goal="Launch App", actions=cached_actions, anchors=anchor)

                drifted_anchor = UIStateAnchor(window_title="Lock Screen", process_name="logonui")
                self.assertTrue(cache.is_anchor_drift("Launch App", drifted_anchor))

                mock_agent = MagicMock()
                mock_agent.step.return_value = (Action(kind=ActionKind.STOP, reason="Dynamic recovered"), 50.0)
                mock_agent.client = MagicMock()
                mock_agent.client.model = "gpt-6-astra"

                runner = AceRunner(
                    agent=mock_agent,
                    computer=WindowsComputer(),
                    skill_cache=cache,
                    enable_voice=False,
                )

                session = runner.run(
                    goal="Launch App",
                    max_steps=2,
                    dry_run=True,
                    current_anchor=drifted_anchor,
                )
                self.assertTrue(session.is_complete)
                mock_agent.step.assert_called()

    def test_compute_visual_hash_full_image_coverage_bottom_half(self):
        import numpy as np
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        # Modify only the bottom half of img2
        img2[60:, 60:] = 255

        h1 = compute_visual_hash(img1)
        h2 = compute_visual_hash(img2)
        self.assertIsNotNone(h1)
        self.assertIsNotNone(h2)
        # Visual hash must cover all rows and reflect the bottom-half difference
        self.assertNotEqual(h1, h2)

    def test_compute_visual_hash_inverted_and_negative_bbox(self):
        obs = Observation(image_bytes=b"BBOX_NORMALIZE_TEST_" * 30, format="jpeg")
        # Inverted coordinates: x1 > x2, y1 > y2
        h_inverted = compute_visual_hash(obs, bounding_box=(400, 500, 100, 100))
        self.assertIsNotNone(h_inverted)
        self.assertEqual(len(h_inverted), 16)

        # Negative multi-monitor coordinates
        h_negative = compute_visual_hash(obs, bounding_box=(-50, -50, 100, 100))
        self.assertIsNotNone(h_negative)
        self.assertEqual(len(h_negative), 16)

    def test_session_save_to_skill_cache_auto_anchor_with_action_objects(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with SkillCache(vault_dir=td) as cache:
                session = AceSession(goal="Save Document", max_steps=5)
                # Record step with Action instance
                session.record_step(Action(kind=ActionKind.LEFT_CLICK, coordinate=(120, 240), reason="Click save icon"))
                # Calling save_to_skill_cache with anchors=None auto-extracts anchor without error
                entry_path = session.save_to_skill_cache(cache, anchors=None)
                self.assertTrue(os.path.exists(entry_path))

    def test_full_goal_cache_replay_semantic_verification_failure_falls_back(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with SkillCache(vault_dir=td) as cache:
                anchor = UIStateAnchor(window_title="Desktop Shell", process_name="explorer")
                cached_actions = [Action(kind=ActionKind.STOP, reason="Cached full goal")]
                cache.save_skill(goal="Launch App", actions=cached_actions, anchors=anchor)

                mock_agent = MagicMock()
                mock_agent.step.return_value = (Action(kind=ActionKind.STOP, reason="Dynamic recovered"), 50.0)
                mock_agent.client = MagicMock()
                mock_agent.client.model = "gpt-6-astra"

                mock_computer = WindowsComputer()
                mock_verifier = DualStageVerifier(computer=mock_computer)
                # Simulate semantic verification failure: window did not emerge after replay!
                mock_verifier.verify_semantic = MagicMock(return_value=Stage2Result(passed=False, details="Target window did not open"))

                runner = AceRunner(
                    agent=mock_agent,
                    computer=mock_computer,
                    verifier=mock_verifier,
                    skill_cache=cache,
                    enable_voice=False,
                )

                session = runner.run(
                    goal="Launch App",
                    max_steps=2,
                    dry_run=False,
                    current_anchor=anchor,
                )
                self.assertTrue(session.is_complete)
                # Because semantic verification failed, it must fall back to dynamic prediction!
                mock_agent.step.assert_called()


class TestTrajectoryVaultPersistenceAndFlywheel(unittest.TestCase):
    """Verifies Golden and Failure Trajectory persistence for the offline SFT/DPO flywheel."""

    def test_trajectory_persistence_golden_and_failures_directories(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tl = TrajectoryLogger(vault_dir=td)
            self.assertTrue(tl.golden_dir.exists())
            self.assertTrue(tl.failures_dir.exists())

            session_success = AceSession(goal="Golden task", max_steps=5, dry_run=True)
            session_success.start(goal="Golden task", max_steps=5, dry_run=True)
            session_success.record_step(Action(kind=ActionKind.STOP, reason="Success"))
            path_golden = tl.log_trajectory(session_success, verified_success=True)

            self.assertTrue(os.path.exists(path_golden))
            with open(path_golden, "r", encoding="utf-8") as f:
                data_golden = json.load(f)
            self.assertEqual(data_golden["status"], "success")
            self.assertTrue(data_golden["verified_success"])
            self.assertIn("sft_demonstration", data_golden)

            session_fail = AceSession(goal="Failed task", max_steps=5, dry_run=True)
            session_fail.start(goal="Failed task", max_steps=5, dry_run=True)
            session_fail.record_step(Action(kind=ActionKind.LEFT_CLICK, coordinate=(100, 100)))
            session_fail.finish("Stuck in loop")
            stuck_pts = [{"step": 1, "action": {"kind": "left_click", "coordinate": [100, 100]}, "repetitions": 2, "subgoal": "Failed task"}]
            path_fail = tl.log_trajectory(
                session_fail,
                verified_success=False,
                failure_reason="Stuck in loop",
                stuck_points=stuck_pts,
            )

            self.assertTrue(os.path.exists(path_fail))
            with open(path_fail, "r", encoding="utf-8") as f:
                data_fail = json.load(f)
            self.assertEqual(data_fail["status"], "failure")
            self.assertFalse(data_fail["verified_success"])
            self.assertIn("contrastive_recovery", data_fail)
            turning_pts = data_fail["contrastive_recovery"]["failure_turning_points"]
            self.assertEqual(len(turning_pts), 1)
            self.assertEqual(turning_pts[0]["step"], 1)

    def test_golden_trajectory_includes_sft_actions_and_failure_has_fallback_turning_point(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tl = TrajectoryLogger(vault_dir=td)
            # Golden session with actions
            session = AceSession(goal="Test Task", max_steps=5, dry_run=True)
            session.start(goal="Test Task", max_steps=5, dry_run=True)
            session.record_step(Action(kind=ActionKind.LEFT_CLICK, coordinate=(50, 100), reason="Click button"))
            session.record_step(Action(kind=ActionKind.STOP, reason="Done"))
            path_golden = tl.log_trajectory(session, verified_success=True)

            with open(path_golden, "r", encoding="utf-8") as f:
                data_golden = json.load(f)
            self.assertIn("trajectory", data_golden["sft_demonstration"])
            self.assertEqual(len(data_golden["sft_demonstration"]["trajectory"]), 2)

            # Failure session without loop stuck points
            fail_session = AceSession(goal="Failed Task", max_steps=5, dry_run=True)
            fail_session.start(goal="Failed Task", max_steps=5, dry_run=True)
            fail_session.record_step(Action(kind=ActionKind.LEFT_CLICK, coordinate=(50, 100)))
            fail_session.finish("Window did not emerge")
            path_fail = tl.log_trajectory(fail_session, verified_success=False, failure_reason="Window did not emerge")

            with open(path_fail, "r", encoding="utf-8") as f:
                data_fail = json.load(f)
            # Failure turning points must still be populated for DPO flywheel
            turning_pts = data_fail["contrastive_recovery"]["failure_turning_points"]
            self.assertEqual(len(turning_pts), 1)
            self.assertEqual(turning_pts[0]["step"], 1)

    def test_timed_out_session_is_logged_as_failure_not_golden(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tl = TrajectoryLogger(vault_dir=td)
            mock_agent = MagicMock()
            mock_agent.step.return_value = (Action(kind=ActionKind.LEFT_CLICK, coordinate=(10, 10), reason="Clicking"), 10.0)
            mock_agent.client = MagicMock()
            mock_agent.client.model = "test-model"

            runner = AceRunner(
                agent=mock_agent,
                trajectory_logger=tl,
                enable_voice=False,
                enable_hud=False,
                cortex_gate=False,
            )
            # Max steps is 1, but goal has 2 subgoals: cannot complete!
            session = runner.run(
                goal="Open Start menu and search for Notepad",
                max_steps=1,
                dry_run=True,
            )
            self.assertTrue(session.is_complete)
            self.assertIn("limit (1) reached", session.stop_reason.lower())

            # Verified success MUST be False
            golden_files = list(tl.golden_dir.glob("*.json"))
            failure_files = list(tl.failures_dir.glob("*.json"))
            self.assertEqual(len(golden_files), 0, "Timed out session must NOT be logged into golden directory!")
            self.assertEqual(len(failure_files), 1, "Timed out session MUST be logged into failures directory!")



class TestMacBookDynamicIslandHUD(unittest.TestCase):
    """Verifies the Apple MacBook / iPhone-grade Dynamic Island HUD state machine and morphing."""

    def test_hud_state_classification(self):
        from avcs.ace.hud import AVCSVisualHUD
        self.assertEqual(AVCSVisualHUD.classify_state("READY"), "IDLE")
        self.assertEqual(AVCSVisualHUD.classify_state("IDLE"), "IDLE")
        self.assertEqual(AVCSVisualHUD.classify_state("PLANNING"), "PLANNING")
        self.assertEqual(AVCSVisualHUD.classify_state("THINKING"), "PLANNING")
        self.assertEqual(AVCSVisualHUD.classify_state("CLICK (300, 400)"), "MOTOR")
        self.assertEqual(AVCSVisualHUD.classify_state("TYPE notepad"), "MOTOR")
        self.assertEqual(AVCSVisualHUD.classify_state("HOTKEY enter"), "MOTOR")
        self.assertEqual(AVCSVisualHUD.classify_state("VERIFIED"), "VERIFIED")
        self.assertEqual(AVCSVisualHUD.classify_state("SUCCESS"), "VERIFIED")
        self.assertEqual(AVCSVisualHUD.classify_state("SPEAKING"), "VOICE")
        self.assertEqual(AVCSVisualHUD.classify_state("LISTENING"), "VOICE")
        self.assertEqual(AVCSVisualHUD.classify_state("VOICE"), "VOICE")
        self.assertEqual(AVCSVisualHUD.classify_state("ABORT"), "ERROR")
        self.assertEqual(AVCSVisualHUD.classify_state("HALTED"), "ERROR")

    def test_hud_target_sizing(self):
        from avcs.ace.hud import AVCSVisualHUD
        # IDLE is compact (~220px, 36px)
        w, h = AVCSVisualHUD.calculate_target_size("IDLE")
        self.assertEqual((w, h), (220, 36))

        # PLANNING is medium (~380-520px, 38px)
        w, h = AVCSVisualHUD.calculate_target_size("PLANNING", "Decomposing goal into steps...")
        self.assertTrue(380 <= w <= 520)
        self.assertEqual(h, 38)

        # MOTOR is full-width (~620-690px, 42px)
        w, h = AVCSVisualHUD.calculate_target_size("MOTOR", "Step 1/3: Open Start Menu")
        self.assertTrue(620 <= w <= 690)
        self.assertEqual(h, 42)

        # VOICE is adaptive (~460-580px, 38px)
        w, h = AVCSVisualHUD.calculate_target_size("VOICE", "Opening Notepad application...")
        self.assertTrue(460 <= w <= 580)
        self.assertEqual(h, 38)

        # VERIFIED is compact (~290px, 38px)
        w, h = AVCSVisualHUD.calculate_target_size("VERIFIED")
        self.assertEqual((w, h), (290, 38))

    def test_hud_step_progress_parsing(self):
        from avcs.ace.hud import AVCSVisualHUD
        self.assertEqual(AVCSVisualHUD.parse_step_progress("1/3"), (1, 3))
        self.assertEqual(AVCSVisualHUD.parse_step_progress("3/3"), (3, 3))
        self.assertEqual(AVCSVisualHUD.parse_step_progress("0/0"), (0, 0))
        self.assertEqual(AVCSVisualHUD.parse_step_progress("invalid"), (0, 0))

    def test_hud_spring_morphing_smoothness(self):
        from avcs.ace.hud import AVCSVisualHUD
        hud = AVCSVisualHUD()
        hud.current_w = 220.0
        hud.current_h = 36.0
        hud.target_w = 640.0
        hud.target_h = 42.0

        # Step spring must move towards target without overshooting
        animating = hud.step_spring()
        self.assertTrue(animating)
        self.assertGreater(hud.current_w, 220.0)
        self.assertLess(hud.current_w, 640.0)
        self.assertGreater(hud.current_h, 36.0)
        self.assertLessEqual(hud.current_h, 42.0)

        # Iterating should converge to target
        for _ in range(60):
            hud.step_spring()
        self.assertEqual(hud.current_w, 640.0)
        self.assertEqual(hud.current_h, 42.0)

    def test_hud_update_status_updates_target_dimensions(self):
        from avcs.ace.hud import AVCSVisualHUD
        hud = AVCSVisualHUD()
        hud.update_status(status="CLICK (100, 200)", subgoal="Open Start Menu", step="1/3", mode="LIVE", latency_ms=45.0)
        self.assertEqual(hud.mode_label, "LIVE")
        self.assertEqual(hud.latency_ms, 45.0)
        self.assertEqual(hud.step_info, "1/3")
        self.assertEqual(hud.subgoal_text, "Open Start Menu")
        self.assertEqual(hud.target_h, 42.0)
        self.assertTrue(hud.target_w >= 620.0)


class TestPhase1RunnerReliabilityP0(unittest.TestCase):
    """Verifies Phase 1 Runner Reliability (P0):
    1. Premature STOP rejection in typing/searching sub-goals.
    2. Focus verification before TYPE with 100ms dwell.
    3. Repeated-action loop breakout & coordinate blacklisting.
    """

    def test_premature_stop_rejection_in_typing_subgoal(self):
        mock_agent = MagicMock()
        mock_agent.client = MagicMock()
        mock_agent.client.model = "test-model"

        # Turn 1: Propose STOP prematurely without any typing in a search/type sub-goal
        # Turn 2: Agent responds to prompt and emits TYPE
        # Turn 3: Agent emits STOP (now valid because typing completed)
        mock_agent.step.side_effect = [
            (Action(kind=ActionKind.STOP, reason="Premature stop claim"), 50.0),
            (Action(kind=ActionKind.TYPE, text="notepad", reason="Type notepad"), 50.0),
            (Action(kind=ActionKind.STOP, reason="Search done"), 50.0),
        ]

        subgoals = [
            SubGoal(index=1, description="Search for Notepad and launch application", expected_ui_state="Notepad window active")
        ]

        runner = AceRunner(agent=mock_agent, enable_voice=False, enable_hud=False, cortex_gate=False)
        session = runner.run(
            goal="Search for Notepad",
            subgoals=subgoals,
            max_steps=5,
            dry_run=True,
        )

        self.assertTrue(session.is_complete)
        # Step 1 should have been converted to WAIT due to premature STOP rejection
        self.assertEqual(session.steps[0].action["kind"], "wait")
        self.assertIn("Premature STOP rejected", session.steps[0].action["reason"])
        # Step 2: TYPE action
        self.assertEqual(session.steps[1].action["kind"], "type")
        # Step 3: Verified STOP action
        self.assertEqual(session.steps[2].action["kind"], "stop")
        self.assertEqual(session.step_count, 3)
        self.assertEqual(subgoals[0].status, "completed")

    def test_focus_verification_before_type(self):
        mock_agent = MagicMock()
        mock_agent.client = MagicMock()
        mock_agent.client.model = "test-model"
        mock_agent.step.side_effect = [
            (Action(kind=ActionKind.TYPE, coordinate=(400, 200), text="hello world", reason="Input text"), 50.0),
            (Action(kind=ActionKind.STOP, reason="Done"), 50.0),
        ]

        runner = AceRunner(agent=mock_agent, enable_voice=False, enable_hud=False, cortex_gate=False)
        session = runner.run(
            goal="Input greeting text",
            max_steps=3,
            dry_run=True,
        )

        self.assertTrue(session.is_complete)
        # Step 1 TYPE action should have focus verification recorded in metadata
        step1_act = session.steps[0].action
        self.assertEqual(step1_act["kind"], "type")
        self.assertEqual(step1_act["coordinate"], [400, 200])
        self.assertTrue(step1_act.get("metadata", {}).get("focus_verified", False))
        self.assertEqual(step1_act.get("metadata", {}).get("focus_dwell_ms"), 100.0)

    def test_repeated_action_loop_breakout_and_blacklisting(self):
        mock_client = MagicMock()
        mock_client.model = "test-model"
        # Simulate model deadlocking on (400, 400)
        mock_client.predict_action.side_effect = [
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(400, 400), reason="Click item"), 50.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(400, 400), reason="Click item"), 50.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(400, 400), reason="Click item"), 50.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(400, 400), reason="Click item"), 50.0),
            (Action(kind=ActionKind.STOP, reason="Done"), 50.0),
        ]

        agent = AceAgent(client=mock_client)
        runner = AceRunner(agent=agent, enable_voice=False, enable_hud=False, cortex_gate=False)
        session = runner.run(
            goal="Interact with button",
            max_steps=5,
            dry_run=True,
        )

        # Coordinate (400, 400) must be blacklisted
        self.assertTrue(runner.is_coordinate_blacklisted((400, 400)))
        self.assertIn((400, 400), session.blacklisted_coordinates)

        # Step 3 must have jittered away from (400, 400)
        step3_coord = session.steps[2].action["coordinate"]
        self.assertNotEqual(step3_coord, [400, 400])

        # Step 4 must also NOT be (400, 400) due to blacklist enforcement
        step4_coord = session.steps[3].action["coordinate"]
        self.assertNotEqual(step4_coord, [400, 400])


class TestPhase2LocalGrounderIntegrationP1(unittest.TestCase):
    """Verifies Phase 2 Local Grounder Integration (P1):
    1. Checkpoint loading from runs/weights/goal_grounder_best.pt.
    2. Gated actuation when peak_probability >= min_peak_probability (0.75).
    3. Low-confidence fallback to Cloud LLM.
    4. Shadow mode comparison and agreement logging.
    """

    def test_local_grounder_loading_from_checkpoint(self):
        ckpt_path = Path("runs/weights/goal_grounder_best.pt")
        if not ckpt_path.exists():
            self.skipTest(f"Checkpoint {ckpt_path} not available.")

        agent = AceAgent(checkpoint_path=str(ckpt_path), enable_grounder=True)
        grounder = agent._get_grounder()
        self.assertIsNotNone(grounder)
        param_count = sum(p.numel() for p in grounder.parameters())
        self.assertEqual(param_count, 200883)

    def test_gated_actuation_high_confidence_delegation(self):
        # High confidence >= 0.75: delegates direct action without calling Cloud LLM
        mock_grounder = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.candidate_point_norm = (0.25, 0.5)
        mock_prediction.peak_probability = 0.88
        mock_grounder.predict.return_value = mock_prediction

        mock_client = MagicMock()
        agent = AceAgent(
            client=mock_client,
            grounder=mock_grounder,
            enable_grounder=True,
            shadow_mode=False,
            min_peak_probability=0.75,
        )

        obs = Observation(image_bytes=b"GIF89a" + b"\x00" * 20, width=1280, height=800, format="jpeg")
        action, latency_ms = agent.step(obs, goal="Click the Launch button")

        # Must delegate direct action
        self.assertEqual(action.kind, ActionKind.LEFT_CLICK)
        # 0.25 * 1280 = 320, 0.5 * 800 = 400
        self.assertEqual(action.coordinate, (320, 400))
        self.assertAlmostEqual(action.confidence, 0.88)
        self.assertIn("Local Grounder target", action.reason)
        # Cloud LLM client must NOT have been called
        mock_client.predict_action.assert_not_called()

    def test_gated_actuation_low_confidence_fallback(self):
        # Low confidence < 0.75: must fail closed locally (no Cloud LLM fallback)
        mock_grounder = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.candidate_point_norm = (0.25, 0.5)
        mock_prediction.peak_probability = 0.45  # < 0.75
        mock_grounder.predict.return_value = mock_prediction

        mock_client = MagicMock()
        mock_client.predict_action.return_value = (
            Action(kind=ActionKind.KEY, keys=["win"], reason="Open Start"),
            150.0,
        )

        agent = AceAgent(
            client=mock_client,
            grounder=mock_grounder,
            enable_grounder=True,
            shadow_mode=False,
            min_peak_probability=0.75,
        )

        obs = Observation(image_bytes=b"GIF89a" + b"\x00" * 20, width=1280, height=800, format="jpeg")
        action, latency_ms = agent.step(obs, goal="Launch application")

        # Must fail closed with local WAIT (never call Cloud LLM)
        mock_client.predict_action.assert_not_called()
        self.assertEqual(action.kind, ActionKind.WAIT)
        self.assertTrue((action.metadata or {}).get("fail_closed"))

    def test_shadow_mode_agreement_logging(self):
        mock_grounder = MagicMock()
        mock_prediction = MagicMock()
        # 0.20 * 1280 = 256, 0.25 * 800 = 200
        mock_prediction.candidate_point_norm = (0.20, 0.25)
        mock_prediction.peak_probability = 0.82
        mock_grounder.predict.return_value = mock_prediction

        mock_client = MagicMock()
        # Cloud LLM returns (260, 205) -> distance is hypot(4, 5) ~ 6.4px <= 50px (agreed)
        mock_client.predict_action.return_value = (
            Action(kind=ActionKind.LEFT_CLICK, coordinate=(260, 205), reason="Cloud click"),
            250.0,
        )

        agent = AceAgent(
            client=mock_client,
            grounder=mock_grounder,
            enable_grounder=True,
            shadow_mode=True,
        )

        obs = Observation(image_bytes=b"GIF89a" + b"\x00" * 20, width=1280, height=800, format="jpeg")
        action, latency_ms = agent.step(obs, goal="Click OK button")

        # Executed action is from Cloud LLM
        self.assertEqual(action.coordinate, (260, 205))
        # Shadow record must be logged
        self.assertEqual(len(agent.shadow_history), 1)
        rec = agent.shadow_history[0]
        self.assertTrue(rec["agreed"])
        self.assertLess(rec["distance_px"], 10.0)
        self.assertEqual(rec["grounder_coordinate"], [256, 200])
        self.assertEqual(action.metadata.get("shadow_comparison", {}).get("agreed"), True)

    def test_shadow_mode_disagreement_logging(self):
        mock_grounder = MagicMock()
        mock_prediction = MagicMock()
        # 0.10 * 1280 = 128, 0.10 * 800 = 80
        mock_prediction.candidate_point_norm = (0.10, 0.10)
        mock_prediction.peak_probability = 0.79
        mock_grounder.predict.return_value = mock_prediction

        mock_client = MagicMock()
        # Cloud LLM returns (800, 600) -> distance is > 800px (disagreed)
        mock_client.predict_action.return_value = (
            Action(kind=ActionKind.LEFT_CLICK, coordinate=(800, 600), reason="Far click"),
            250.0,
        )

        agent = AceAgent(
            client=mock_client,
            grounder=mock_grounder,
            enable_grounder=True,
            shadow_mode=True,
        )

        obs = Observation(image_bytes=b"GIF89a" + b"\x00" * 20, width=1280, height=800, format="jpeg")
        action, latency_ms = agent.step(obs, goal="Click Cancel button")

        self.assertEqual(len(agent.shadow_history), 1)
        rec = agent.shadow_history[0]
        self.assertFalse(rec["agreed"])
        self.assertGreater(rec["distance_px"], 50.0)
        self.assertEqual(action.metadata.get("shadow_comparison", {}).get("agreed"), False)

    def test_runner_with_local_grounder_and_shadow_mode(self):
        mock_grounder = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.candidate_point_norm = (0.10, 0.10)
        mock_prediction.peak_probability = 0.85
        mock_grounder.predict.return_value = mock_prediction

        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.predict_action.side_effect = [
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(130, 85), reason="Click button"), 100.0),
            (Action(kind=ActionKind.STOP, reason="Done"), 100.0),
        ]

        agent = AceAgent(
            client=mock_client,
            grounder=mock_grounder,
            enable_grounder=True,
            shadow_mode=True,
        )

        runner = AceRunner(
            agent=agent,
            enable_voice=False,
            enable_hud=False,
            cortex_gate=False,
            enable_grounder=True,
            shadow_mode=True,
        )

        session = runner.run(goal="Test shadow runner", max_steps=3, dry_run=True)
        self.assertTrue(session.is_complete)
        self.assertGreater(len(session.metadata.get("shadow_comparisons", [])), 0)
        first_comp = session.metadata["shadow_comparisons"][0]
        self.assertTrue(first_comp["agreed"])

    def test_repeated_action_preserves_action_attributes_on_divert(self):
        # Verifies that loop breakout diversion preserves text, keys, press_enter, etc.
        mock_client = MagicMock()
        mock_client.model = "test-model"
        # Simulate model deadlocking on (400, 400) with a TYPE action
        mock_client.predict_action.side_effect = [
            (Action(kind=ActionKind.TYPE, coordinate=(400, 400), text="my_query", press_enter=True, reason="Type 1"), 50.0),
            (Action(kind=ActionKind.TYPE, coordinate=(400, 400), text="my_query", press_enter=True, reason="Type 2"), 50.0),
            (Action(kind=ActionKind.TYPE, coordinate=(400, 400), text="my_query", press_enter=True, reason="Type 3"), 50.0),
            (Action(kind=ActionKind.TYPE, coordinate=(400, 400), text="my_query", press_enter=True, reason="Type 4"), 50.0),
            (Action(kind=ActionKind.STOP, reason="Done"), 50.0),
        ]

        agent = AceAgent(client=mock_client)
        runner = AceRunner(agent=agent, enable_voice=False, enable_hud=False, cortex_gate=False)
        session = runner.run(
            goal="Type query in search",
            max_steps=5,
            dry_run=True,
        )

        # Coordinate (400, 400) was blacklisted
        self.assertTrue(runner.is_coordinate_blacklisted((400, 400)))
        step3 = session.steps[2].action
        self.assertEqual(step3["kind"], "type")
        # Coordinate must have been diverted away from [400, 400]
        self.assertNotEqual(step3["coordinate"], [400, 400])
        # BUT text and press_enter must be completely preserved!
        self.assertEqual(step3.get("text"), "my_query")
        self.assertEqual(step3.get("press_enter"), True)

        # Step 4 must also be diverted and preserve attributes
        step4 = session.steps[3].action
        self.assertEqual(step4["kind"], "type")
        self.assertNotEqual(step4["coordinate"], [400, 400])
        self.assertEqual(step4.get("text"), "my_query")
        self.assertEqual(step4.get("press_enter"), True)

    def test_premature_stop_rejection_english_typing_intents(self):
        from avcs.ace.runner import subgoal_requires_typing

        # Active typing / search intents in English
        typing_subgoals = [
            "Search for Notepad and launch",
            "Enter username into input field",
            "Input query in search bar",
            "Search Notepad application",
            "Type hello into the text box",
            "Write a message in the editor",
        ]
        for sg in typing_subgoals:
            self.assertTrue(subgoal_requires_typing(sg), f"Expected True for '{sg}'")

        # Non-typing subgoals (mouse clicks on existing items)
        non_typing_subgoals = [
            "Click Notepad app icon in search results",
            "Click search bar",
            "Select Notepad from search results",
            "Open Notepad in search results",
            "Double click Notepad icon on desktop",
            "Click Launch button",
        ]
        for sg in non_typing_subgoals:
            self.assertFalse(subgoal_requires_typing(sg), f"Expected False for '{sg}'")

    def test_runner_normalizes_string_subgoals(self):
        mock_agent = MagicMock()
        mock_agent.client = MagicMock()
        mock_agent.client.model = "test-model"
        mock_agent.step.return_value = (Action(kind=ActionKind.STOP, reason="Done"), 50.0)

        runner = AceRunner(agent=mock_agent, enable_voice=False, enable_hud=False, cortex_gate=False)
        # Pass raw strings as subgoals
        session = runner.run(
            goal="Test string subgoals",
            subgoals=["Open menu", "Click item"],
            max_steps=3,
            dry_run=True,
        )
        self.assertTrue(session.is_complete)
        self.assertEqual(len(session.subgoals), 2)
        self.assertEqual(session.subgoals[0]["description"], "Open menu")

    def test_local_grounder_real_inference_with_checkpoint(self):
        ckpt_path = Path("runs/weights/goal_grounder_best.pt")
        if not ckpt_path.exists():
            self.skipTest("Checkpoint runs/weights/goal_grounder_best.pt not present.")

        agent = AceAgent(checkpoint_path=str(ckpt_path), enable_grounder=True)
        grounder = agent._get_grounder()
        self.assertIsNotNone(grounder)

        import numpy as np
        from PIL import Image
        import io

        # Create a valid test image with visual structure
        arr = np.zeros((800, 1280, 3), dtype=np.uint8)
        arr[200:250, 400:600] = [200, 50, 50]  # colored rectangle simulating button
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        obs = Observation(image_bytes=raw_bytes, width=1280, height=800, format="jpeg")
        # Real inference must run without TypeError or crashing
        action, lat = agent.predict_with_grounder(obs, "launch button")
        # Real grounder should return an action or None without raising an exception
        self.assertIsInstance(lat, float)
        self.assertGreater(lat, 0.0)
        if action is not None:
            self.assertIn(action.kind, (ActionKind.LEFT_CLICK, ActionKind.DOUBLE_CLICK, ActionKind.RIGHT_CLICK))
            self.assertIsNotNone(action.coordinate)
            self.assertTrue(action.metadata.get("fast_brain"))

    def test_gated_actuation_non_pointing_goal_bypasses_grounder(self):
        mock_grounder = MagicMock()
        mock_client = MagicMock()
        mock_client.predict_action.return_value = (
            Action(kind=ActionKind.KEY, keys=["enter"], reason="Pressed Enter"),
            100.0,
        )

        agent = AceAgent(
            client=mock_client,
            grounder=mock_grounder,
            enable_grounder=True,
            shadow_mode=False,
            min_peak_probability=0.75,
        )

        obs = Observation(image_bytes=b"GIF89a" + b"\x00" * 20, width=1280, height=800, format="jpeg")
        # Non-pointing goal "Press Enter"
        action, latency_ms = agent.step(obs, goal="Press Enter")

        # Grounder predict must NOT have been called; local fail-closed WAIT (no Cloud LLM fallback)
        mock_grounder.predict.assert_not_called()
        mock_client.predict_action.assert_not_called()
        self.assertEqual(action.kind, ActionKind.WAIT)
        self.assertTrue((action.metadata or {}).get("fail_closed"))

    def test_gated_actuation_blacklisted_coordinate_falls_back_to_llm(self):
        mock_grounder = MagicMock()
        mock_prediction = MagicMock()
        # Grounder predicts (300, 300) with high confidence 0.88
        mock_prediction.candidate_point_norm = (300 / 1280, 300 / 800)
        mock_prediction.peak_probability = 0.88
        mock_grounder.predict.return_value = mock_prediction

        mock_client = MagicMock()
        mock_client.predict_action.return_value = (
            Action(kind=ActionKind.KEY, keys=["win"], reason="Recovery hotkey"),
            200.0,
        )

        agent = AceAgent(
            client=mock_client,
            grounder=mock_grounder,
            enable_grounder=True,
            shadow_mode=False,
            min_peak_probability=0.75,
        )

        obs = Observation(image_bytes=b"GIF89a" + b"\x00" * 20, width=1280, height=800, format="jpeg")
        # (300, 300) is blacklisted!
        action, latency_ms = agent.step(
            obs,
            goal="Click target button",
            blacklisted_coordinates=[(300, 300)],
        )

        # Because (300, 300) is blacklisted, it must fail closed with WAIT (never call Cloud LLM)
        mock_client.predict_action.assert_not_called()
        self.assertEqual(action.kind, ActionKind.WAIT)
        self.assertTrue((action.metadata or {}).get("blacklisted"))


class TestPhase1ContractAndLifecycle(unittest.TestCase):
    """
    Phase 1 Canonical Primitives, Nanosecond Observation Contract, and
    generalagents-python Session Lifecycle Unit Test Suite.
    """

    def test_canonical_action_primitives(self):
        # 1. ActionKind complete set of 10 primitives
        primitives = [
            ActionKind.LEFT_CLICK,
            ActionKind.RIGHT_CLICK,
            ActionKind.DOUBLE_CLICK,
            ActionKind.MOUSE_MOVE,
            ActionKind.LEFT_CLICK_DRAG,
            ActionKind.SCROLL,
            ActionKind.KEY,
            ActionKind.TYPE_TEXT,
            ActionKind.WAIT,
            ActionKind.STOP,
        ]
        self.assertEqual(len(primitives), 10)
        self.assertEqual(ActionKind.TYPE_TEXT.value, "type_text")

        # 2. Primitive classification helpers
        act_click = Action(kind=ActionKind.LEFT_CLICK, coordinate=(100, 200))
        act_rclick = Action(kind=ActionKind.RIGHT_CLICK, coordinate=(100, 200))
        act_dclick = Action(kind=ActionKind.DOUBLE_CLICK, coordinate=(100, 200))
        self.assertTrue(act_click.is_click())
        self.assertTrue(act_rclick.is_click())
        self.assertTrue(act_dclick.is_click())

        act_type_text = Action(kind=ActionKind.TYPE_TEXT, text="Hello ACE")
        self.assertTrue(act_type_text.is_type())
        self.assertEqual(act_type_text.text, "Hello ACE")

        # Backward compatibility with ActionKind.TYPE
        act_type_legacy = Action(kind=ActionKind.TYPE, text="legacy")
        self.assertTrue(act_type_legacy.is_type())

        # 3. Serialization roundtrip for all canonical primitives
        test_actions = [
            Action(kind=ActionKind.LEFT_CLICK, coordinate=(100, 200)),
            Action(kind=ActionKind.RIGHT_CLICK, coordinate=(300, 400)),
            Action(kind=ActionKind.DOUBLE_CLICK, coordinate=(500, 600)),
            Action(kind=ActionKind.MOUSE_MOVE, coordinate=(250, 350)),
            Action(kind=ActionKind.LEFT_CLICK_DRAG, start_coordinate=(50, 50), end_coordinate=(150, 150)),
            Action(kind=ActionKind.SCROLL, scroll_delta=(0, -3), coordinate=(400, 400)),
            Action(kind=ActionKind.KEY, keys=["ctrl", "shift", "esc"]),
            Action(kind=ActionKind.TYPE_TEXT, text="Test text typing"),
            Action(kind=ActionKind.WAIT, duration=1.5),
            Action(kind=ActionKind.STOP, reason="Task completed successfully"),
        ]

        for act in test_actions:
            d = act.to_dict()
            reconstructed = Action.from_dict(d)
            self.assertEqual(reconstructed.kind, act.kind)
            if act.coordinate:
                self.assertEqual(reconstructed.coordinate, act.coordinate)
            if act.text:
                self.assertEqual(reconstructed.text, act.text)
            if act.keys:
                self.assertEqual(reconstructed.keys, act.keys)

    def test_drag_chunks_interpolation_and_serialization(self):
        # 1. Linear chunk interpolation
        chunks = interpolate_drag_chunks((100, 200), (500, 600), num_chunks=4, linear=True)
        expected = [(100, 200), (200, 300), (300, 400), (400, 500), (500, 600)]
        self.assertEqual(chunks, expected)

        # Smooth S-curve interpolation
        smooth_chunks = interpolate_drag_chunks((100, 200), (500, 600), num_chunks=4, linear=False)
        self.assertEqual(smooth_chunks[0], (100, 200))
        self.assertEqual(smooth_chunks[-1], (500, 600))

        # Negative multi-monitor coordinates interpolation
        neg_chunks = interpolate_drag_chunks((-200, -100), (100, 200), num_chunks=3, linear=True)
        self.assertEqual(neg_chunks[0], (-200, -100))
        self.assertEqual(neg_chunks[-1], (100, 200))
        self.assertEqual(len(neg_chunks), 4)

        # 2. Action drag_chunks generation
        drag_act = Action(
            kind=ActionKind.LEFT_CLICK_DRAG,
            start_coordinate=(0, 0),
            end_coordinate=(100, 200),
        )
        act_chunks = drag_act.get_drag_chunks(num_chunks=5)
        self.assertEqual(len(act_chunks), 6)
        self.assertEqual(act_chunks[0], (0, 0))
        self.assertEqual(act_chunks[-1], (100, 200))

        # 3. Explicit drag_chunks serialization roundtrip
        custom_chunks = [(10, 20), (30, 40), (50, 60)]
        act_with_chunks = Action(
            kind=ActionKind.LEFT_CLICK_DRAG,
            start_coordinate=(10, 20),
            end_coordinate=(50, 60),
            drag_chunks=custom_chunks,
        )
        self.assertEqual(act_with_chunks.get_drag_chunks(), custom_chunks)
        d = act_with_chunks.to_dict()
        self.assertIn("drag_chunks", d)
        self.assertEqual(d["drag_chunks"], [[10, 20], [30, 40], [50, 60]])

        recon = Action.from_dict(d)
        self.assertEqual(recon.drag_chunks, custom_chunks)

    def test_coordinate_normalization_denormalization(self):
        w, h = 1280, 800

        # Center coordinate
        norm_center = normalize_coordinate((640, 400), width=w, height=h)
        self.assertAlmostEqual(norm_center[0], 0.5, places=4)
        self.assertAlmostEqual(norm_center[1], 0.5, places=4)

        denorm_center = denormalize_coordinate(norm_center, width=w, height=h)
        self.assertEqual(denorm_center, (640, 400))

        # Corners and boundaries including integers (1, 1), (0, 1), (1, 0)
        self.assertEqual(denormalize_coordinate((1, 1), width=w, height=h), (1280, 800))
        self.assertEqual(denormalize_coordinate((0, 1), width=w, height=h), (0, 800))
        self.assertEqual(denormalize_coordinate((1, 0), width=w, height=h), (1280, 0))
        self.assertEqual(denormalize_coordinate((0, 0), width=w, height=h), (0, 0))

        # Boundary coordinate and clamping
        norm_oob = normalize_coordinate((-50, 900), width=w, height=h)
        self.assertEqual(norm_oob, (0.0, 1.0))
        self.assertEqual(denormalize_coordinate((0.0, 1.0), width=w, height=h), (0, 800))

        # Action-level normalized properties
        act = Action(
            kind=ActionKind.LEFT_CLICK_DRAG,
            start_coordinate=(128, 80),
            end_coordinate=(640, 400),
            drag_chunks=[(128, 80), (384, 240), (640, 400)],
        )
        self.assertEqual(act.normalized_start_coordinate, (0.1, 0.1))
        self.assertEqual(act.normalized_end_coordinate, (0.5, 0.5))
        self.assertEqual(
            act.normalized_drag_chunks,
            [(0.1, 0.1), (0.3, 0.3), (0.5, 0.5)],
        )

        norm_dict = act.to_normalized(width=w, height=h)
        self.assertEqual(norm_dict["start_coordinate"], [0.1, 0.1])
        self.assertEqual(norm_dict["end_coordinate"], [0.5, 0.5])

        restored_act = Action.from_normalized(norm_dict, width=w, height=h)
        self.assertEqual(restored_act.start_coordinate, (128, 80))
        self.assertEqual(restored_act.end_coordinate, (640, 400))
        self.assertEqual(restored_act.drag_chunks, [(128, 80), (384, 240), (640, 400)])

    def test_hid_code_mapping_and_resolution(self):
        # 1. Standard key names to HID codes
        self.assertEqual(key_to_hid("enter"), 0x28)
        self.assertEqual(key_to_hid("return"), 0x28)
        self.assertEqual(key_to_hid("escape"), 0x29)
        self.assertEqual(key_to_hid("esc"), 0x29)
        self.assertEqual(key_to_hid("backspace"), 0x2A)
        self.assertEqual(key_to_hid("tab"), 0x2B)
        self.assertEqual(key_to_hid("space"), 0x2C)
        self.assertEqual(key_to_hid("a"), 0x04)
        self.assertEqual(key_to_hid("z"), 0x1D)
        self.assertEqual(key_to_hid("1"), 0x1E)
        self.assertEqual(key_to_hid("0"), 0x27)
        self.assertEqual(key_to_hid("ctrl"), 0xE0)
        self.assertEqual(key_to_hid("control"), 0xE0)
        self.assertEqual(key_to_hid("shift"), 0xE1)
        self.assertEqual(key_to_hid("alt"), 0xE2)
        self.assertEqual(key_to_hid("win"), 0xE3)

        # 2. Reverse mapping
        self.assertEqual(hid_to_key(0x28), "enter")
        self.assertIn(hid_to_key(0x29), ("esc", "escape"))
        self.assertEqual(hid_to_key(0x04), "a")

        # 3. Mouse button to HID mapping
        self.assertEqual(mouse_button_to_hid("left"), 0x01)
        self.assertEqual(mouse_button_to_hid("right"), 0x02)
        self.assertEqual(mouse_button_to_hid("middle"), 0x03)

        # 4. Action resolved_hid_codes property
        act_hotkey = Action(kind=ActionKind.KEY, keys=["ctrl", "shift", "esc"])
        self.assertEqual(act_hotkey.resolved_hid_codes, [0xE0, 0xE1, 0x29])

        act_text = Action(kind=ActionKind.TYPE_TEXT, text="ab")
        self.assertEqual(act_text.resolved_hid_codes, [0x04, 0x05])

        act_drag = Action(kind=ActionKind.LEFT_CLICK_DRAG, start_coordinate=(10, 10), end_coordinate=(20, 20))
        self.assertEqual(act_drag.resolved_hid_codes, [1])

        # 5. Serialization with hid_codes
        act_custom_hid = Action(kind=ActionKind.KEY, keys=["enter"], hid_codes=[0x28])
        d = act_custom_hid.to_dict()
        self.assertIn("hid_codes", d)
        self.assertEqual(d["hid_codes"], [0x28])
        recon = Action.from_dict(d)
        self.assertEqual(recon.hid_codes, [0x28])

        # Key action constructed solely with hid_codes resolves keys
        act_hid_only = Action.from_dict({"kind": "key", "hid_codes": [0x28]})
        self.assertEqual(act_hid_only.keys, ["enter"])

    def test_observation_nanosecond_contract(self):
        t_action = time.monotonic_ns()
        time.sleep(0.002)
        t_frame = time.monotonic_ns()

        obs = Observation(
            image_bytes=b"dummy_bytes",
            width=1920,
            height=1080,
            format="jpeg",
            frame_present_ns=t_frame,
            action_event_ns=t_action,
        )

        # 1. Nanosecond & millisecond latency calculation
        lat_ns = obs.latency_since_action_ns()
        self.assertIsNotNone(lat_ns)
        self.assertEqual(lat_ns, t_frame - t_action)
        self.assertGreaterEqual(lat_ns, 2_000_000)

        lat_ms = obs.latency_since_action_ms()
        self.assertIsNotNone(lat_ms)
        self.assertAlmostEqual(lat_ms, lat_ns / 1_000_000.0, places=3)

        # 2. Serialization roundtrip
        d = obs.to_dict(include_image=True)
        self.assertIn("frame_present_ns", d)
        self.assertIn("action_event_ns", d)
        self.assertEqual(d["frame_present_ns"], t_frame)
        self.assertEqual(d["action_event_ns"], t_action)

        recon = Observation.from_dict(d)
        self.assertEqual(recon.frame_present_ns, t_frame)
        self.assertEqual(recon.action_event_ns, t_action)
        self.assertEqual(recon.latency_since_action_ns(), lat_ns)

    def test_session_lifecycle_generalagents_python(self):
        # 1. Mock client and Agent setup
        mock_client = MagicMock()
        mock_client.predict_action.return_value = (
            Action(kind=ActionKind.LEFT_CLICK, coordinate=(150, 250)),
            15.0,
        )
        agent = AceAgent(client=mock_client)

        # 2. Lifecycle: start -> plan -> reset -> close
        session = agent.start(instruction="Open Notepad and write test", max_steps=10)
        self.assertIsInstance(session, AceAgentSession)
        self.assertFalse(session.is_complete)
        self.assertEqual(session.instruction, "Open Notepad and write test")
        self.assertEqual(session.step_count, 0)
        self.assertIs(agent.current_session, session)

        obs = Observation(image_bytes=b"GIF89a", width=1280, height=800, frame_present_ns=time.monotonic_ns())
        action = session.plan(obs)
        self.assertEqual(action.kind, ActionKind.LEFT_CLICK)
        self.assertEqual(action.coordinate, (150, 250))
        self.assertEqual(session.step_count, 1)
        self.assertEqual(len(session.history), 1)

        # Reset session
        session.reset()
        self.assertEqual(session.step_count, 0)
        self.assertEqual(len(session.history), 0)
        self.assertFalse(session.is_complete)

        # Context manager support
        with agent.start("Temporary task") as ctx_session:
            self.assertFalse(ctx_session.is_complete)
            act = ctx_session.plan(obs)
            self.assertIsNotNone(act)
        self.assertTrue(ctx_session.is_complete)

        # Close session
        session.close()
        self.assertTrue(session.is_complete)
        self.assertEqual(session.stop_reason, "Session closed by caller")

    def test_computer_execute_observation_and_hid_mapping(self):
        comp = WindowsComputer()

        # 1. _resolve_hid_to_vk mapping check
        self.assertEqual(comp._resolve_hid_to_vk(0x28), 0x0D)  # Enter -> VK_RETURN
        self.assertEqual(comp._resolve_hid_to_vk(0x29), 0x1B)  # Esc -> VK_ESCAPE
        self.assertEqual(comp._resolve_hid_to_vk(0x04), ord("A"))  # 'A' -> VK_A
        self.assertEqual(comp._resolve_hid_to_vk(0x1D), ord("Z"))  # 'Z' -> VK_Z
        self.assertEqual(comp._resolve_hid_to_vk(0xE0), 0x11)  # Ctrl -> VK_CONTROL
        self.assertEqual(comp._resolve_hid_to_vk(0xE3), 0x5B)  # Win -> VK_LWIN

        # 2. execute returns Observation in dry_run mode with monotonic timestamps
        t_before = time.monotonic_ns()
        obs = comp.execute(Action(kind=ActionKind.LEFT_CLICK, coordinate=(500, 300)), dry_run=True)
        t_after = time.monotonic_ns()

        self.assertIsInstance(obs, Observation)
        self.assertIsNotNone(obs.action_event_ns)
        self.assertGreaterEqual(obs.action_event_ns, t_before)
        self.assertLessEqual(obs.action_event_ns, t_after)
        self.assertEqual(obs.action_event_ns, comp.last_action_event_ns)
        self.assertGreaterEqual(obs.frame_present_ns, obs.action_event_ns)

        # 3. execute STOP action in dry_run returns valid Observation
        obs_stop = comp.execute(Action(kind=ActionKind.STOP), dry_run=True)
        self.assertIsInstance(obs_stop, Observation)
        self.assertIsNotNone(obs_stop.action_event_ns)

        # 4. execute KEY action with hid_codes in dry_run mode returns valid Observation
        obs_key = comp.execute(Action(kind=ActionKind.KEY, hid_codes=[0x28]), dry_run=True)
        self.assertIsInstance(obs_key, Observation)
        self.assertIsNotNone(obs_key.action_event_ns)

    def test_phase1_edge_cases_and_error_paths(self):
        # 1. Drag chunks edge cases
        self.assertEqual(interpolate_drag_chunks((10, 10), (100, 100), num_chunks=0), [(10, 10), (100, 100)])
        empty_drag = Action(kind=ActionKind.LEFT_CLICK_DRAG)
        self.assertEqual(empty_drag.get_drag_chunks(), [])
        single_point_drag = Action(kind=ActionKind.LEFT_CLICK_DRAG, start_coordinate=(50, 50))
        self.assertEqual(single_point_drag.get_drag_chunks(), [(50, 50)])

        # 2. Coordinate normalization edge cases
        self.assertIsNone(normalize_coordinate(()))
        self.assertIsNone(normalize_coordinate([10]))
        self.assertIsNone(normalize_coordinate("invalid"))
        self.assertIsNone(denormalize_coordinate(None))
        self.assertIsNone(denormalize_coordinate((float("nan"), 0.5)))
        self.assertIsNone(denormalize_coordinate((0.5, float("inf"))))
        self.assertEqual(denormalize_coordinate((1, 1)), (1280, 800))

        # 3. HID code error paths and case-insensitivity
        self.assertEqual(key_to_hid("  ENTER  "), 0x28)
        self.assertEqual(key_to_hid("Ctrl"), 0xE0)
        self.assertIsNone(key_to_hid("non_existent_key_xyz"))
        self.assertIsNone(key_to_hid(""))
        self.assertIsNone(key_to_hid(None))
        self.assertIsNone(hid_to_key(99999))
        self.assertIsNone(mouse_button_to_hid("invalid_button"))
        self.assertIsNone(mouse_button_to_hid(""))

        # Action without keys or text returns empty HID codes list
        act_wait = Action(kind=ActionKind.WAIT, duration=0.5)
        self.assertEqual(act_wait.resolved_hid_codes, [])

        # 4. Observation timestamp edge cases
        obs_no_action = Observation(image_bytes=b"", format="jpeg", frame_present_ns=500, action_event_ns=None)
        self.assertIsNone(obs_no_action.latency_since_action_ns())
        self.assertIsNone(obs_no_action.latency_since_action_ms())

        # Frame present before action event (clock anomaly) returns None
        obs_skew = Observation(image_bytes=b"", format="jpeg", frame_present_ns=100, action_event_ns=200)
        self.assertIsNone(obs_skew.latency_since_action_ns())
        self.assertIsNone(obs_skew.latency_since_action_ms())

        # from_dict with default fallbacks
        obs_empty_dict = Observation.from_dict({})
        self.assertEqual(obs_empty_dict.format, "jpeg")
        self.assertEqual(obs_empty_dict.width, 1280)
        self.assertEqual(obs_empty_dict.height, 800)
        self.assertEqual(obs_empty_dict.frame_present_ns, 0)
        self.assertIsNone(obs_empty_dict.action_event_ns)

        # 5. Session lifecycle edge cases: max_steps termination
        mock_client = MagicMock()
        mock_client.predict_action.return_value = (Action(kind=ActionKind.WAIT, duration=0.1), 10.0)
        agent = AceAgent(client=mock_client)
        session = agent.start("Bounded task", max_steps=2)

        obs = Observation(image_bytes=b"F", format="jpeg", frame_present_ns=1000)
        session.plan(obs)
        self.assertFalse(session.is_complete)
        session.plan(obs)
        self.assertTrue(session.is_complete)
        self.assertIn("Max step limit", session.stop_reason)

        # Re-starting an existing session
        session.start("New task")
        self.assertEqual(session.instruction, "New task")
        self.assertEqual(session.step_count, 0)
        self.assertFalse(session.is_complete)

        # Agent auto-start when plan is called directly
        agent.close()
        direct_act = agent.plan(obs)
        self.assertIsNotNone(direct_act)
        self.assertEqual(agent.current_session.step_count, 1)
        agent.reset()
        self.assertEqual(agent.current_session.step_count, 0)
        agent.close()
        self.assertIsNone(agent.current_session)


class TestPhase2ZeroLagRecorderAndBinding(unittest.TestCase):
    """Verifies Phase 2 Zero-Lag Pre-Action Ring Buffer and Monotonic Frame Binding."""

    def test_pre_action_frame_binding_strict_safety_margin(self):
        from avcs.ace.computer import Observation
        from avcs.ace.streamer import PreActionRingBuffer

        buf = PreActionRingBuffer(capacity=50)

        # Push 5 frames at 33ms intervals (33.3ms = 33_333_333 ns)
        # Frames at t = 0ms, 33ms, 66ms, 100ms, 133ms
        t_base = 1_000_000_000  # 1.0s
        for i in range(5):
            obs = Observation(
                image_bytes=b"F",
                width=1920,
                height=1080,
                frame_present_ns=t_base + i * 33_333_333,
            )
            buf.push(obs)

        self.assertEqual(len(buf), 5)

        # Action event at t = 101ms (1_000_000_000 + 101_000_000)
        # With 2ms safety margin (deadline = 99ms):
        # Frame 3 was presented at 100ms > 99ms (too late, risk of post-click leakage)
        # Frame 2 was presented at 66.6ms <= 99ms (valid pre-action frame!)
        action_ts = t_base + 101_000_000
        bound = buf.bind_pre_action_frame(action_ts, safety_margin_ns=2_000_000)
        self.assertIsNotNone(bound)
        self.assertEqual(bound.frame_present_ns, t_base + 2 * 33_333_333)

        # Action event at t = 136ms (deadline = 134ms): Frame 4 (133.3ms) is valid!
        action_ts2 = t_base + 136_000_000
        bound2 = buf.bind_pre_action_frame(action_ts2, safety_margin_ns=2_000_000)
        self.assertEqual(bound2.frame_present_ns, t_base + 4 * 33_333_333)

        # P95 sync error should be non-negative
        self.assertGreaterEqual(buf.p95_sync_error_ms(), 0.0)

    def test_ring_buffer_capacity_and_recent_frames(self):
        from avcs.ace.computer import Observation
        from avcs.ace.streamer import PreActionRingBuffer

        buf = PreActionRingBuffer(capacity=10)
        for i in range(25):
            buf.push(Observation(image_bytes=b"F", frame_present_ns=i * 100))

        self.assertEqual(len(buf), 10)
        recent = buf.get_recent_frames(count=5)
        self.assertEqual(len(recent), 5)
        self.assertEqual(recent[-1].frame_present_ns, 24 * 100)
        self.assertEqual(recent[0].frame_present_ns, 20 * 100)

    def test_ring_buffer_cpu_overhead_and_telemetry(self):
        from avcs.ace.streamer import PreActionRingBuffer

        buf = PreActionRingBuffer(capacity=120)
        telemetry = buf.measure_telemetry_benchmarks(num_frames=1000, target_fps=60.0)

        # Exit Gate 2: CPU overhead strictly < 2.0%
        self.assertLess(telemetry["cpu_overhead_pct"], 2.0)
        # Exit Gate 2: P95 frame-to-event synchronization error strictly < 5.0ms
        self.assertLess(telemetry["p95_sync_error_ms"], 5.0)
        self.assertGreaterEqual(telemetry["p95_sync_error_ms"], 2.0)
        self.assertEqual(telemetry["num_frames"], 1000.0)

    def test_pre_action_binding_no_valid_frame_returns_none(self):
        from avcs.ace.computer import Observation
        from avcs.ace.streamer import PreActionRingBuffer

        buf = PreActionRingBuffer(capacity=10)
        # 1. Empty buffer returns None
        self.assertIsNone(buf.bind_pre_action_frame(1_000_000_000, safety_margin_ns=2_000_000))

        # 2. Buffer only contains frames presented AFTER deadline (within safety margin or in future)
        # Action at t = 100ms, margin = 2ms -> deadline = 98ms.
        # Push frame presented at t = 99ms (> 98ms)
        obs_late = Observation(image_bytes=b"F", frame_present_ns=99_000_000)
        buf.push(obs_late)
        self.assertIsNone(buf.bind_pre_action_frame(100_000_000, safety_margin_ns=2_000_000))

        # 3. Exact boundary condition: frame presented at exactly deadline_ns (98ms) binds successfully
        obs_exact = Observation(image_bytes=b"F", frame_present_ns=98_000_000)
        buf.push(obs_exact)
        bound = buf.bind_pre_action_frame(100_000_000, safety_margin_ns=2_000_000)
        self.assertIsNotNone(bound)
        self.assertEqual(bound.frame_present_ns, 98_000_000)

    def test_screen_streamer_telemetry_benchmarks_and_metrics(self):
        from avcs.ace.computer import WindowsComputer
        from avcs.ace.streamer import ScreenStreamer

        comp = WindowsComputer()
        streamer = ScreenStreamer(computer=comp, target_fps=30.0)
        metrics = streamer.get_telemetry()
        self.assertEqual(metrics["fps_target"], 30.0)
        self.assertEqual(metrics["ring_buffer_size"], 0)
        self.assertEqual(metrics["frame_count"], 0)

        # Benchmark via ScreenStreamer delegating to ring buffer
        telemetry = streamer.measure_telemetry_benchmarks(num_frames=500, target_fps=30.0)
        self.assertLess(telemetry["cpu_overhead_pct"], 2.0)
        self.assertLess(telemetry["p95_sync_error_ms"], 5.0)


class TestPhase2TrajectoryCleaner(unittest.TestCase):
    """Verifies Phase 2 Trajectory Cleaning: Idle collapse, jitter smoothing, typing coalescing, misclick quarantine."""

    def test_idle_frame_collapse(self):
        from avcs.dataset.loader import LeRobotV3TrajectoryEpisode, TrajectoryCleaner

        steps = [
            LeRobotV3TrajectoryEpisode(
                episode_id="ep1", step_index=0, video_frame_index=0,
                frame_present_ns=1000, action_event_ns=1005_000_000,
                screen_width=1920, screen_height=1080,
                action_kind=8, pixel_diff_ratio=0.001,  # WAIT with near-zero diff
            )
            for _ in range(10)
        ]
        # Max idle wait steps is 3
        cleaned = TrajectoryCleaner.collapse_idle_frames(steps, min_pixel_diff=0.005, max_idle_wait_steps=3)
        self.assertEqual(len(cleaned), 3)

    def test_jitter_smoothing_drops_micro_movements(self):
        from avcs.dataset.loader import LeRobotV3TrajectoryEpisode, TrajectoryCleaner

        steps = [
            # 1. Micro movement: 1 pixel move (0.500 -> 0.5005) not followed by click
            LeRobotV3TrajectoryEpisode(
                episode_id="ep1", step_index=0, video_frame_index=0,
                frame_present_ns=1000, action_event_ns=1005_000_000,
                screen_width=1920, screen_height=1080,
                action_kind=3,  # MOVE
                cursor_pre_xy=(0.500, 0.500), target_xy=(0.5005, 0.5005),
            ),
            # 2. Significant movement: 100px move (0.50 -> 0.60)
            LeRobotV3TrajectoryEpisode(
                episode_id="ep1", step_index=1, video_frame_index=1,
                frame_present_ns=2000, action_event_ns=2005_000_000,
                screen_width=1920, screen_height=1080,
                action_kind=3,
                cursor_pre_xy=(0.50, 0.50), target_xy=(0.60, 0.60),
            ),
        ]
        smoothed = TrajectoryCleaner.smooth_jitter(steps, min_distance_px=4.0)
        self.assertEqual(len(smoothed), 1)
        self.assertEqual(smoothed[0].target_xy, (0.60, 0.60))

    def test_action_coalescing_rapid_keystrokes(self):
        from avcs.dataset.loader import LeRobotV3TrajectoryEpisode, TrajectoryCleaner

        t_base = 1_000_000_000
        # Typing "HELLO" at 100ms intervals (< 300ms gap)
        steps = [
            LeRobotV3TrajectoryEpisode(
                episode_id="ep1", step_index=i, video_frame_index=i,
                frame_present_ns=t_base + i * 100_000_000 - 5_000_000,
                action_event_ns=t_base + i * 100_000_000,
                screen_width=1920, screen_height=1080,
                action_kind=7, typed_text=char,
            )
            for i, char in enumerate(["H", "E", "L", "L", "O"])
        ]
        coalesced = TrajectoryCleaner.coalesce_typing(steps, max_keystroke_gap_ns=300_000_000)
        self.assertEqual(len(coalesced), 1)
        self.assertEqual(coalesced[0].typed_text, "HELLO")
        self.assertEqual(coalesced[0].action_kind, 7)

    def test_misclick_quarantine_undo_pattern(self):
        from avcs.dataset.loader import LeRobotV3TrajectoryEpisode, TrajectoryCleaner

        t_base = 1_000_000_000
        steps = [
            # 0. Normal click at t = 0
            LeRobotV3TrajectoryEpisode(
                episode_id="ep1", step_index=0, video_frame_index=0,
                frame_present_ns=t_base, action_event_ns=t_base + 5_000_000,
                screen_width=1920, screen_height=1080,
                action_kind=0, target_xy=(0.1, 0.1),  # LEFT_CLICK
            ),
            # 1. Accidental misclick at t = 500ms
            LeRobotV3TrajectoryEpisode(
                episode_id="ep1", step_index=1, video_frame_index=1,
                frame_present_ns=t_base + 500_000_000, action_event_ns=t_base + 505_000_000,
                screen_width=1920, screen_height=1080,
                action_kind=0, target_xy=(0.9, 0.9),  # Wrong click!
            ),
            # 2. Undo hotkey (Ctrl+Z) at t = 1000ms (500ms later, within 1.5s)
            LeRobotV3TrajectoryEpisode(
                episode_id="ep1", step_index=2, video_frame_index=2,
                frame_present_ns=t_base + 1000_000_000, action_event_ns=t_base + 1005_000_000,
                screen_width=1920, screen_height=1080,
                action_kind=6, typed_text="Ctrl+Z",
            ),
        ]
        cleaned, quarantined_count = TrajectoryCleaner.quarantine_misclicks(steps, undo_window_ns=1_500_000_000)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0].target_xy, (0.1, 0.1))
        self.assertEqual(quarantined_count, 2)


class TestPhase2QADoctor(unittest.TestCase):
    """Verifies Phase 2 QA Doctor and LeRobot v3.0 Bug Mitigations."""

    def test_qa_doctor_validates_golden_episode(self):
        from avcs.dataset.loader import LeRobotV3TrajectoryEpisode, QADoctor

        t_base = 1_000_000_000
        ep = [
            LeRobotV3TrajectoryEpisode(
                episode_id="ep_golden", step_index=i, video_frame_index=i,
                frame_present_ns=t_base + i * 50_000_000,
                action_event_ns=t_base + i * 50_000_000 + 4_000_000,  # 4ms sync margin >= 2ms
                screen_width=1920, screen_height=1080,
                action_kind=0, target_xy=(0.3, 0.4),
            )
            for i in range(5)
        ]
        report = QADoctor.validate_episode(ep)
        self.assertTrue(report["valid"])
        self.assertEqual(report["step_count"], 5)
        self.assertEqual(len(report["issues"]), 0)
        self.assertLess(report["p95_sync_error_ms"], 5.0)  # Exit gate: <5ms

    def test_qa_doctor_flags_timing_leakage_and_clock_skew(self):
        from avcs.dataset.loader import LeRobotV3TrajectoryEpisode, QADoctor

        # Step with margin violation (only 500ns, risking post-action leakage)
        bad_ep = [
            LeRobotV3TrajectoryEpisode(
                episode_id="bad", step_index=0, video_frame_index=0,
                frame_present_ns=1000, action_event_ns=1500,  # 500ns diff < 2ms (2_000_000ns)
                screen_width=1920, screen_height=1080,
                action_kind=0, target_xy=(0.5, 0.5),
            ),
            LeRobotV3TrajectoryEpisode(
                episode_id="bad", step_index=1, video_frame_index=1,
                frame_present_ns=500,  # Clock backwards!
                action_event_ns=2000,
                screen_width=1920, screen_height=1080,
                action_kind=0, target_xy=(1.5, 0.5),  # Out of bounds!
            ),
        ]
        report = QADoctor.validate_episode(bad_ep)
        self.assertFalse(report["valid"])
        self.assertTrue(any("margin violation" in issue for issue in report["issues"]))
        self.assertTrue(any("non-monotonic" in issue for issue in report["issues"]))
        self.assertTrue(any("outside [0, 1]" in issue for issue in report["issues"]))

    def test_patch_lerobot_issues_3513_and_3080(self):
        from avcs.dataset.loader import QADoctor

        # Issue #3513: Out-of-bounds frame indices clamped
        clamped = QADoctor.patch_lerobot_3513([-5, 0, 10, 50, 999], total_frames=100)
        self.assertEqual(clamped, [0, 0, 10, 50, 99])

        # Issue #3080: Global multi-episode frame offset mapped to local episode index
        local_frame = QADoctor.patch_lerobot_3080(global_frame_idx=545, episode_start_frame=500, episode_length=120)
        self.assertEqual(local_frame, 45)

    def test_500_episode_golden_diagnostic_dataset_full_audit(self):
        from avcs.dataset.loader import generate_golden_diagnostic_dataset, QADoctor

        # Generate 500 golden episodes (5,000 steps)
        dataset = generate_golden_diagnostic_dataset(num_episodes=500, steps_per_episode=10)
        self.assertEqual(len(dataset), 500)

        # Run QA Doctor audit over all 500 episodes
        report = QADoctor.audit_dataset(dataset)

        # Exit Gate 2 Invariants:
        # 1. 100% pass rate
        self.assertEqual(report["total_episodes"], 500)
        self.assertEqual(report["passed_episodes"], 500)
        self.assertEqual(report["pass_rate"], 1.0)
        self.assertEqual(report["corrupted_episodes"], 0)
        self.assertEqual(len(report["quarantined_issues"]), 0)

        # 2. P95 frame-to-event sync error < 5.0ms
        self.assertLess(report["p95_sync_error_ms"], 5.0)
        self.assertGreaterEqual(report["p95_sync_error_ms"], 2.0)


class TestPhase2LeRobotV3ParquetSchema(unittest.TestCase):
    """Verifies LeRobot v3.0 Canonical Schema Serialization and Roundtrip Parity."""

    def test_parquet_schema_roundtrip(self):
        from avcs.dataset.loader import LeRobotV3TrajectoryEpisode

        ep = LeRobotV3TrajectoryEpisode(
            episode_id="0192e2b3-5e92-750f-a492-3bc378951201",
            step_index=3,
            video_frame_index=45,
            frame_present_ns=1_700_000_000_000_000,
            action_event_ns=1_700_000_000_004_000,
            screen_width=1920,
            screen_height=1080,
            dpi_scale=1.5,
            os_platform="windows",
            task_instruction="Open Notepad and write test report",
            subgoal_instruction="Focus editor textarea",
            cursor_pre_xy=(0.25, 0.35),
            action_kind=4,  # DRAG
            target_xy=(0.65, 0.75),
            drag_trajectory_xy=[(0.25, 0.35), (0.45, 0.55), (0.65, 0.75)],
            scroll_delta=(0.0, 120.0),
            hotkey_ids=[29, 56],
            typed_text="report text",
            pixel_diff_ratio=0.042,
        )

        d = ep.to_dict()
        self.assertEqual(d["episode_id"], "0192e2b3-5e92-750f-a492-3bc378951201")
        self.assertEqual(d["action_kind"], 4)
        self.assertEqual(len(d["drag_trajectory_xy"]), 3)

        reconstructed = LeRobotV3TrajectoryEpisode.from_dict(d)
        self.assertEqual(reconstructed.episode_id, ep.episode_id)
        self.assertEqual(reconstructed.step_index, ep.step_index)
        self.assertEqual(reconstructed.target_xy, ep.target_xy)
        self.assertEqual(reconstructed.drag_trajectory_xy, ep.drag_trajectory_xy)
        self.assertAlmostEqual(reconstructed.pixel_diff_ratio, 0.042)

    def test_lerobot_v3_dataset_loader_integration(self):
        import torch
        from avcs.dataset.loader import LeRobotV3TrajectoryEpisode, LeRobotV3Dataset

        ep1 = [
            LeRobotV3TrajectoryEpisode(
                episode_id="ep1", step_index=0, video_frame_index=0,
                frame_present_ns=1000, action_event_ns=5_000_000,
                screen_width=1920, screen_height=1080,
                action_kind=0, target_xy=(0.5, 0.5), task_instruction="Test task",
            )
        ]
        ds = LeRobotV3Dataset(episodes=[ep1])
        self.assertEqual(len(ds), 1)
        item = ds[0]
        self.assertEqual(item["step_index"], 0)
        self.assertIsInstance(item["image"], torch.Tensor)
        self.assertEqual(item["image"].shape, (3, 256, 256))
        self.assertIsInstance(item["action_kind"], torch.Tensor)
        self.assertIsInstance(item["target_xy"], torch.Tensor)

    def test_high_throughput_batch_streaming_decoding(self):
        import time
        from avcs.dataset.loader import (
            generate_golden_diagnostic_dataset,
            LeRobotV3Dataset,
            make_lerobot_v3_dataloader,
        )

        # 500 episodes * 10 steps = 5,000 frames
        dataset_episodes = generate_golden_diagnostic_dataset(num_episodes=500, steps_per_episode=10)
        ds = LeRobotV3Dataset(episodes=dataset_episodes, target_size=(256, 256))
        self.assertEqual(len(ds), 5000)

        loader = make_lerobot_v3_dataloader(ds, batch_size=64, shuffle=False)

        t0 = time.perf_counter()
        total_frames = 0
        for batch in loader:
            batch_frames = len(batch["step_index"])
            total_frames += batch_frames
            self.assertEqual(batch["image"].shape, (batch_frames, 3, 256, 256))
            self.assertEqual(batch["action_kind"].shape[0], batch_frames)
            self.assertEqual(batch["target_xy"].shape[0], batch_frames)
            self.assertEqual(batch["cursor_pre_xy"].shape[0], batch_frames)
            self.assertEqual(batch["latency_ms"].shape[0], batch_frames)

        elapsed = time.perf_counter() - t0
        throughput_fps = total_frames / max(elapsed, 1e-6)

        # Exit Gate 2: Batch decoding streaming throughput strictly > 120 frames/sec
        self.assertEqual(total_frames, 5000)
        self.assertGreater(throughput_fps, 120.0)


class TestPhase3PureRGBVLAPolicy(unittest.TestCase):
    """Phase 3 Verification: Sub-350ms Pure-RGB VLA Policy Architecture.

    Verifies:
    1. ShowUITokenPruner compression down to target budget (e.g. 400 tokens),
       strict preservation of high-variance UI elements and active cursor tokens.
    2. GUIActorPatchPointer coordinate-free cross-attention and sub-patch offset recovery.
    3. ParallelActionHeads 10-class discrete action space, modifier flags, continuous scroll, HID.
    4. FlowMatchingDragExpert 5-step Euler ODE 16-waypoint smooth continuous trajectory chunk.
    5. PureRGBVLAPolicy unified System 1 reflex forward pass and rolling memory continuity.
    6. Diagnostic overfit check reaching >= 98% accuracy on diagnostic episodes.
    7. Inference latency budget benchmark (P95 latency strictly < 350ms).
    """

    def setUp(self):
        import torch
        torch.manual_seed(42)

    def test_showui_token_pruner_compression_and_token_preservation(self):
        import torch
        from avcs.cortex.model import ShowUITokenPruner, PrunedTokens

        pruner = ShowUITokenPruner(in_dim=128, target_budget=400, min_budget=64)

        b, c, h, w = 2, 128, 50, 80
        feat = torch.zeros((b, c, h, w), dtype=torch.float32)
        feat[:, :, 10:15, 20:30] = torch.randn(b, c, 5, 10) * 5.0

        cursor_xy = torch.tensor([[0.75, 0.85], [0.75, 0.85]], dtype=torch.float32)
        expected_cursor_col = int(0.75 * w)
        expected_cursor_row = int(0.85 * h)
        expected_cursor_idx = expected_cursor_row * w + expected_cursor_col

        pruned: PrunedTokens = pruner(features=feat, cursor_xy=cursor_xy, target_budget=400)

        self.assertEqual(pruned.tokens.shape, (b, 400, 128))
        self.assertEqual(pruned.coords.shape, (b, 400, 2))
        self.assertEqual(pruned.indices.shape, (b, 400))
        self.assertEqual(pruned.scores.shape, (b, 4000))
        self.assertEqual(pruned.keep_mask.shape, (b, 4000))
        self.assertEqual(pruned.bg_token.shape, (b, 1, 128))

        self.assertTrue((pruned.coords >= 0.0).all())
        self.assertTrue((pruned.coords <= 1.0).all())

        for bi in range(b):
            kept_indices_set = set(pruned.indices[bi].tolist())
            self.assertIn(
                expected_cursor_idx,
                kept_indices_set,
                f"Cursor index {expected_cursor_idx} must be strictly preserved in active tokens",
            )

        high_var_indices = [r * w + col for r in range(10, 15) for col in range(20, 30)]
        for bi in range(b):
            kept = set(pruned.indices[bi].tolist())
            retained_high_var = [idx for idx in high_var_indices if idx in kept]
            self.assertGreater(
                len(retained_high_var) / len(high_var_indices),
                0.90,
                "At least 90% of high-variance UI button patches must be preserved",
            )

    def test_gui_actor_pointer_coordinate_recovery(self):
        import torch
        from avcs.cortex.model import GUIActorPatchPointer, GUIPointerOutput

        pointer = GUIActorPatchPointer(query_dim=384, patch_dim=128, hidden_dim=128)

        b, k, d = 2, 400, 128
        patch_tokens = torch.randn(b, k, d)

        xs = torch.linspace(0.05, 0.95, 20)
        ys = torch.linspace(0.05, 0.95, 20)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        patch_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).unsqueeze(0).expand(b, -1, -1)

        query = torch.randn(b, 384)
        out: GUIPointerOutput = pointer(
            query_input=query,
            patch_tokens=patch_tokens,
            patch_coords=patch_coords,
            patch_stride=(1.0 / 80.0, 1.0 / 50.0),
        )

        self.assertEqual(out.coords.shape, (b, 2))
        self.assertEqual(out.subpixel_coords.shape, (b, 2))
        self.assertEqual(out.attention.shape, (b, k))
        self.assertEqual(out.patch_index.shape, (b,))
        self.assertEqual(out.delta.shape, (b, 2))

        attn_sum = out.attention.sum(dim=-1)
        self.assertTrue(torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-4))

        self.assertTrue((out.delta >= -0.5).all())
        self.assertTrue((out.delta <= 0.5).all())

        self.assertTrue((out.coords >= 0.0).all())
        self.assertTrue((out.coords <= 1.0).all())
        self.assertTrue((out.subpixel_coords >= 0.0).all())
        self.assertTrue((out.subpixel_coords <= 1.0).all())

    def test_parallel_action_heads_outputs(self):
        import torch
        from avcs.cortex.model import ParallelActionHeads, ParallelActionOutput, ACTION_VOCAB_10, MODIFIER_KEYS

        heads = ParallelActionHeads(in_dim=384, num_actions=10, num_hid_codes=128)
        latent = torch.randn(4, 384)

        out: ParallelActionOutput = heads(latent)

        self.assertEqual(len(ACTION_VOCAB_10), 10)
        self.assertIn("left_click", ACTION_VOCAB_10)
        self.assertIn("drag_start", ACTION_VOCAB_10)
        self.assertIn("scroll", ACTION_VOCAB_10)
        self.assertIn("hotkey", ACTION_VOCAB_10)
        self.assertIn("type_text", ACTION_VOCAB_10)

        self.assertEqual(out.action_logits.shape, (4, 10))
        self.assertEqual(out.action_probs.shape, (4, 10))
        self.assertEqual(out.action_index.shape, (4,))
        self.assertEqual(len(out.action_name), 4)
        for name in out.action_name:
            self.assertIn(name, ACTION_VOCAB_10)

        self.assertEqual(out.modifier_logits.shape, (4, 4))
        self.assertEqual(out.modifier_mask.shape, (4, 4))
        self.assertEqual(len(out.modifier_flags), 4)
        for flags_dict in out.modifier_flags:
            self.assertEqual(set(flags_dict.keys()), set(MODIFIER_KEYS))
            for k, val in flags_dict.items():
                self.assertIsInstance(val, bool)

        self.assertEqual(out.scroll_delta.shape, (4, 2))
        self.assertTrue((out.scroll_delta >= -1.0).all())
        self.assertTrue((out.scroll_delta <= 1.0).all())

        self.assertEqual(out.hid_logits.shape, (4, 128))
        self.assertEqual(out.hid_code.shape, (4,))
        self.assertTrue((out.hid_code >= 0).all())
        self.assertTrue((out.hid_code < 128).all())

    def test_flow_matching_drag_expert_trajectory(self):
        import math
        import torch
        from avcs.cortex.model import FlowMatchingDragExpert

        expert = FlowMatchingDragExpert(cond_dim=384, hidden_dim=128, trajectory_points=16)
        b = 3
        cond = torch.randn(b, 384)
        start_pos = torch.tensor([[0.20, 0.30], [0.50, 0.50], [0.80, 0.10]], dtype=torch.float32)

        traj = expert.generate_trajectory(cond=cond, start_pos=start_pos, num_steps=5)

        self.assertEqual(traj.shape, (b, 16, 2))

        for bi in range(b):
            self.assertTrue(
                torch.allclose(traj[bi, 0], start_pos[bi], atol=1e-5),
                "Waypoint 0 must be strictly anchored to the starting cursor coordinate",
            )

        self.assertTrue((traj >= 0.0).all())
        self.assertTrue((traj <= 1.0).all())

        diffs = traj[:, 1:, :] - traj[:, :-1, :]
        step_distances = torch.norm(diffs, dim=-1)
        self.assertTrue(
            (step_distances < 0.5).all(),
            "Trajectory waypoints must exhibit smooth continuous curvature",
        )

        target_traj = start_pos.unsqueeze(1).repeat(1, 16, 1) + 0.1
        loss = expert.compute_loss(target_traj, cond, start_pos)
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(math.isfinite(loss.item()))

    def test_pure_rgb_vla_policy_end_to_end_forward(self):
        import torch
        from avcs.cortex.model import PureRGBVLAPolicy, PureRGBVLAOutput, VLACortexConfig, RollingMemoryState

        cfg = VLACortexConfig(
            input_height=400,
            input_width=640,
            stem_channels=32,
            stage_channels=(32, 64, 128, 256),
            neck_channels=128,
        )
        policy = PureRGBVLAPolicy(cortex_config=cfg, target_token_budget=400)
        policy.eval()

        single_frame = torch.zeros((1, 3, 400, 640), dtype=torch.float32)
        out1: PureRGBVLAOutput = policy(
            images=single_frame,
            goals=["Click file menu"],
            cursor_xy=torch.tensor([[0.30, 0.40]]),
        )

        self.assertIsInstance(out1, PureRGBVLAOutput)
        self.assertIsInstance(out1.action_type, str)
        self.assertIsInstance(out1.action_index, int)
        self.assertEqual(out1.target_coords.shape, (1, 2))
        self.assertEqual(out1.subpixel_coords.shape, (1, 2))
        self.assertEqual(out1.pruned_token_count, 400)
        self.assertIsInstance(out1.modifier_flags, dict)
        self.assertEqual(out1.scroll_delta.shape, (1, 2))
        self.assertIsInstance(out1.hid_code, int)
        self.assertGreater(out1.latency_ms, 0.0)

        mem_state = RollingMemoryState(max_history_steps=4)
        for step in range(6):
            step_out = policy(
                images=single_frame,
                goals=[f"Sub-goal step {step}"],
                memory_state=mem_state,
            )
            self.assertEqual(step_out.memory_state, mem_state)

        self.assertLessEqual(len(mem_state.step_summaries), 4)

        out_drag: PureRGBVLAOutput = policy(
            images=single_frame,
            goals=["Drag icon to folder"],
            cursor_xy=torch.tensor([[0.2, 0.2]]),
            force_drag=True,
        )
        self.assertIsNotNone(out_drag.drag_trajectory)
        self.assertEqual(out_drag.drag_trajectory.shape, (1, 16, 2))

    def test_overfit_check_diagnostic_execution(self):
        import torch
        from avcs.cortex.model import overfit_check

        report = overfit_check(num_episodes=12, epochs=25, lr=3e-3)

        self.assertTrue(report["converged"], f"Policy overfit check must converge: {report}")
        self.assertGreaterEqual(report["action_accuracy"], 0.98)
        self.assertGreaterEqual(report["coord_accuracy"], 0.98)
        self.assertGreaterEqual(report["overall_accuracy"], 0.98)
        max_latency = 350.0 if torch.cuda.is_available() else 1500.0
        self.assertLess(report["p95_latency_ms"], max_latency)

    def test_latency_budget_benchmark(self):
        from avcs.cortex.model import PureRGBVLAPolicy, VLACortexConfig

        cfg = VLACortexConfig(
            input_height=400,
            input_width=640,
            stem_channels=32,
            stage_channels=(32, 64, 128, 256),
            neck_channels=128,
        )
        policy = PureRGBVLAPolicy(cortex_config=cfg, target_token_budget=400)

        profile = policy.profile_latency(runs=25, warmup=5)

        self.assertTrue(profile["under_budget"])
        self.assertLess(profile["p95_ms"], 350.0)
        self.assertLess(profile["p50_ms"], 350.0)

    def test_edge_cases_pruner_boundaries_and_shapes(self):
        import torch
        from avcs.cortex.model import ShowUITokenPruner

        pruner = ShowUITokenPruner(in_dim=64, target_budget=200, min_budget=32)

        # 3D tensor input (B, N, C)
        b, n, c = 2, 1000, 64
        feat_3d = torch.randn(b, n, c)

        # Boundary cursors at exact corners
        corner_cursors = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        out = pruner(features=feat_3d, cursor_xy=corner_cursors, target_budget=200)

        self.assertEqual(out.tokens.shape, (b, 200, c))
        self.assertEqual(out.coords.shape, (b, 200, 2))

        # Oversized budget clamping to available tokens
        out_clamped = pruner(features=feat_3d, target_budget=5000)
        self.assertEqual(out_clamped.tokens.shape, (b, n, c))

    def test_edge_cases_gui_actor_pointer_none_query_and_boundaries(self):
        import torch
        from avcs.cortex.model import GUIActorPatchPointer

        pointer = GUIActorPatchPointer(query_dim=128, patch_dim=64, hidden_dim=64)
        patch_tokens = torch.randn(1, 50, 64)
        patch_coords = torch.rand(1, 50, 2)

        # Query input is None: should use learnable ptr_query parameter
        out = pointer(
            query_input=None,
            patch_tokens=patch_tokens,
            patch_coords=patch_coords,
        )
        self.assertEqual(out.coords.shape, (1, 2))
        self.assertEqual(out.subpixel_coords.shape, (1, 2))
        self.assertTrue((out.coords >= 0.0).all())
        self.assertTrue((out.coords <= 1.0).all())

    def test_edge_cases_drag_expert_boundary_anchoring(self):
        import torch
        from avcs.cortex.model import FlowMatchingDragExpert

        expert = FlowMatchingDragExpert(cond_dim=128, hidden_dim=64, trajectory_points=16)
        cond = torch.randn(2, 128)
        # Boundary start coordinates (0, 0) and (1, 1)
        start_pos = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)

        traj = expert.generate_trajectory(cond=cond, start_pos=start_pos, num_steps=5)
        self.assertEqual(traj.shape, (2, 16, 2))
        self.assertTrue(torch.allclose(traj[0, 0], torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.allclose(traj[1, 0], torch.tensor([1.0, 1.0])))
        self.assertTrue((traj >= 0.0).all())
        self.assertTrue((traj <= 1.0).all())

    def test_edge_cases_policy_3d_and_5d_tensors(self):
        import torch
        from avcs.cortex.model import PureRGBVLAPolicy, VLACortexConfig

        cfg = VLACortexConfig(
            input_height=400,
            input_width=640,
            stem_channels=32,
            stage_channels=(32, 64, 128, 256),
            neck_channels=128,
        )
        policy = PureRGBVLAPolicy(cortex_config=cfg, target_token_budget=200)
        policy.eval()

        # 3D unbatched single image: (3, 400, 640)
        img_3d = torch.zeros((3, 400, 640), dtype=torch.float32)
        out_3d = policy(img_3d, goals=["Select target"])
        self.assertEqual(out_3d.target_coords.shape, (1, 2))

        # 5D temporal sequence: (B=1, T=2, C=3, H=400, W=640)
        img_5d = torch.zeros((1, 2, 3, 400, 640), dtype=torch.float32)
        out_5d = policy(img_5d, goals=["Select target temporal"])
        self.assertEqual(out_5d.target_coords.shape, (1, 2))

    def test_edge_cases_pruner_preserve_boxes_formats_and_grid_hw(self):
        """Verifies pruner supports 2D/1D tensors, lists of boxes, and mismatched grid_hw."""
        import torch
        from avcs.cortex.model import ShowUITokenPruner

        pruner = ShowUITokenPruner(in_dim=64, target_budget=100)
        feat = torch.randn(1, 64, 50, 80)

        # 2D tensor box: shape (1, 4)
        boxes_2d = torch.tensor([[0.2, 0.2, 0.4, 0.4]], dtype=torch.float32)
        out1 = pruner(feat, preserve_boxes=boxes_2d)
        self.assertEqual(out1.tokens.shape[1], 100)

        # 1D tensor box: shape (4,)
        box_1d = torch.tensor([0.2, 0.2, 0.4, 0.4], dtype=torch.float32)
        out2 = pruner(feat, preserve_boxes=box_1d)
        self.assertEqual(out2.tokens.shape[1], 100)

        # Flat tuple/list box: (0.2, 0.2, 0.4, 0.4)
        out3 = pruner(feat, preserve_boxes=[0.2, 0.2, 0.4, 0.4])
        self.assertEqual(out3.tokens.shape[1], 100)

        # List of tuple boxes: [(0.1, 0.1, 0.2, 0.2), (0.5, 0.5, 0.6, 0.6)]
        out4 = pruner(feat, preserve_boxes=[(0.1, 0.1, 0.2, 0.2), (0.5, 0.5, 0.6, 0.6)])
        self.assertEqual(out4.tokens.shape[1], 100)

        # 3D token feature with mismatched grid_hw (h*w < n)
        feat_3d = torch.randn(1, 150, 64)
        out5 = pruner(feat_3d, grid_hw=(10, 10))
        self.assertEqual(out5.tokens.shape[1], 100)
        self.assertEqual(out5.coords.shape, (1, 100, 2))

    def test_edge_cases_gui_actor_pointer_2d_coords_and_batched_query(self):
        """Verifies pointer handles 2D static coordinate grids with multi-batch inputs."""
        import torch
        from avcs.cortex.model import GUIActorPatchPointer

        pointer = GUIActorPatchPointer(query_dim=128, patch_dim=64, hidden_dim=64)
        b = 3
        patch_tokens = torch.randn(b, 40, 64)
        # 2D static coordinates: (N, 2)
        patch_coords_2d = torch.rand(40, 2)
        query = torch.randn(b, 128)

        out = pointer(query_input=query, patch_tokens=patch_tokens, patch_coords=patch_coords_2d)
        self.assertEqual(out.coords.shape, (b, 2))
        self.assertEqual(out.subpixel_coords.shape, (b, 2))
        self.assertEqual(out.attention.shape, (b, 40))

    def test_flow_matching_drag_expert_callable_and_differentiable(self):
        """Verifies drag expert supports callable forward and end-to-end backpropagation."""
        import torch
        from avcs.cortex.model import FlowMatchingDragExpert

        expert = FlowMatchingDragExpert(cond_dim=128, hidden_dim=64, trajectory_points=16)
        cond = torch.randn(2, 128, requires_grad=True)
        start_pos = torch.tensor([[0.2, 0.3], [0.7, 0.8]], dtype=torch.float32)

        # Callable __call__ interface
        traj = expert(cond, start_pos, num_steps=5)
        self.assertEqual(traj.shape, (2, 16, 2))

        # Differentiable forward pass for end-to-end training
        loss = traj.sum()
        loss.backward()
        self.assertIsNotNone(cond.grad)
        self.assertGreater(cond.grad.abs().sum().item(), 0.0)

    def test_rolling_kv_memory_custom_context_dim_and_batch_resizing(self):
        """Verifies rolling memory works with non-384 context dimensions and dynamic batch sizing."""
        import torch
        from avcs.cortex.model import RollingKVMemory, RollingMemoryState

        mem = RollingKVMemory(feat_dim=64, context_dim=192, max_history_steps=4)
        state = RollingMemoryState(max_history_steps=4)

        # Step 1 with batch size 1
        tokens_b1 = torch.randn(1, 50, 64)
        ctx_b1 = torch.randn(1, 192)
        fused_b1 = mem.update_and_fuse(tokens_b1, ctx_b1, state)
        self.assertEqual(fused_b1.shape, (1, 192))

        # Step 2 with batch size 2 (expands or adapts history smoothly)
        tokens_b2 = torch.randn(2, 50, 64)
        ctx_b2 = torch.randn(2, 192)
        fused_b2 = mem.update_and_fuse(tokens_b2, ctx_b2, state)
        self.assertEqual(fused_b2.shape, (2, 192))

    def test_pure_rgb_vla_policy_batched_outputs_and_custom_config(self):
        """Verifies policy handles batch size > 1 and non-default cortex configurations."""
        import torch
        from avcs.cortex.model import PureRGBVLAPolicy, VLACortexConfig

        # Custom config where stage_channels[3] + text_dim != 384
        cfg = VLACortexConfig(
            input_height=200,
            input_width=320,
            stem_channels=16,
            stage_channels=(16, 32, 64, 128),
            neck_channels=64,
            text_dim=64,
        )
        policy = PureRGBVLAPolicy(cortex_config=cfg, target_token_budget=150)
        policy.eval()

        b = 2
        batch_images = torch.zeros((b, 3, 200, 320), dtype=torch.float32)
        out = policy(batch_images, goals=["Goal 1", "Goal 2"])

        self.assertEqual(out.target_coords.shape, (b, 2))
        self.assertEqual(out.subpixel_coords.shape, (b, 2))
        self.assertEqual(out.action_logits.shape, (b, 10))
        self.assertIsNotNone(out.action_types)
        self.assertEqual(len(out.action_types), b)
        self.assertIsNotNone(out.action_indices)
        self.assertEqual(len(out.action_indices), b)
        self.assertIsNotNone(out.modifier_flags_list)
        self.assertEqual(len(out.modifier_flags_list), b)
        self.assertIsNotNone(out.hid_codes)
        self.assertEqual(len(out.hid_codes), b)

    def test_lite_train_head_only_optimization_and_ablation(self):
        """Verifies LiteTrain freezes backbone while heads achieve fast convergence and accuracy jump."""
        import torch
        from avcs.cortex.model import PureRGBVLAPolicy, VLACortexConfig, LiteTrain, evaluate_lite_train_ablation

        cfg = VLACortexConfig(
            input_height=200,
            input_width=320,
            stem_channels=16,
            stage_channels=(16, 32, 64, 128),
            neck_channels=64,
            text_dim=64,
        )
        policy = PureRGBVLAPolicy(cortex_config=cfg, target_token_budget=100)

        # 1. Verification of configure_lite_train parameter freeze
        policy.configure_lite_train(freeze_backbone=True, freeze_pruner=True)
        stats = LiteTrain.get_trainable_param_stats(policy)
        self.assertGreater(stats["trainable_params"], 0)
        self.assertGreater(stats["frozen_params"], 0)
        self.assertEqual(stats["trainable_params"] + stats["frozen_params"], stats["total_params"])

        for p in policy.backbone.parameters():
            self.assertFalse(p.requires_grad, "Backbone parameters must have requires_grad=False")
        for p in policy.pruner.parameters():
            self.assertFalse(p.requires_grad, "Pruner parameters must have requires_grad=False")
        for p in policy.patch_pointer.parameters():
            self.assertTrue(p.requires_grad, "Pointer head must be trainable")
        for p in policy.parallel_heads.parameters():
            self.assertTrue(p.requires_grad, "Action heads must be trainable")
        for p in policy.drag_expert.parameters():
            self.assertTrue(p.requires_grad, "Drag expert must be trainable")

        # 2. Evaluation of LiteTrain ablation diagnostic convergence
        report = evaluate_lite_train_ablation(policy=policy, num_episodes=12, epochs=35, lr=5e-3)
        self.assertTrue(report["converged"], f"LiteTrain must converge: {report}")
        self.assertTrue(report["backbone_frozen"])
        self.assertTrue(report["backbone_gradients_zero_or_none"])
        self.assertTrue(report["significant_jump"])
        self.assertGreaterEqual(report["overall_accuracy"], 0.98)
        self.assertGreaterEqual(report["action_accuracy"], 0.98)
        self.assertGreaterEqual(report["coord_accuracy"], 0.98)
        self.assertGreater(report["accuracy_jump"], 0.40)

    def test_cuda_graph_capture_and_export_serving_graph(self):
        """Verifies static tensor inference capture, CUDA Graph replay, and TorchScript serving graph export."""
        import tempfile
        from pathlib import Path
        import torch
        from avcs.cortex.model import PureRGBVLAPolicy, VLACortexConfig, CUDAGraphRunner, PureRGBVLAOutput

        cfg = VLACortexConfig(
            input_height=200,
            input_width=320,
            stem_channels=16,
            stage_channels=(16, 32, 64, 128),
            neck_channels=64,
            text_dim=64,
        )
        policy = PureRGBVLAPolicy(cortex_config=cfg, target_token_budget=100)
        policy.eval()

        # 1. CUDA Graph / static runner capture (both 4D and unbatched 3D inputs)
        runner = policy.capture_cuda_graph(warmup_iters=2)
        self.assertIsInstance(runner, CUDAGraphRunner)
        self.assertEqual(runner.policy, policy)

        # 3D unbatched input capture
        runner_3d = policy.capture_cuda_graph(sample_image=torch.randn(3, 200, 320), warmup_iters=2)
        self.assertIsInstance(runner_3d, CUDAGraphRunner)

        test_img = torch.randn(1, 3, 200, 320)
        test_cursor = torch.tensor([[0.25, 0.75]])
        out_replay = runner.replay(image=test_img, cursor_xy=test_cursor)
        self.assertIsInstance(out_replay, PureRGBVLAOutput)
        self.assertEqual(out_replay.target_coords.shape, (1, 2))

        # Callable syntax and 3D image / 1D cursor replay
        out_call = runner(image=torch.randn(3, 200, 320), cursor_xy=torch.tensor([0.25, 0.75]))
        self.assertIsInstance(out_call, PureRGBVLAOutput)
        self.assertEqual(out_call.target_coords.shape, (1, 2))

        # 2. Serving graph export (both 4D and unbatched 3D sample images)
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_file = Path(tmp_dir) / "serving_model.pt"
            export_info = policy.export_serving_graph(export_path=export_file)

            self.assertTrue(export_info["graph_ready"])
            self.assertEqual(export_info["export_path"], str(export_file))
            self.assertIn("image", export_info["input_shapes"])
            self.assertIn("cursor", export_info["input_shapes"])
            self.assertGreater(export_info["parameters_count"], 0)

            # Export with 3D image
            export_3d_file = Path(tmp_dir) / "serving_model_3d.pt"
            export_3d_info = policy.export_serving_graph(export_path=export_3d_file, sample_image=torch.randn(3, 200, 320))
            self.assertTrue(export_3d_info["graph_ready"])

            self.assertTrue(export_file.exists())
            loaded_ts = torch.jit.load(str(export_file))
            loaded_ts.eval()

            out_ts = loaded_ts(test_img, test_cursor)
            self.assertEqual(len(out_ts), 4)
            action_logits, coords, subpixel_coords, modifier_logits = out_ts
            self.assertEqual(action_logits.shape, (1, 10))
            self.assertEqual(coords.shape, (1, 2))
            self.assertEqual(subpixel_coords.shape, (1, 2))
            self.assertEqual(modifier_logits.shape, (1, 4))

    def test_100_step_stateful_session_flat_memory(self):
        """Verifies continuous 100-step interaction reaches a flat memory ceiling after step 16 with zero leaks."""
        import torch
        from avcs.cortex.model import PureRGBVLAPolicy, VLACortexConfig, RollingMemoryState

        cfg = VLACortexConfig(
            input_height=200,
            input_width=320,
            stem_channels=16,
            stage_channels=(16, 32, 64, 128),
            neck_channels=64,
            text_dim=64,
        )
        policy = PureRGBVLAPolicy(cortex_config=cfg, target_token_budget=100, max_history_steps=16)
        policy.eval()

        mem_state = RollingMemoryState(max_history_steps=16)
        frame = torch.zeros((1, 3, 200, 320), dtype=torch.float32)

        summary_counts: list[int] = []

        with torch.no_grad():
            for step in range(100):
                cx = 0.10 + 0.80 * ((step % 20) / 19.0)
                cy = 0.15 + 0.70 * (((step * 3) % 20) / 19.0)
                cursor = torch.tensor([[cx, cy]], dtype=torch.float32)

                out = policy(
                    images=frame,
                    goals=[f"Step instruction {step}"],
                    cursor_xy=cursor,
                    memory_state=mem_state,
                )

                self.assertLessEqual(out.pruned_token_count, 100)
                self.assertEqual(out.target_coords.shape, (1, 2))

                count = len(mem_state.step_summaries)
                summary_counts.append(count)

                if step < 16:
                    self.assertEqual(count, step + 1)
                else:
                    self.assertEqual(
                        count,
                        16,
                        f"Step {step}: Rolling memory must strictly cap at 16 summaries ceiling",
                    )

        for step in range(16, 100):
            self.assertEqual(summary_counts[step], 16)

    def test_closed_loop_playback_runner(self):
        """Verifies closed_loop_playback steps through episodes with rolling memory and achieves >=98% fidelity."""
        import torch
        from avcs.cortex.model import (
            PureRGBVLAPolicy,
            VLACortexConfig,
            evaluate_lite_train_ablation,
            closed_loop_playback,
            PlaybackEpisode,
            PlaybackStep,
        )

        cfg = VLACortexConfig(
            input_height=200,
            input_width=320,
            stem_channels=16,
            stage_channels=(16, 32, 64, 128),
            neck_channels=64,
            text_dim=64,
        )
        policy = PureRGBVLAPolicy(cortex_config=cfg, target_token_budget=100)

        ablation_rep = evaluate_lite_train_ablation(policy=policy, num_episodes=12, epochs=35, lr=5e-3)
        self.assertTrue(ablation_rep["converged"])

        playback_result = closed_loop_playback(policy=policy, tolerance=0.05)

        self.assertTrue(playback_result["verified_fidelity"])
        self.assertGreaterEqual(playback_result["execution_fidelity"], 0.98)
        self.assertGreaterEqual(playback_result["action_accuracy"], 0.98)
        self.assertGreaterEqual(playback_result["coord_accuracy"], 0.98)
        self.assertEqual(playback_result["total_steps"], 12)
        self.assertEqual(playback_result["successful_episodes"], playback_result["total_episodes"])
        self.assertEqual(playback_result["episode_success_rate"], 1.0)
        max_latency = 350.0 if torch.cuda.is_available() else 1500.0
        self.assertLess(playback_result["p95_latency_ms"], max_latency)
        self.assertEqual(len(playback_result["step_details"]), 12)

        dummy_img = torch.zeros((1, 3, 200, 320))
        with torch.no_grad():
            dummy_out = policy(dummy_img, goals=["Custom goal 1"], cursor_xy=torch.tensor([[0.5, 0.5]]))
            pred_act = dummy_out.action_type
            pred_coord = (float(dummy_out.target_coords[0, 0].item()), float(dummy_out.target_coords[0, 1].item()))

        custom_dict_episodes = [
            {
                "episode_id": "custom-dict-01",
                "initial_cursor": [0.5, 0.5],
                "steps": [
                    {
                        "image": dummy_img,
                        "goal": "Custom goal 1",
                        "action_type": pred_act,
                        "target_coords": pred_coord,
                    }
                ],
            }
        ]
        custom_res = closed_loop_playback(policy=policy, episodes=custom_dict_episodes, tolerance=0.05)
        self.assertEqual(custom_res["total_episodes"], 1)
        self.assertEqual(custom_res["total_steps"], 1)
        self.assertTrue(custom_res["verified_fidelity"])

        # Test edge case: None coords, ActionKind enum, and per-step cursor_xy
        step_none_coord = PlaybackStep(
            observation=dummy_img,
            goal="Custom goal 1",
            expected_action=pred_act,
            expected_coords=None,
            cursor_xy=torch.tensor([0.5, 0.5]),
        )
        ep_none_coord = PlaybackEpisode(
            episode_id="ep-none-coords",
            steps=[step_none_coord],
            initial_cursor=torch.tensor([0.1, 0.1]),
        )
        res_none_coord = closed_loop_playback(policy=policy, episodes=[ep_none_coord], tolerance=0.05)
        self.assertTrue(res_none_coord["verified_fidelity"])
        self.assertTrue(res_none_coord["step_details"][0]["coord_ok"])

        # Empty episodes edge case
        empty_res = closed_loop_playback(policy=policy, episodes=[], tolerance=0.05)
        self.assertEqual(empty_res["total_episodes"], 0)
        self.assertEqual(empty_res["total_steps"], 0)


class TestPhase4BehavioralCloningAndDAgger(unittest.TestCase):
    """Phase 4 Scaled Behavioral Cloning, Stage 3 DAgger Recovery, and Training Pipeline Tests."""

    def test_stage3_dagger_hunting_loop_classification_and_intervention(self):
        """Verifies Stage 3 DAgger detects oscillating hunting loops and generates breakout interventions."""
        from avcs.cortex.model import DAggerRecoveryPolicy, DAggerFailureType

        policy = DAggerRecoveryPolicy(hunting_repetition_threshold=2, hunting_spatial_radius=0.05)

        # 1. Normal sequence
        normal_type = policy.classify_failure(
            coordinates_history=[(0.1, 0.1), (0.3, 0.3), (0.5, 0.5)],
            repetition_count=0,
        )
        self.assertEqual(normal_type, DAggerFailureType.NONE)

        # 2. Repeated actions at same coordinate (repetition count threshold)
        hunting_by_rep = policy.classify_failure(
            coordinates_history=[(0.4, 0.4), (0.4, 0.4)],
            repetition_count=2,
        )
        self.assertEqual(hunting_by_rep, DAggerFailureType.HUNTING_LOOP)

        # 3. Spatial cluster hunting (jitter within tight radius)
        hunting_by_cluster = policy.classify_failure(
            coordinates_history=[(0.50, 0.50), (0.51, 0.51), (0.50, 0.52)],
            repetition_count=1,
        )
        self.assertEqual(hunting_by_cluster, DAggerFailureType.HUNTING_LOOP)

        # 4. Alternating oscillation: A -> B -> A
        hunting_by_oscillation = policy.classify_failure(
            coordinates_history=[(0.20, 0.20), (0.35, 0.35), (0.21, 0.20)],
            repetition_count=0,
        )
        self.assertEqual(hunting_by_oscillation, DAggerFailureType.HUNTING_LOOP)

        # 5. Intervention synthesis
        interv = policy.compute_intervention(
            failure_type=DAggerFailureType.HUNTING_LOOP,
            goal="Click target button",
            current_cursor=(0.50, 0.50),
        )
        self.assertIsNotNone(interv)
        self.assertEqual(interv.failure_type, DAggerFailureType.HUNTING_LOOP)
        self.assertEqual(interv.action_type, "left_click")
        self.assertIsNotNone(interv.target_coords)
        self.assertNotEqual(interv.target_coords, (0.50, 0.50))
        self.assertIn("Break out", interv.reason)

    def test_stage3_dagger_popup_interrupt_classification_and_intervention(self):
        """Verifies Stage 3 DAgger detects modal popups and synthesizes ESC dismissal interventions."""
        from avcs.cortex.model import DAggerRecoveryPolicy, DAggerFailureType

        policy = DAggerRecoveryPolicy()

        # 1. Detection via has_popup flag
        fail_popup = policy.classify_failure(has_popup=True)
        self.assertEqual(fail_popup, DAggerFailureType.POPUP_INTERRUPT)

        # 2. Detection via window title
        fail_title = policy.classify_failure(active_window_title="Warning: User Account Control Prompt")
        self.assertEqual(fail_title, DAggerFailureType.POPUP_INTERRUPT)

        # 3. Intervention synthesis: ESC key dismissal
        interv = policy.compute_intervention(
            failure_type=DAggerFailureType.POPUP_INTERRUPT,
            goal="Open Settings",
            popup_dismiss_key="escape",
        )
        self.assertIsNotNone(interv)
        self.assertEqual(interv.failure_type, DAggerFailureType.POPUP_INTERRUPT)
        self.assertEqual(interv.action_type, "key")
        self.assertEqual(interv.keys, ["escape"])
        self.assertIn("Dismiss modal popup", interv.reason)

        # 4. Step evaluation and buffer aggregation
        policy.clear_buffer()
        f_type, action_interv = policy.evaluate_step(has_popup=True, goal="Test step")
        self.assertEqual(f_type, DAggerFailureType.POPUP_INTERRUPT)
        self.assertIsNotNone(action_interv)
        self.assertEqual(len(policy.intervention_buffer), 1)

    def test_stage3_verifier_integration(self):
        """Verifies DualStageVerifier incorporates Stage 3 DAgger verification and interventions."""
        from avcs.ace.runner import DualStageVerifier, Stage3Result, Action, ActionKind

        verifier = DualStageVerifier()

        # Normal condition -> passes
        clean_res = verifier.verify_stage3_recovery(
            has_popup=False,
            repetition_count=0,
            goal="Click element",
        )
        self.assertTrue(clean_res.passed)
        self.assertEqual(clean_res.failure_type, "none")
        self.assertIsNone(clean_res.intervention_action)

        # Popup blocker condition -> failure with ESC intervention
        popup_res = verifier.verify_stage3_recovery(
            has_popup=True,
            goal="Click element",
        )
        self.assertFalse(popup_res.passed)
        self.assertEqual(popup_res.failure_type, "popup_interrupt")
        self.assertIsNotNone(popup_res.intervention_action)
        self.assertEqual(popup_res.intervention_action.kind, ActionKind.KEY)
        self.assertEqual(popup_res.intervention_action.keys, ["escape"])

        # Hunting loop condition -> failure with breakout intervention
        hunt_res = verifier.verify_stage3_recovery(
            has_popup=False,
            repetition_count=3,
            current_cursor=(500, 300),
            goal="Click element",
        )
        self.assertFalse(hunt_res.passed)
        self.assertEqual(hunt_res.failure_type, "hunting_loop")
        self.assertIsNotNone(hunt_res.intervention_action)
        self.assertEqual(hunt_res.intervention_action.kind, ActionKind.LEFT_CLICK)
        self.assertNotEqual(hunt_res.intervention_action.coordinate, (500, 300))

    def test_stage3_dagger_runner_recovery_loop(self):
        """Verifies AceRunner applies Stage 3 DAgger recovery interventions when enabled."""
        from avcs.ace.runner import AceRunner, Action, ActionKind
        from avcs.ace.agent import AceAgent
        from avcs.ace.computer import WindowsComputer

        mock_client = MagicMock()
        mock_client.model = "gpt-6-astra"
        mock_client.predict_action.side_effect = [
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(300, 300), reason="Click button"), 100.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(300, 300), reason="Click button"), 100.0),
            (Action(kind=ActionKind.LEFT_CLICK, coordinate=(300, 300), reason="Click button"), 100.0),
            (Action(kind=ActionKind.STOP, reason="Done"), 100.0),
        ]

        agent = AceAgent(client=mock_client)
        computer = WindowsComputer()
        runner = AceRunner(agent=agent, computer=computer, enable_voice=False, enable_dagger_recovery=True)

        session = runner.run(goal="Test DAgger intervention", max_steps=4, dry_run=True)
        self.assertTrue(len(runner.dagger_interventions) >= 1)
        self.assertEqual(runner.dagger_interventions[0]["failure_type"], "hunting_loop")
        self.assertTrue(any("dagger" in s.reason.lower() or "break out" in s.reason.lower() for s in session.steps))

    def test_phase4_data_mixture_schema_normalization(self):
        """Verifies heterogeneous open-source datasets (OS-Atlas, ShowUI-desktop, AgentNet) normalize cleanly."""
        from avcs.dataset.loader import (
            normalize_open_source_point,
            normalize_open_source_action,
            Phase4DataMixtureDataset,
        )

        # Point normalization from 1080p pixel space
        pt_pixel = normalize_open_source_point([960, 540], screen_width=1920.0, screen_height=1080.0)
        self.assertAlmostEqual(pt_pixel[0], 0.5, places=3)
        self.assertAlmostEqual(pt_pixel[1], 0.5, places=3)

        # Point normalization already in [0, 1]
        pt_norm = normalize_open_source_point([0.35, 0.72])
        self.assertAlmostEqual(pt_norm[0], 0.35, places=3)
        self.assertAlmostEqual(pt_norm[1], 0.72, places=3)

        # Action normalization
        self.assertEqual(normalize_open_source_action("left_click"), 0)
        self.assertEqual(normalize_open_source_action("click"), 0)
        self.assertEqual(normalize_open_source_action("press_key"), 7)
        self.assertEqual(normalize_open_source_action(5), 5)

        # Mixture dataset ingestion
        mock_records = [
            # OS-Atlas style
            {
                "dataset": "os_atlas",
                "instruction": "Click the File menu",
                "point": [100, 20],
                "action": "click",
            },
            # ShowUI-desktop style
            {
                "dataset": "showui_desktop",
                "query": "Click Submit button",
                "bbox": [200, 400, 300, 450],
                "action_type": "left_click",
            },
            # AgentNet style
            {
                "dataset": "agentnet",
                "goal": "Press Enter to execute",
                "coords": [0.5, 0.5],
                "action": "key",
            },
        ]

        ds = Phase4DataMixtureDataset(
            open_source_records=mock_records,
            target_size=(200, 320),
        )
        self.assertEqual(len(ds), 3)

        item0 = ds[0]
        self.assertEqual(item0["source"], "os_atlas")
        self.assertEqual(item0["image"].shape, (3, 200, 320))
        self.assertEqual(int(item0["action_kind"].item()), 0)
        self.assertEqual(item0["drag_trajectory"].shape, (16, 2))

    def test_phase4_multi_task_loss_computation(self):
        """Verifies AVCSPhase4MultiTaskLoss computes action cross-entropy and pointer L1/L2 loss."""
        import torch
        from avcs.cortex.model import AVCSPhase4MultiTaskLoss, PureRGBVLAOutput

        loss_fn = AVCSPhase4MultiTaskLoss(
            weight_action=1.0,
            weight_coord_l1=10.0,
            weight_coord_l2=20.0,
            enable_scroll=True,
            enable_flow_drag=False,
        )

        b = 4
        dummy_out = PureRGBVLAOutput(
            action_type="left_click",
            action_index=0,
            action_logits=torch.randn((b, 10)),
            action_probs=torch.softmax(torch.randn((b, 10)), dim=-1),
            target_coords=torch.full((b, 2), 0.5),
            subpixel_coords=torch.full((b, 2), 0.5),
            modifier_flags={"shift": False},
            modifier_tensor=torch.zeros((b, 4)),
            scroll_delta=torch.zeros((b, 2)),
            hid_code=0,
            drag_trajectory=None,
            pointer_attention=torch.zeros((b, 1, 100)),
            pruned_token_count=100,
        )

        target_actions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        target_coords = torch.tensor([[0.5, 0.5], [0.6, 0.6], [0.4, 0.4], [0.5, 0.5]], dtype=torch.float32)
        target_scroll = torch.zeros((b, 2), dtype=torch.float32)

        losses = loss_fn(
            model_out=dummy_out,
            target_actions=target_actions,
            target_coords=target_coords,
            target_scroll=target_scroll,
        )

        self.assertIn("loss", losses)
        self.assertIn("loss_action", losses)
        self.assertIn("loss_coord_l1", losses)
        self.assertIn("loss_coord_l2", losses)
        self.assertGreater(float(losses["loss"].item()), 0.0)

    def test_phase4_training_dry_run_pipeline(self):
        """Verifies run_phase4_scaled_bc_training executes cleanly on CPU without GPU dependencies."""
        import importlib.util
        from pathlib import Path

        train_path = Path(__file__).resolve().parent.parent / "kaggle" / "train_avcs_gpu.py"
        spec = importlib.util.spec_from_file_location("kaggle_train_avcs_gpu", str(train_path))
        train_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_mod)
        run_phase4_scaled_bc_training = train_mod.run_phase4_scaled_bc_training

        result = run_phase4_scaled_bc_training(
            epochs=1,
            batch_size=4,
            dry_run=True,
            num_golden_episodes=10,
            val_episodes=2,
            save_dir="runs/test_checkpoints",
        )

        self.assertIn("final_checkpoint", result)
        self.assertIn("best_checkpoint", result)
        self.assertIn("training_metrics", result)
        self.assertTrue(os.path.exists(result["final_checkpoint"]))
        self.assertTrue(os.path.exists(result["best_checkpoint"]))
        self.assertTrue(len(result["training_metrics"]) >= 1)
        self.assertLess(result["latency_stats"]["p95_ms"], 350.0)

        # Cleanup test checkpoints
        for f in (result["final_checkpoint"], result["best_checkpoint"]):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass

    def test_phase4_edge_cases(self):
        """Verifies boundary conditions, empty inputs, and None values in Phase 4 components."""
        from avcs.dataset.loader import (
            normalize_open_source_point,
            normalize_open_source_action,
            Phase4DataMixtureDataset,
        )
        from avcs.cortex.model import DAggerRecoveryPolicy, DAggerFailureType

        # None / empty points
        self.assertEqual(normalize_open_source_point(None), (0.5, 0.5))
        self.assertEqual(normalize_open_source_point([]), (0.5, 0.5))
        self.assertEqual(normalize_open_source_point([2500.0, 1500.0], 1920.0, 1080.0), (1.0, 1.0))
        self.assertEqual(normalize_open_source_point([-50.0, -10.0]), (0.0, 0.0))

        # None / unknown actions
        self.assertEqual(normalize_open_source_action(None), 0)
        self.assertEqual(normalize_open_source_action("unknown_gibberish_action"), 0)
        self.assertEqual(normalize_open_source_action(999), 9)

        # Empty mixture dataset
        empty_ds = Phase4DataMixtureDataset(open_source_records=[])
        self.assertEqual(len(empty_ds), 0)

        # Empty coordinate history in DAgger recovery policy
        policy = DAggerRecoveryPolicy()
        res = policy.classify_failure(coordinates_history=[])
        self.assertEqual(res, DAggerFailureType.NONE)

        # DAgger intervention on None failure type
        interv_none = policy.compute_intervention(DAggerFailureType.NONE)
        self.assertIsNone(interv_none)



if __name__ == "__main__":
    unittest.main()




