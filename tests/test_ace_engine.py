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

from avcs.ace.actions import Action, ActionAdapter, ActionKind
from avcs.ace.agent import AceAgent
from avcs.ace.client import CustomLLMClient
from avcs.ace.computer import (
    INPUT,
    ExecutionSafetyFirewall,
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
        os.environ["AVCS_ACE_ENDPOINT"] = "https://custom-gateway.internal/v1/chat/completions"
        os.environ["AVCS_ACE_API_KEY"] = "key-from-env"
        os.environ["AVCS_ACE_MODEL"] = "fable-5.5"
        os.environ["AVCS_ACE_EXTRA_HEADERS"] = json.dumps({"X-Organization": "ResearchLab"})

        client = CustomLLMClient()
        self.assertEqual(client.endpoint, "https://custom-gateway.internal/v1/chat/completions")
        self.assertEqual(client.api_key, "key-from-env")
        self.assertEqual(client.model, "fable-5.5")
        self.assertEqual(client.extra_headers.get("X-Organization"), "ResearchLab")

    def test_protocol_auto_detection(self):
        # Anthropic
        c_anthropic = CustomLLMClient(endpoint="https://api.anthropic.com/v1/messages")
        self.assertEqual(c_anthropic.protocol, "anthropic")

        # Gemini
        c_gemini = CustomLLMClient(endpoint="https://generativelanguage.googleapis.com/v1beta/models")
        self.assertEqual(c_gemini.protocol, "gemini")

        # OpenAI compatible
        c_openai = CustomLLMClient(endpoint="https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(c_openai.protocol, "openai")

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
        client = CustomLLMClient(endpoint="https://api.anthropic.com/v1/messages", model="fable-5.5")
        payload = client._build_payload(
            b64_str="BBBB",
            mime="image/png",
            data_uri="data:image/png;base64,BBBB",
            goal="Save document",
            step_index=1,
        )
        self.assertEqual(payload["model"], "fable-5.5")
        self.assertIn("system", payload)
        self.assertEqual(payload["messages"][0]["content"][0]["type"], "image")
        self.assertEqual(payload["messages"][0]["content"][0]["source"]["data"], "BBBB")

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
        # Groq key auto-detection
        os.environ["GROQ_API_KEY"] = "gsk_test12345"
        c_groq = CustomLLMClient()
        self.assertIn("api.groq.com", c_groq.endpoint)
        self.assertEqual(c_groq.protocol, "openai")
        del os.environ["GROQ_API_KEY"]

        # Anthropic key auto-detection
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test12345"
        c_ant = CustomLLMClient()
        self.assertIn("api.anthropic.com", c_ant.endpoint)
        self.assertEqual(c_ant.protocol, "anthropic")
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

        # Payload uses max_tokens across protocols
        c_anthropic = CustomLLMClient(endpoint="https://api.anthropic.com/v1/messages", max_tokens=1024)
        payload_ant = c_anthropic._build_payload(b64_str="X", mime="image/webp", data_uri="data:", goal="test")
        self.assertEqual(payload_ant["max_tokens"], 1024)

        c_gemini = CustomLLMClient(endpoint="https://generativelanguage.googleapis.com/v1beta/models", max_tokens=1024)
        payload_gem = c_gemini._build_payload(b64_str="X", mime="image/webp", data_uri="data:", goal="test")
        self.assertEqual(payload_gem["generationConfig"]["maxOutputTokens"], 1024)

        c_openai = CustomLLMClient(max_tokens=1024)
        payload_oai = c_openai._build_payload(b64_str="X", mime="image/webp", data_uri="data:", goal="test")
        self.assertEqual(payload_oai["max_tokens"], 1024)

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
            session = runner.run(goal="Start menu open karo aur Notepad search karo", max_steps=5, dry_run=True)

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

        session = runner.run(goal="Start menu open karo aur Notepad search karo", max_steps=5, dry_run=True)
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
             patch.object(runner.user32, "GetClassNameW", side_effect=lambda hwnd, buf, size: setattr(buf, "value", "ConsoleWindowClass")):
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

    def test_semantic_fallback_hotkey_precedence_and_hinglish(self):
        # Hotkey in reason must NOT be clobbered by substring single key checks
        a1 = Action.from_dict({"kind": "key", "reason": "Press ctrl+shift+esc to open task manager"})
        self.assertEqual(a1.keys, ["ctrl", "shift", "esc"])

        # Type single word without trailing punctuation
        a2 = Action.from_dict({"kind": "type", "reason": "Type Notepad"})
        self.assertEqual(a2.kind, ActionKind.TYPE)
        self.assertEqual(a2.text, "Notepad")

        # Hinglish extraction
        a3 = Action.from_dict({"kind": "type", "reason": "Notepad type karo"})
        self.assertEqual(a3.kind, ActionKind.TYPE)
        self.assertEqual(a3.text, "Notepad")

        a4 = Action.from_dict({"kind": "type", "reason": "Type karo calculator"})
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

        # Unicode Hindi text in grammar
        act2 = ActionAdapter.adapt("ACT:TYPE नमस्ते दुनिया")
        self.assertEqual(act2.kind, ActionKind.TYPE)
        self.assertEqual(act2.text, "नमस्ते दुनिया")

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

    def test_multilingual_decomposition_hinglish_start_notepad(self):
        goal = "Start menu open karo aur Notepad search karo"
        subgoals = self.planner.decompose(goal)
        self.assertEqual(len(subgoals), 2)

        self.assertEqual(subgoals[0].index, 1)
        self.assertIn("Start Menu", subgoals[0].description)
        self.assertIn("Start menu opened", subgoals[0].expected_ui_state)

        self.assertEqual(subgoals[1].index, 2)
        self.assertIn("Notepad", subgoals[1].description)
        self.assertIn("Notepad application window", subgoals[1].expected_ui_state)

    def test_multilingual_decomposition_hinglish_chrome_search(self):
        goal = "Chrome kholo aur Google par OpenAI search karo"
        subgoals = self.planner.decompose(goal)
        self.assertEqual(len(subgoals), 2)

        self.assertEqual(subgoals[0].index, 1)
        self.assertIn("Google Chrome", subgoals[0].description)

        self.assertEqual(subgoals[1].index, 2)
        self.assertIn("OpenAI", subgoals[1].description)
        self.assertIn("Google", subgoals[1].description)

    def test_multilingual_decomposition_hindi_devanagari(self):
        goal = "क्रोम खोलो और गूगल पर सर्च करो"
        subgoals = self.planner.decompose(goal)
        self.assertEqual(len(subgoals), 2)
        self.assertEqual(subgoals[0].index, 1)
        self.assertIn("Google Chrome", subgoals[0].description)
        self.assertEqual(subgoals[1].index, 2)
        self.assertIn("Google", subgoals[1].description)
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
        # Strategic alternative route uses Win+R shortcut
        self.assertTrue(any("Win+R" in s.description or "Run" in s.description for s in replanned))

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
        # Strategic alternative route presses Escape to dismiss modal and retries
        self.assertTrue(any("Escape" in s.description for s in replanned))

    def test_dynamic_replanning_console_firewall_hazard(self):
        failed_sg = SubGoal(index=1, description="Search for Notepad query", expected_ui_state="Search results")
        replanned = self.planner.replan(
            goal="Search query",
            failed_subgoal=failed_sg,
            failure_reason="Execution Safety Firewall: Active console detected. Rejecting ungrounded typing.",
        )
        self.assertGreaterEqual(len(replanned), 1)
        # Strategic alternative route focuses input via keyboard shortcut
        self.assertTrue(any("keyboard" in s.description.lower() or "ctrl" in s.description.lower() for s in replanned))

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
            goal="Start menu open karo aur Notepad search karo",
            max_steps=10,
            dry_run=True,
        )

        self.assertTrue(session.is_complete)
        self.assertEqual(session.step_count, 4)
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
                    goal="Start menu open karo aur Notepad search karo",
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
        goal = "Chrome kholo aur fir YouTube open karo karke music search karo aur fir play karo"
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

    def test_multilingual_decomposition_kholkar_and_devanagari(self):
        planner = CognitivePlanner()
        # Hinglish with kholkar
        s1 = planner.decompose("Chrome kholkar Google par OpenAI search karo")
        self.assertEqual(len(s1), 2)
        self.assertIn("Google Chrome", s1[0].description)
        self.assertIn("OpenAI", s1[1].description)
        self.assertIn("Google", s1[1].description)

        # Devanagari Start and Notepad -> outputs clean English directives
        s2 = planner.decompose("स्टार्ट मेन्यू खोलो और नोटपैड सर्च करो")
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


if __name__ == "__main__":
    unittest.main()


