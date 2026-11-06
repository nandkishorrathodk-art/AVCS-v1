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
import sys
import unittest
from unittest.mock import MagicMock, patch

from avcs.ace.actions import Action, ActionKind
from avcs.ace.agent import AceAgent
from avcs.ace.client import CustomLLMClient
from avcs.ace.computer import INPUT, Observation, WindowsComputer
from avcs.ace.runner import AceRunner, SafetyBreakoutError
from avcs.ace.session import AceSession


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
        streamer.start()
        try:
            self.assertTrue(streamer.is_running)
            time.sleep(0.1)
            t0 = time.time()
            obs = streamer.get_latest_observation()
            read_latency_ms = (time.time() - t0) * 1000.0
            self.assertIsNotNone(obs)
            self.assertLess(read_latency_ms, 5.0)  # Instant in-memory pointer read
            self.assertEqual(obs.width, 1280)
            self.assertEqual(obs.height, 800)
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


if __name__ == "__main__":
    unittest.main()

