"""AVCS System 2 Reasoning Providers.

Implements the pluggable reasoning interface between high-level task intent and
AVCS System 1 visual observations.

Providers:
1. ReasoningProvider: Abstract base contract.
2. MockReasoningProvider: Deterministic, rule-driven & scriptable provider for automated tests,
   error recovery verification, and adversarial input testing.
3. LLMReasoningProvider: Configurable provider interfacing with external pretrained LLMs
   via standard environment variables without hardcoded credentials.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlsplit

from avcs.reasoning.contracts import (
    ActionIntent,
    ActionOp,
    System2Decision,
)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _is_local_endpoint(endpoint: Optional[str]) -> bool:
    """Return True only when ``endpoint`` targets a local loopback host."""
    raw = (endpoint or "").strip()
    if not raw:
        return False
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    return parsed.hostname.lower() in _LOOPBACK_HOSTS


# Server-sent-event framing used by OpenAI-compatible streaming endpoints.
_SSE_DATA_PREFIX = "data: "
_SSE_DONE = "[DONE]"


class ReasoningProvider(ABC):
    """Abstract interface for System 2 high-level task planning and reasoning."""

    @abstractmethod
    def plan_step(
        self,
        task: str,
        observation: Dict[str, Any],
        history: List[Dict[str, Any]],
        step_index: int = 1,
        system_prompt: Optional[str] = None,
    ) -> System2Decision:
        """Evaluates current observation and history to propose the next machine-readable intent.

        ``system_prompt`` lets a caller swap the role instructions (a live
        conversation wants a different voice than a silent planner). ``None``
        keeps the provider's default.
        """
        raise NotImplementedError

    def stream_step(
        self,
        task: str,
        observation: Dict[str, Any],
        history: List[Dict[str, Any]],
        step_index: int = 1,
        system_prompt: Optional[str] = None,
    ):
        """Yield the raw reply in fragments as the model produces it.

        The default implementation is not streaming: it calls ``plan_step`` and
        yields the whole reply as one fragment. Providers that speak SSE override
        this so the assistant can start talking before the reply is finished.
        """
        decision = self.plan_step(task, observation, history, step_index, system_prompt)
        yield (decision.clarification_prompt or decision.thought or "")

    def reset(self) -> None:
        """Resets any internal session state."""
        pass


class MockReasoningProvider(ReasoningProvider):
    """Deterministic, scriptable reasoning provider for automated unit tests and benchmarks.

    Supports:
    - Pre-scripted decision queues for deterministic replays.
    - Rule-based dynamic reasoning based on observed target presence, ambiguity, and stability.
    - Fault injection: malformed intents, coordinate injection attempts, stale frame IDs, provider timeouts.
    """

    def __init__(
        self,
        scripted_decisions: Optional[List[System2Decision]] = None,
        simulated_latency_ms: float = 5.0,
        inject_malformed: bool = False,
        inject_coordinates: bool = False,
        inject_stale_frame: bool = False,
        inject_provider_error: bool = False,
    ) -> None:
        self.scripted_decisions: List[System2Decision] = list(scripted_decisions or [])
        self.decision_index: int = 0
        self.simulated_latency_ms = simulated_latency_ms
        self.inject_malformed = inject_malformed
        self.inject_coordinates = inject_coordinates
        self.inject_stale_frame = inject_stale_frame
        self.inject_provider_error = inject_provider_error

    def reset(self) -> None:
        self.decision_index = 0

    def plan_step(
        self,
        task: str,
        observation: Dict[str, Any],
        history: List[Dict[str, Any]],
        step_index: int = 1,
        system_prompt: Optional[str] = None,
    ) -> System2Decision:
        if self.simulated_latency_ms > 0:
            time.sleep(self.simulated_latency_ms / 1000.0)

        if self.inject_provider_error:
            raise RuntimeError("Injected LLM Provider Gateway Error (HTTP 504 Gateway Timeout)")

        # 1. Use pre-scripted decision if available
        if self.decision_index < len(self.scripted_decisions):
            decision = self.scripted_decisions[self.decision_index]
            self.decision_index += 1
            return decision

        # 2. Adversarial fault injections
        curr_frame_seq = observation.get("frame_sequence", 1)
        target_seq = (curr_frame_seq - 5) if self.inject_stale_frame else curr_frame_seq

        if self.inject_malformed:
            raw_intent = {
                "step_index": step_index,
                "op": "completely_unsupported_nonexistent_op",
                "target_description": None,
                "frame_sequence": target_seq,
            }
            intent, _ = ActionIntent.from_dict(raw_intent, default_step=step_index)
            return System2Decision(
                thought="Testing malformed intent rejection.",
                task_status="in_progress",
                next_intent=intent,
            )

        if self.inject_coordinates:
            raw_intent = {
                "step_index": step_index,
                "op": "click",
                "target_description": "Search button",
                "x": 450.5,
                "y": 120.0,
                "frame_sequence": curr_frame_seq,
                "rationale": "Attempting unauthorized coordinate injection.",
            }
            intent, had_injection = ActionIntent.from_dict(raw_intent, default_step=step_index)
            # If from_dict stripped it, re-inject for validation testing
            object.__setattr__(intent, "parameters", {"injected_x": 450.5, "injected_y": 120.0})
            return System2Decision(
                thought="Attempting coordinate injection to test System 1 validator.",
                task_status="in_progress",
                next_intent=intent,
            )

        if self.inject_stale_frame:
            intent = ActionIntent.create(
                step_index=step_index,
                op=ActionOp.CLICK,
                target_description="Settings icon",
                rationale="Testing stale frame rejection.",
                expected_outcome="Target click rejected due to stale frame reference",
                frame_sequence=max(0, curr_frame_seq - 5),
            )
            return System2Decision(
                thought="Proposing action on outdated frame sequence.",
                task_status="in_progress",
                next_intent=intent,
            )

        # 3. Dynamic Rule-Based Reasoning Engine
        target_presence = observation.get("target_presence", "ABSENT")
        presence_conf = observation.get("presence_confidence", 0.0)
        is_ambiguous = observation.get("is_ambiguous", False)
        temporal = observation.get("temporal_change") or {}
        stability = temporal.get("stability_state", "stable")
        uncertainty_flags = observation.get("uncertainty_flags", [])

        # A. If sudden modal/overlay or unstable frame detected -> wait for stability
        if stability in ("surprise", "busy") or "frame_motion_unstable" in uncertainty_flags:
            intent = ActionIntent.create(
                step_index=step_index,
                op=ActionOp.WAIT,
                rationale=f"Screen is temporally unstable ({stability}); waiting for visual settling.",
                expected_outcome="Screen reaches stable visual state",
                frame_sequence=curr_frame_seq,
            )
            return System2Decision(
                thought=f"Detected screen transition ({stability}). Pausing motor planning until stable.",
                task_status="waiting_for_observation",
                next_intent=intent,
            )

        # B. If target was explicitly requested but is ABSENT -> ask user or stop
        active_q = observation.get("active_query")
        if active_q and target_presence == "ABSENT":
            intent = ActionIntent.create(
                step_index=step_index,
                op=ActionOp.STOP,
                rationale=f"Requested target '{active_q}' is verified absent on screen (confidence {presence_conf:.2f}).",
                expected_outcome="Session safely halted without unauthorized clicks",
                frame_sequence=curr_frame_seq,
            )
            return System2Decision(
                thought=f"Target '{active_q}' could not be located. Stopping to avoid blind actions.",
                task_status="aborted",
                next_intent=intent,
            )

        # C. If target is AMBIGUOUS -> request clarification or inspect candidate
        if is_ambiguous or "multimodal_target_ambiguity" in uncertainty_flags:
            intent = ActionIntent.create(
                step_index=step_index,
                op=ActionOp.ASK_USER,
                target_description=active_q,
                rationale=f"Multiple conflicting instances of '{active_q}' found. Requesting clarification.",
                expected_outcome="User provides disambiguating context",
                frame_sequence=curr_frame_seq,
            )
            return System2Decision(
                thought=f"Ambiguous target peaks detected for '{active_q}'. Halting for clarification.",
                task_status="needs_clarification",
                next_intent=intent,
                clarification_prompt=f"Which '{active_q}' did you want to interact with?",
            )

        # D. If target is verified PRESENT with action authorization -> propose click
        if target_presence == "PRESENT" and observation.get("action_authorized", False):
            intent = ActionIntent.create(
                step_index=step_index,
                op=ActionOp.CLICK,
                target_description=active_q,
                rationale=f"Neural grounding verified target '{active_q}' with confidence {presence_conf:.2f}.",
                expected_outcome=f"Target '{active_q}' activates and advances task state",
                frame_sequence=curr_frame_seq,
            )
            return System2Decision(
                thought=f"Target '{active_q}' is grounded and authorized. Proposing click.",
                task_status="in_progress",
                next_intent=intent,
            )

        # E. If no active query yet, or looking for initial target -> request observation
        target_name = self._infer_target_from_task(task, history)
        if target_name:
            intent = ActionIntent.create(
                step_index=step_index,
                op=ActionOp.OBSERVE_TARGET,
                target_description=target_name,
                rationale=f"Task '{task}' requires locating '{target_name}'.",
                expected_outcome=f"Visual grounding on '{target_name}'",
                frame_sequence=curr_frame_seq,
            )
            return System2Decision(
                thought=f"To accomplish '{task}', requesting visual grounding for '{target_name}'.",
                task_status="in_progress",
                next_intent=intent,
            )

        # F. Default completion
        intent = ActionIntent.create(
            step_index=step_index,
            op=ActionOp.FINISH,
            rationale="All planned subgoals satisfied.",
            expected_outcome="Task finished",
            frame_sequence=curr_frame_seq,
        )
        return System2Decision(
            thought="Task requirements satisfied based on observed visual state.",
            task_status="completed",
            next_intent=intent,
        )

    def _infer_target_from_task(self, task: str, history: List[Dict[str, Any]]) -> Optional[str]:
        """Simple heuristic extraction of target entities from natural-language task prompt."""
        t_low = task.lower()
        if "settings" in t_low and len(history) == 0:
            return "Settings icon"
        if "save" in t_low and len(history) == 0:
            return "Save button"
        if "file" in t_low and len(history) == 0:
            return "File menu"
        if "terminal" in t_low and len(history) == 0:
            return "Terminal tab"
        if "search" in t_low and len(history) == 0:
            return "Search input field"
        return None


def _load_dotenv_if_needed() -> None:
    """Loads environment variables from local .env if present without external dependencies."""
    try:
        from pathlib import Path
        for p in [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]:
            if p.is_file():
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
                break
    except Exception:
        pass


class LLMReasoningProvider(ReasoningProvider):
    """External LLM-based reasoning provider using standard HTTP API endpoints.

    Features:
    - Auto-loads .env credentials from project root if not in environment.
    - Zero credential leakage: credentials are never logged, formatted, or exposed in exceptions.
    - Robust retries with exponential backoff on HTTP 429 (rate limit) or transient 5xx errors.
    - Accurate token telemetry: tracks prompt, completion, and total tokens per turn and cumulatively.
    - Resilient JSON parsing: extracts schema even when framed with markdown fences or dialogue.
    - Connection verification probe via verify_live_connection().
    """

    SYSTEM_PROMPT = """You are AVCS System 2, the high-level reasoning and planning component of the Artificial Visual Cortex System.
Your job is to inspect structured visual state observations from AVCS System 1 and decide the next logical interaction step to achieve the user's task.

CRITICAL RULES:
1. DO NOT GENERATE SCREEN COORDINATES (x, y, or bbox). System 1 owns all visual grounding and coordinate geometry.
2. Specify targets purely through descriptive natural language (e.g., 'Settings icon', 'Submit button', 'Search input field').
3. If the observation indicates a requested target is 'ABSENT', do not attempt to click it. Replan or emit 'ask_user' or 'stop'.
4. If the observation indicates temporal instability ('busy' or 'surprise'), propose 'wait' until the screen settles.
5. If multiple targets are ambiguous or confidence is low, emit 'ask_user' or narrow the target description.
6. When the task is complete, emit 'finish'.
7. Output valid JSON only, matching the exact format:
{
  "thought": "Your step-by-step reasoning about the observation and task progress",
  "task_status": "in_progress" | "waiting_for_observation" | "needs_clarification" | "completed" | "aborted",
  "next_intent": {
    "op": "observe_target" | "click" | "double_click" | "type_text" | "press_key" | "scroll" | "wait" | "ask_user" | "stop" | "finish",
    "target_description": "Natural language target or null",
    "text_input": "Text to type or null",
    "key_name": "Key name or null",
    "expected_outcome": "What visual evidence you expect to observe next",
    "rationale": "Why this action is chosen"
  },
  "clarification_prompt": "Optional question for user"
}
"""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        _load_dotenv_if_needed()
        # Resolve endpoint/key/model from the caller, the user's environment, or
        # the platform-managed key. Doing this in one place is what makes the
        # brain *live* out of the box inside a managed runtime instead of
        # silently degrading to the rule brain.
        from avcs.reasoning.credentials import resolve_llm_credentials

        creds = resolve_llm_credentials(api_key=api_key, endpoint=endpoint, model=model)
        self.api_key = creds.api_key
        explicit_endpoint = (endpoint or "").strip()
        self.endpoint = (explicit_endpoint or creds.endpoint).rstrip("/")
        self.model = creds.model
        self.credential_source = creds.source

        self.timeout_seconds = float(os.environ.get("AVCS_LLM_TIMEOUT", timeout_seconds))
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        # Telemetry & Token Tracking
        self.last_usage: Optional[Dict[str, int]] = None
        self.total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.request_count: int = 0
        self.retry_count: int = 0
        self.error_count: int = 0
        self.last_latency_ms: float = 0.0

    @property
    def masked_api_key(self) -> str:
        """Returns safely masked API key for logging without leaking secrets."""
        if not self.api_key:
            return "None"
        if len(self.api_key) <= 8:
            return "***"
        return f"{self.api_key[:6]}...{self.api_key[-4:]}"

    def __repr__(self) -> str:
        return f"LLMReasoningProvider(endpoint='{self.endpoint}', model='{self.model}', api_key='{self.masked_api_key}')"

    def reset(self) -> None:
        """Resets session-level telemetry."""
        self.last_usage = None
        self.last_latency_ms = 0.0

    def verify_live_connection(self) -> Dict[str, Any]:
        """Probes the configured endpoint with a minimal completion to verify credentials and latency."""
        t0 = time.perf_counter()
        probe_payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "Respond with JSON: {\"status\": \"verified\"}"}
            ],
            # Reasoning models spend the first tokens on hidden reasoning, so a
            # tight budget returns content:null and the probe looks failed when
            # the endpoint is fine. Leave room for the visible answer.
            "max_tokens": 512,
            "temperature": 0.0,
        }
        raw = self._post_request(probe_payload)
        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "connected",
            "endpoint": self.endpoint,
            "model": self.model,
            "latency_ms": round(dt, 2),
            "masked_key": self.masked_api_key,
            "last_usage": self.last_usage,
            "probe_output": (raw or "").strip(),
        }

    def plan_step(
        self,
        task: str,
        observation: Dict[str, Any],
        history: List[Dict[str, Any]],
        step_index: int = 1,
        system_prompt: Optional[str] = None,
    ) -> System2Decision:
        payload = self._build_payload(task, observation, history, step_index, system_prompt)

        t0 = time.perf_counter()
        raw_response = self._post_request(payload)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        return self._parse_llm_json(raw_response, step_index, observation.get("frame_sequence"))

    def _extract_message_text(self, data: Dict[str, Any]) -> str:
        """Assistant text from an OpenAI-compatible body (reasoning-model safe)."""
        from avcs.reasoning.credentials import extract_message_text

        return extract_message_text(data)

    def raw_step(
        self,
        task: str,
        observation: Dict[str, Any],
        history: List[Dict[str, Any]],
        step_index: int = 1,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Return the model's raw reply text for an arbitrary system prompt.

        ``plan_step`` is bound to the System 2 *planner* schema (``next_intent``).
        A live conversation uses a different schema (``speech`` / ``kind`` /
        ``action``), so the voice brain needs the reply text itself and parses it
        against the conversation contract. Without this the model's ``action``
        field was silently dropped and every turn degraded to a bare
        ``observe_target``.
        """
        payload = self._build_payload(task, observation, history, step_index, system_prompt)

        t0 = time.perf_counter()
        raw_response = self._post_request(payload)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return raw_response

    def _build_payload(self, task, observation, history, step_index, system_prompt=None):
        """Build the OpenAI-compatible chat payload shared by step and stream.

        ``history`` arrives in one of two shapes. The System 2 control loop passes
        step records (``step_index`` / ``proposed_action``), which are summarised
        compactly. The voice brain passes real conversation turns
        (``{"role", "content"}``); those are forwarded verbatim as chat messages,
        otherwise the model would answer each turn in isolation and the
        conversation would have no memory.
        """
        convo = [h for h in (history or []) if isinstance(h, dict) and "content" in h]
        steps = [h for h in (history or []) if isinstance(h, dict) and "content" not in h]

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt or self.SYSTEM_PROMPT},
        ]
        for turn in convo:
            role = turn.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            messages.append({"role": role, "content": str(turn.get("content", ""))})
        messages.append({
            "role": "user",
            "content": json.dumps({
                "task": task,
                "current_step": step_index,
                "observation": observation,
                "step_history_summary": [
                    {"step": h.get("step_index"), "op": h.get("proposed_action", {}).get("op")}
                    for h in steps[-3:]
                ],
            }, indent=2),
        })
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

    def stream_step(
        self,
        task,
        observation,
        history,
        step_index=1,
        system_prompt=None,
    ):
        """Yield the reply in fragments as the provider produces them (SSE).

        Falls back to a single non-streamed fragment if the endpoint rejects
        ``stream=true`` or the connection drops mid-stream, so a caller can treat
        this generator exactly like a chunk iterator without special cases.
        """
        if not _is_local_endpoint(self.endpoint):
            raise RuntimeError(
                "External cloud LLM endpoints are disabled in 100% self-contained local VLA mode."
            )
        return self._stream_step_iter(task, observation, history, step_index, system_prompt)

    def _stream_step_iter(
        self,
        task,
        observation,
        history,
        step_index=1,
        system_prompt=None,
    ):
        import json as _json
        import urllib.request

        payload = self._build_payload(task, observation, history, step_index, system_prompt)
        payload["stream"] = True
        url = f"{self.endpoint.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "AVCS-System2-Bridge/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            url, data=_json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        got_any = False
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith(_SSE_DATA_PREFIX):
                        continue
                    data = line[len(_SSE_DATA_PREFIX):].strip()
                    if data == _SSE_DONE:
                        break
                    try:
                        obj = _json.loads(data)
                    except _json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    for piece_key in ("reasoning_content", "content"):
                        piece = delta.get(piece_key)
                        if piece:
                            got_any = True
                            yield piece
        except Exception:
            if got_any:
                return
            decision = self.plan_step(task, observation, history, step_index)
            yield (decision.clarification_prompt or decision.thought or "")
            return
        if not got_any:
            decision = self.plan_step(task, observation, history, step_index)
            yield (decision.clarification_prompt or decision.thought or "")

    def _post_request(self, payload: Dict[str, Any]) -> str:
        """Executes HTTP request to LLM provider with retry/backoff and token metrics."""
        import urllib.error
        import urllib.request

        if not _is_local_endpoint(self.endpoint):
            raise RuntimeError(
                "External cloud LLM endpoints are disabled in 100% self-contained local VLA mode."
            )

        url = f"{self.endpoint.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AVCS-System2-Bridge/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req_data = json.dumps(payload).encode("utf-8")
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            self.request_count += 1
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    resp_bytes = resp.read()
                    data = json.loads(resp_bytes.decode("utf-8"))

                    # Track token usage if reported by provider
                    usage = data.get("usage")
                    if isinstance(usage, dict):
                        self.last_usage = {
                            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                            "completion_tokens": int(usage.get("completion_tokens", 0)),
                            "total_tokens": int(usage.get("total_tokens", 0)),
                        }
                        for k, v in self.last_usage.items():
                            self.total_usage[k] = self.total_usage.get(k, 0) + v

                    return self._extract_message_text(data)

            except urllib.error.HTTPError as err:
                last_err = err
                self.error_count += 1
                # Retry on rate-limit (429) or server errors (500, 502, 503, 504)
                if err.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    self.retry_count += 1
                    backoff = self.retry_delay_seconds * (2 ** (attempt - 1))
                    time.sleep(backoff)
                    continue

                # Read body for diagnostics, masking any sensitive credentials
                err_body = ""
                try:
                    err_body = err.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                raise RuntimeError(
                    f"LLM Provider HTTP {err.code} on model '{self.model}' [endpoint={self.endpoint}]: {err_body}"
                ) from err

            except Exception as err:
                last_err = err
                self.error_count += 1
                if attempt < self.max_retries:
                    self.retry_count += 1
                    time.sleep(self.retry_delay_seconds)
                    continue
                raise RuntimeError(
                    f"LLM Provider connection failed on model '{self.model}' [endpoint={self.endpoint}]: {err}"
                ) from err

        raise RuntimeError(f"LLM Provider exceeded {self.max_retries} attempts: {last_err}")

    def _parse_llm_json(
        self,
        raw_text: str,
        step_index: int,
        frame_sequence: Optional[int],
    ) -> System2Decision:
        """Parses LLM JSON response with markdown fence stripping and robust fallback handling."""
        cleaned = raw_text.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if match:
                cleaned = match.group(1).strip()

        # Fallback regex extraction if leading/trailing commentary exists
        if not (cleaned.startswith("{") and cleaned.endswith("}")):
            brace_match = re.search(r"(\{[\s\S]*\})", cleaned)
            if brace_match:
                cleaned = brace_match.group(1).strip()

        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            # Fallback for malformed JSON
            intent = ActionIntent.create(
                step_index=step_index,
                op=ActionOp.STOP,
                rationale=f"Failed to parse LLM JSON response: {e}",
                expected_outcome="Session halted due to malformed reasoning output",
                frame_sequence=frame_sequence,
            )
            return System2Decision(
                thought=f"Malformed JSON emitted by LLM provider: {raw_text[:120]}. Aborting.",
                task_status="aborted",
                next_intent=intent,
            )

        raw_intent = parsed.get("next_intent", {})
        intent, had_injection = ActionIntent.from_dict(raw_intent, default_step=step_index)
        if frame_sequence and intent.frame_sequence is None:
            object.__setattr__(intent, "frame_sequence", frame_sequence)

        return System2Decision(
            thought=parsed.get("thought", "Reasoning step completed."),
            task_status=parsed.get("task_status", "in_progress"),
            next_intent=intent,
            clarification_prompt=parsed.get("clarification_prompt"),
            confidence=float(parsed.get("confidence", 1.0)),
        )
