"""Cognitive Intelligence & Sub-Goal Planner (Layer 1).

Decouples high-level strategic reasoning from low-level motor actuation:
1. Operates at low frequency (0.5 - 1.0 Hz).
2. Ingests user instructions (pure English).
3. Decomposes high-level intent into an ordered sequence of atomic sub-goals.
4. Enforces strict architectural invariant: NEVER outputs low-level (x, y) coordinates.
5. Ingests failure reflection diagnostics from Layer 5 to synthesize alternative strategic routes.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("avcs.ace.planner")


def _validate_no_coordinates(text: str) -> None:
    """Enforces strict architectural invariant: SubGoal must never contain low-level (x, y) coordinates."""
    if not text:
        return

    # Prohibited coordinate patterns
    coord_patterns = [
        r'[\(\[<\{]\s*-?\d+(?:\.\d+)?(?:px)?\s*,\s*-?\d+(?:\.\d+)?(?:px)?\s*[\)\]>\}]',  # (100, 200), [100, 200], <100, 200>, {100, 200}
        r'\{\s*[\'"]?[xy][\'"]?\s*[:=]\s*-?\d+',                                      # {"x": 100} or {x: 100}
        r'\b(?:x|y)\s*[:=]\s*-?\d+',                                                   # x=100, x: 100, y=200, y: 200
        r'\bx\s*=\s*\d+\s+y\s*=\s*\d+',                                               # x=500 y=400
        r'ACT:(?:[A-Z_]+)\s+\d+',                                                      # ACT:CLICK 100 200, ACT:LEFT_CLICK 100 200
        r'\b(?:point|loc|location|pixel|pos|position|coord(?:inate)?s?)\s*[:=]?\s*[\(\[<\{]?\s*-?\d+', # point: 500, 400
        r'\d+\s*px(?:\s*,\s*|\s+)\d+\s*px',                                           # 500px, 400px or 500px 400px
        r'\b(?:click|tap|press|move|mouse)\s+(?:at\s+)?[\(\[<\{]?\s*-?\d+(?:\.\d+)?(?:px)?\s*[,x\s]\s*-?\d+', # click at 100, 200
        r'\bat\s+[\(\[<\{]?\s*-?\d+(?:\.\d+)?(?:px)?\s*[,x\s]\s*-?\d+(?:\.\d+)?(?:px)?[\)\]>\}]?', # at (100, 200) or at 100, 200
    ]

    for pat in coord_patterns:
        if re.search(pat, text, re.IGNORECASE):
            raise ValueError(
                f"Invariant violation: SubGoal must never contain low-level (x, y) coordinates. "
                f"Found forbidden coordinate pattern {pat!r} in text: {text!r}"
            )


def _validate_english_only(text: str) -> None:
    """Enforces strict user requirement: SubGoal must be strictly in English."""
    if not text:
        return
    if any(ord(c) > 127 for c in text):
        raise ValueError(
            f"Invariant violation: SubGoal must be in English only (pure English code only). "
            f"Found non-English text: {text!r}"
        )


def _sanitize_coordinates(text: str) -> str:
    """Sanitizes raw (x, y) coordinates into semantic directives so planner never leaks coordinates."""
    if not text:
        return text

    s = text
    # Replace tuple/bracket/angle/brace coords like (100, 200) or [100, 200] or <100, 200> or {100, 200}
    s = re.sub(r'[\(\[<\{]\s*-?\d+(?:\.\d+)?(?:px)?\s*,\s*-?\d+(?:\.\d+)?(?:px)?\s*[\)\]>\}]', 'target element', s)
    # Replace point: 500, 400 or location: 500, 400 or coords: [100, 200]
    s = re.sub(r'\b(?:point|loc|location|pixel|px|pos|position|coord(?:inate)?s?)\s*[:=]?\s*[\(\[<\{]?\s*-?\d+.*?(?:[\)\]>\}]|$)', 'target element', s, flags=re.IGNORECASE)
    # Replace explicit x=..., y=... or x: ..., y: ... or x=500 y=400
    s = re.sub(r'\bx\s*=\s*\d+\s+y\s*=\s*\d+', 'target element', s, flags=re.IGNORECASE)
    s = re.sub(r'\b[xy]\s*[:=]\s*-?\d+(?:\.\d+)?', 'target element', s, flags=re.IGNORECASE)
    # Replace 500px, 400px or 500px 400px
    s = re.sub(r'\d+\s*px(?:\s*,\s*|\s+)\d+\s*px', 'target element', s, flags=re.IGNORECASE)
    # Replace ACT:CLICK 100 200 or ACT:LEFT_CLICK 100 200
    s = re.sub(r'ACT:(?:[A-Z_]+)\s+\d+(?:\s+\d+)?', 'interact with element', s, flags=re.IGNORECASE)
    # Replace click at 100, 200
    s = re.sub(r'\b(?:click|tap|press|move)\s+at\s+[\(\[<\{]?\s*-?\d+(?:\.\d+)?(?:px)?\s*[,x\s]\s*-?\d+(?:px)?[\)\]>\}]?', 'click target element', s, flags=re.IGNORECASE)
    s = re.sub(r'\bat\s+[\(\[<\{]?\s*-?\d+(?:\.\d+)?(?:px)?\s*[,x\s]\s*-?\d+(?:px)?[\)\]>\}]?', 'at target element', s, flags=re.IGNORECASE)
    # Clean up excess spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _translate_to_english_directive(text: str) -> Tuple[str, str]:
    """Normalizes arbitrary text into clean English description and expected UI state."""
    s = re.sub(r'[\(\[<\{].*?[\)\]>\}]', '', text)
    s = re.sub(r'[^a-zA-Z0-9_\-\.\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if not s:
        s = "Execute target desktop action"
    desc = s[0].upper() + s[1:]
    expected = f"Completed '{desc}'"
    return desc, expected


@dataclass
class SubGoal:
    """Atomic semantic sub-goal directive emitted by Layer 1 Cognitive Planner.

    Strict Invariants:
    1. NEVER contains low-level (x, y) coordinates.
    2. Strictly English only (pure English code only).
    """

    index: int
    description: str
    expected_ui_state: str
    status: str = "pending"  # "pending", "in_progress", "completed", "failed"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_no_coordinates(self.description)
        _validate_no_coordinates(self.expected_ui_state)
        _validate_english_only(self.description)
        _validate_english_only(self.expected_ui_state)

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    @property
    def is_in_progress(self) -> bool:
        return self.status == "in_progress"

    def mark_in_progress(self) -> None:
        self.status = "in_progress"

    def mark_completed(self) -> None:
        self.status = "completed"

    def mark_failed(self, reason: str = "") -> None:
        self.status = "failed"
        if reason:
            self.metadata["failure_reason"] = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "description": self.description,
            "expected_ui_state": self.expected_ui_state,
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SubGoal:
        return cls(
            index=int(data.get("index", 1)),
            description=str(data.get("description", "")),
            expected_ui_state=str(data.get("expected_ui_state", "")),
            status=str(data.get("status", "pending")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ReflectionDiagnostic:
    """Diagnostic telemetry emitted by Layer 5 / Runner when a sub-goal fails or stalls."""

    failed_subgoal: SubGoal
    failure_reason: str
    stuck_repetitions: int = 0
    stuck_action: Optional[Dict[str, Any]] = None
    active_window: Optional[str] = None
    active_process: Optional[str] = None
    optical_delta: float = 0.0
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failed_subgoal": self.failed_subgoal.to_dict(),
            "failure_reason": self.failure_reason,
            "stuck_repetitions": self.stuck_repetitions,
            "stuck_action": self.stuck_action,
            "active_window": self.active_window,
            "active_process": self.active_process,
            "optical_delta": self.optical_delta,
            "details": self.details,
        }


class CognitivePlanner:
    """Layer 1 Cognitive Intelligence & Sub-Goal Planner.

    Operates at low frequency (0.5 - 1.0 Hz) to decompose multilingual user instructions
    into an ordered sequence of coordinate-free atomic sub-goals, and dynamically replans
    when reflection diagnostics report failure or loop deadlock.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        rate_limit_hz: float = 1.0,
        enforce_rate_limit: bool = False,
    ) -> None:
        self.client = client
        self.rate_limit_hz = max(0.1, min(2.0, rate_limit_hz))
        self.min_interval_s = 1.0 / self.rate_limit_hz
        self.enforce_rate_limit = enforce_rate_limit
        self.last_plan_time: float = 0.0

    def _apply_rate_limit(self) -> None:
        """Enforces low-frequency operating frequency (0.5 - 1.0 Hz) without spinning."""
        now = time.monotonic()
        elapsed = now - self.last_plan_time
        if elapsed < self.min_interval_s:
            wait_time = self.min_interval_s - elapsed
            if self.enforce_rate_limit:
                time.sleep(wait_time)
            logger.debug(f"Planner rate-limit: {elapsed:.2f}s elapsed (budget: {self.min_interval_s:.2f}s).")
        self.last_plan_time = time.monotonic()

    def _is_local_client_allowed(self) -> bool:
        """Returns True only when self.client is a local VLA/loopback or in-process client."""
        if self.client is None:
            return False
        from avcs.runtime.client import is_local_endpoint

        for attr in ("endpoint", "base_url", "server_url"):
            val = getattr(self.client, attr, None)
            if isinstance(val, str) and val.strip():
                if not is_local_endpoint(val):
                    logger.warning(
                        f"CognitivePlanner blocked non-local endpoint {val!r}; "
                        "falling back to local deterministic decomposition."
                    )
                    return False

        if type(self.client).__name__ == "CustomLLMClient":
            ep = getattr(self.client, "endpoint", "")
            if not isinstance(ep, str) or not is_local_endpoint(ep):
                logger.warning(
                    f"CognitivePlanner blocked non-local/unconfigured CustomLLMClient ({ep!r}); "
                    "falling back to local deterministic decomposition."
                )
                return False
        return True

    def decompose(self, goal: str) -> List[SubGoal]:
        """Decomposes a user goal into an ordered sequence of atomic sub-goals.

        Guarantees strict invariant: NEVER outputs low-level (x, y) coordinates.
        """
        self._apply_rate_limit()
        goal = (goal or "").strip()
        if not goal:
            return []
        goal = _sanitize_coordinates(goal)

        # 1. Attempt local-only VLA/LLM decomposition if local client is provided
        if self._is_local_client_allowed():
            try:
                llm_subgoals = self._decompose_with_llm(goal)
                if llm_subgoals:
                    return llm_subgoals
            except Exception as e:
                logger.warning(f"CognitivePlanner LLM decomposition fallback to deterministic engine: {e}")

        # 2. Deterministic Rule-Based Semantic Parser
        return self._decompose_multilingual_rules(goal)

    def replan(
        self,
        goal: str,
        failed_subgoal: SubGoal,
        failure_reason: str = "",
        diagnostic: Optional[ReflectionDiagnostic] = None,
        stuck_points: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SubGoal]:
        """Ingests failure reflection diagnostics from Layer 5 and synthesizes alternative strategic routes."""
        self._apply_rate_limit()
        logger.info(
            f"CognitivePlanner Dynamic Replanning invoked for failed sub-goal: '{failed_subgoal.description}'. "
            f"Reason: {failure_reason}"
        )

        # 1. Attempt local-only VLA/LLM replanning if local client is provided
        if self._is_local_client_allowed():
            try:
                llm_replan = self._replan_with_llm(
                    goal=goal,
                    failed_subgoal=failed_subgoal,
                    failure_reason=failure_reason,
                    diagnostic=diagnostic,
                )
                if llm_replan:
                    return llm_replan
            except Exception as e:
                logger.warning(f"CognitivePlanner LLM replanning fallback to deterministic strategies: {e}")

        # 2. Deterministic Alternative Strategic Route Synthesis
        return self._replan_deterministic(
            goal=goal,
            failed_subgoal=failed_subgoal,
            failure_reason=failure_reason,
            diagnostic=diagnostic,
            stuck_points=stuck_points,
        )

    # -------------------------------------------------------------------------
    # Multilingual Rule-Based Decomposition Engine
    # -------------------------------------------------------------------------

    def _split_clauses(self, text: str) -> List[str]:
        """Splits composite instructions on sequential delimiters."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Sequential English delimiters:
        pattern = (
            r'\s*(?:'
            r'[,;]|\->|\||'
            r'\band\s+then\b|\bafter\s+that\b|\band\b|\bthen\b|'
            r'\bnext\b|\bfollowed\s+by\b'
            r')\s*'
        )
        parts = re.split(pattern, text, flags=re.IGNORECASE)
        clauses = [p.strip() for p in parts if p.strip()]
        return clauses if clauses else [text]

    def _decompose_multilingual_rules(self, goal: str) -> List[SubGoal]:
        """Generates coordinate-free semantic sub-goals using deterministic linguistic patterns.

        Guarantees strict invariant: All descriptions and expected UI states are strictly English.
        """
        clauses = self._split_clauses(goal)
        subgoals: List[SubGoal] = []

        # Known application keywords and their clean canonical names
        app_catalog = {
            "notepad": ("Notepad", "Notepad application window active and editor focused"),
            "calculator": ("Calculator", "Calculator application window active"),
            "calc": ("Calculator", "Calculator application window active"),
            "chrome": ("Google Chrome", "Google Chrome window active and address bar visible"),
            "google chrome": ("Google Chrome", "Google Chrome window active and address bar visible"),
            "edge": ("Microsoft Edge", "Microsoft Edge window active"),
            "browser": ("Web Browser", "Browser window active"),
            "paint": ("Paint", "Paint application canvas ready"),
            "explorer": ("File Explorer", "File Explorer window open"),
            "file explorer": ("File Explorer", "File Explorer window open"),
            "cmd": ("Command Prompt", "Command prompt window open"),
            "terminal": ("Terminal", "Terminal window open"),
            "powershell": ("PowerShell", "PowerShell window open"),
            "settings": ("Windows Settings", "Settings window open"),
            "start menu": ("Windows Start Menu", "Start menu opened and search input visible"),
            "start": ("Windows Start Menu", "Start menu opened and search input visible"),
            "task manager": ("Task Manager", "Task Manager window open"),
            "word": ("Microsoft Word", "Word document editor open"),
            "excel": ("Microsoft Excel", "Excel spreadsheet active"),
            "spotify": ("Spotify", "Spotify application window open"),
            "youtube": ("YouTube", "YouTube application or browser tab active"),
            "google": ("Google", "Google search interface visible"),
        }

        idx = 1
        for clause in clauses:
            raw_clean = clause.strip()
            # Enforce strict invariant by sanitizing any low-level coordinates into semantic directives
            clean = _sanitize_coordinates(raw_clean)
            lower = clean.lower()

            # Pattern A: Start menu opening (avoid hijacking "Start <App>" like "Start Chrome")
            if re.search(r'\bstart\s*menu\b', clean, re.IGNORECASE) or lower.strip() in ("start", "open start"):
                if re.search(r'(?:open|start|launch|press|show)', clean, re.IGNORECASE) or lower in ("start", "start menu"):
                    subgoals.append(SubGoal(
                        index=idx,
                        description="Open Windows Start Menu",
                        expected_ui_state="Start menu opened and search input visible",
                    ))
                    idx += 1
                    continue

            # Pattern B0: Close / Exit application or window (must precede launch pattern)
            close_verbs = ("close", "exit", "quit", "terminate", "kill", "shut down", "dismiss")
            if any(re.search(r'\b' + re.escape(cv) + r'\b', lower) for cv in close_verbs):
                matched_app = None
                for app_key, (app_name, _) in app_catalog.items():
                    if re.search(r'(?:\b|^)' + re.escape(app_key) + r'(?:\b|$)', lower):
                        matched_app = app_name
                        break
                if matched_app:
                    subgoals.append(SubGoal(
                        index=idx,
                        description=f"Close {matched_app} application",
                        expected_ui_state=f"{matched_app} application closed and window removed",
                    ))
                    idx += 1
                    continue
                else:
                    subgoals.append(SubGoal(
                        index=idx,
                        description="Close active window",
                        expected_ui_state="Target window closed and desktop returned to foreground",
                    ))
                    idx += 1
                    continue

            # Pattern B: Search query in search engine or app
            search_match = re.search(
                r'^\s*(?:search|find|lookup)\s+(?:for\s+)?([a-zA-Z0-9_\-\.\s]+?)(?:\s+(?:on|in)\s+([a-zA-Z0-9_\-\.\s]+))?$',
                clean,
                re.IGNORECASE,
            )
            if search_match:
                query_raw = search_match.group(1).strip()
                platform_raw = search_match.group(2).strip() if search_match.group(2) else ""
                platform = "Google" if "google" in platform_raw.lower() else platform_raw.title()
                if not platform:
                    matched_app_tuple = None
                    for app_key, (app_name, expected_ui) in app_catalog.items():
                        if app_key == query_raw.lower():
                            matched_app_tuple = (app_name, expected_ui)
                            break
                    if matched_app_tuple:
                        app_name, expected_ui = matched_app_tuple
                        subgoals.append(SubGoal(
                            index=idx,
                            description=f"Search for {app_name} and launch application",
                            expected_ui_state=expected_ui,
                        ))
                        idx += 1
                        continue
                dest = f" on {platform}" if platform else ""
                subgoals.append(SubGoal(
                    index=idx,
                    description=f"Search for '{query_raw}'{dest}",
                    expected_ui_state=f"Search results for '{query_raw}' displayed{dest}",
                ))
                idx += 1
                continue

            # Pattern B2: Direct search: e.g. "Notepad search" or "Notepad search and launch"
            direct_search = re.search(
                r'([a-zA-Z0-9_\-\.\s]+?)\s+(?:search|find|lookup)(?:\s+(?:open|launch))?$',
                clean,
                re.IGNORECASE,
            )
            if direct_search:
                query = direct_search.group(1).strip()
                if query and query.lower() not in ("and", "then"):
                    matched_app_tuple = None
                    for app_key, (app_name, expected_ui) in app_catalog.items():
                        if app_key in query.lower():
                            matched_app_tuple = (app_name, expected_ui)
                            break
                    if matched_app_tuple:
                        app_name, expected_ui = matched_app_tuple
                        subgoals.append(SubGoal(
                            index=idx,
                            description=f"Search for {app_name} and launch application",
                            expected_ui_state=expected_ui,
                        ))
                    else:
                        subgoals.append(SubGoal(
                            index=idx,
                            description=f"Search for '{query}'",
                            expected_ui_state=f"Search results for '{query}' displayed",
                        ))
                    idx += 1
                    continue

            # Pattern C: Application launch / open
            matched_app = None
            for app_key, (app_name, expected_ui) in app_catalog.items():
                if re.search(r'(?:\b|^)' + re.escape(app_key) + r'(?:\b|$)', lower):
                    matched_app = (app_name, expected_ui)
                    break

            launch_verbs = ["open", "launch", "start", "run", "execute"]
            is_standalone_app = lower in app_catalog or any(lower == k for k in app_catalog)
            if matched_app and (any(act in lower for act in launch_verbs) or is_standalone_app):
                app_name, expected_ui = matched_app
                subgoals.append(SubGoal(
                    index=idx,
                    description=f"Launch {app_name} application",
                    expected_ui_state=expected_ui,
                ))
                idx += 1
                continue

            # Pattern D: Typing text
            type_m = re.search(
                r'^\s*(?:type|write|enter|input)\s+(?:text\s+)?(.+)',
                clean,
                re.IGNORECASE,
            )
            if type_m:
                text_to_type = type_m.group(1).strip()
                if (text_to_type.startswith('"') and text_to_type.endswith('"')) or (text_to_type.startswith("'") and text_to_type.endswith("'")):
                    text_to_type = text_to_type[1:-1].strip()
                subgoals.append(SubGoal(
                    index=idx,
                    description=f"Type text '{text_to_type}' into active target field",
                    expected_ui_state=f"Text '{text_to_type}' visible in target field",
                ))
                idx += 1
                continue

            # Pattern E: Calculation
            calc_m = re.search(r'(?:calculate|compute|solve)\s*(.+)?', clean, re.IGNORECASE)
            if calc_m or any(k in lower for k in ["calculate", "compute"]):
                expr = clean
                for w in ["calculate", "compute", "solve"]:
                    expr = expr.replace(w, "")
                expr = expr.strip()
                subgoals.append(SubGoal(
                    index=idx,
                    description=f"Calculate {expr or 'expression'}",
                    expected_ui_state="Calculation result displayed",
                ))
                idx += 1
                continue

            # Pattern F: Save file
            if any(k in lower for k in ["save", "save file", "save document"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Save document or current file",
                    expected_ui_state="File saved confirmation dialog or status",
                ))
                idx += 1
                continue

            # Pattern G: Media playback (audio, song, video)
            if any(k in lower for k in ["play", "play media", "play music", "play song", "play video"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Play media or audio playback",
                    expected_ui_state="Media player or browser playback active",
                ))
                idx += 1
                continue

            # Pattern H: Refresh page or view
            if any(k in lower for k in ["refresh", "reload", "update view"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Refresh current page or view",
                    expected_ui_state="Page reloaded with refreshed content",
                ))
                idx += 1
                continue

            # Pattern I: Window minimize / maximize
            if any(k in lower for k in ["minimize", "minimise"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Minimize active window",
                    expected_ui_state="Active window minimized to taskbar",
                ))
                idx += 1
                continue
            if any(k in lower for k in ["maximize", "maximise"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Maximize active window",
                    expected_ui_state="Active window maximized to full screen",
                ))
                idx += 1
                continue

            # Pattern J: Clipboard copy / paste / select all
            if any(k in lower for k in ["copy", "copy to clipboard"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Copy selected content",
                    expected_ui_state="Selected text or element copied to clipboard",
                ))
                idx += 1
                continue
            if any(k in lower for k in ["paste", "paste from clipboard"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Paste clipboard content",
                    expected_ui_state="Clipboard content pasted into active field",
                ))
                idx += 1
                continue
            if any(k in lower for k in ["select all", "select-all"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Select all content",
                    expected_ui_state="All items or text in active view selected",
                ))
                idx += 1
                continue

            # Default fallback for arbitrary clause (guaranteed pure English)
            desc_en, expected_en = _translate_to_english_directive(clean)
            subgoals.append(SubGoal(
                index=idx,
                description=desc_en,
                expected_ui_state=expected_en,
            ))
            idx += 1

        # Fallback if no subgoals generated
        if not subgoals:
            sanitized_goal = _sanitize_coordinates(goal)
            desc_en, expected_en = _translate_to_english_directive(sanitized_goal)
            subgoals.append(SubGoal(
                index=1,
                description=desc_en,
                expected_ui_state=expected_en,
            ))

        return subgoals

    # -------------------------------------------------------------------------
    # LLM-Based Decomposition Engine
    # -------------------------------------------------------------------------

    def _decompose_with_llm(self, goal: str) -> List[SubGoal]:
        """Calls local VLA/LLM client to produce coordinate-free sub-goals."""
        if not self._is_local_client_allowed():
            return []
        system_prompt = (
            "You are Layer 1 Cognitive Intelligence for AVCS Autonomous Desktop OS. "
            "Decompose user goals into an ordered sequence of atomic sub-goals.\n"
            "STRICT INVARIANT: NEVER output (x, y) coordinates. Only output semantic intent and expected UI state.\n"
            "Format your reply ONLY as a valid JSON array of objects:\n"
            '[{"index": 1, "description": "...", "expected_ui_state": "..."}]'
        )
        user_prompt = f"Decompose this goal: '{goal}'"

        response = self.client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )

        # Parse JSON
        text = response.strip()
        # Strip code blocks
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()

        data = json.loads(text)
        subgoals: List[SubGoal] = []
        if isinstance(data, list):
            for i, item in enumerate(data, start=1):
                desc = _sanitize_coordinates(str(item.get("description", "")).strip())
                ui_state = _sanitize_coordinates(str(item.get("expected_ui_state", "")).strip())
                _validate_no_coordinates(desc)
                _validate_no_coordinates(ui_state)
                subgoals.append(SubGoal(
                    index=i,
                    description=desc,
                    expected_ui_state=ui_state,
                ))
        return subgoals

    # -------------------------------------------------------------------------
    # Dynamic Replanning & Strategic Route Synthesis
    # -------------------------------------------------------------------------

    def _replan_deterministic(
        self,
        goal: str,
        failed_subgoal: SubGoal,
        failure_reason: str,
        diagnostic: Optional[ReflectionDiagnostic] = None,
        stuck_points: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SubGoal]:
        """Synthesizes alternative strategic routes when a sub-goal fails or stalls."""
        desc_lower = failed_subgoal.description.lower()
        reason_lower = (failure_reason or "").lower()
        active_win = (diagnostic.active_window if diagnostic else "") or ""
        stuck_reps = diagnostic.stuck_repetitions if diagnostic else 0
        if stuck_points and not stuck_reps:
            stuck_reps = len(stuck_points)

        replanned: List[SubGoal] = []
        idx = failed_subgoal.index

        # Max retry bound to prevent infinite replanning loops
        MAX_RETRY_BOUND = 3
        if stuck_reps > MAX_RETRY_BOUND or "deadlock" in reason_lower:
            replanned.append(SubGoal(
                index=idx,
                description="Press Escape to dismiss modal dialogs and switch focus with Alt+Tab",
                expected_ui_state="Active dialog dismissed and primary window refocused",
            ))
            replanned.append(SubGoal(
                index=idx + 1,
                description=f"Alternative approach: reopen '{goal}' directly",
                expected_ui_state=f"Fresh session for '{goal}' initialized",
            ))
            return replanned

        # Route 1: Loop deadlock or modal alert blocking UI (repetition >= 2)
        if stuck_reps >= 2 or "loop" in reason_lower or "stuck" in reason_lower:
            clean_desc = failed_subgoal.description
            while True:
                m = re.match(r"^Retry\s*['\"](.*?)['\"](?:\s+(?:visually|using\s+[\w\s]+))?$", clean_desc, re.IGNORECASE)
                if m:
                    clean_desc = m.group(1).strip()
                else:
                    break

            perturbation_dx = (stuck_reps % 3 - 1) * 15  # -15, 0, +15 px
            perturbation_dy = (stuck_reps % 2) * 10       # 0, 10 px
            perturb_dir = []
            if perturbation_dy > 0:
                perturb_dir.append("downward")
            if perturbation_dx < 0:
                perturb_dir.append("left")
            elif perturbation_dx > 0:
                perturb_dir.append("right")
            perturb_note = f" with slight {'-'.join(perturb_dir)} spatial offset" if perturb_dir else ""

            replanned.append(SubGoal(
                index=idx,
                description=f"Click blank area of desktop or active window{perturb_note} to dismiss temporary overlay and refocus",
                expected_ui_state="Screen stabilized and active workspace refocused",
            ))
            replanned.append(SubGoal(
                index=idx + 1,
                description=f"Retry '{clean_desc}' visually (attempt {min(stuck_reps, MAX_RETRY_BOUND)}/{MAX_RETRY_BOUND})",
                expected_ui_state=failed_subgoal.expected_ui_state,
            ))
            return replanned

        # Route 2: Start Menu click failure or stuck
        if "start menu" in desc_lower or "start" in desc_lower:
            replanned.append(SubGoal(
                index=idx,
                description="Click Windows Start icon on taskbar to open Start Menu",
                expected_ui_state="Start menu opened and search input visible",
            ))
            replanned.append(SubGoal(
                index=idx + 1,
                description="Type target application query into search",
                expected_ui_state="Application search results listed",
            ))
            return replanned

        # Route 3: Search bar focus / ungrounded typing blocked
        if "search" in desc_lower or "type" in desc_lower or "console" in reason_lower:
            replanned.append(SubGoal(
                index=idx,
                description="Click active search or text field directly to focus input caret",
                expected_ui_state="Target search field focused with active caret",
            ))
            replanned.append(SubGoal(
                index=idx + 1,
                description="Type query into focused input and select target result",
                expected_ui_state="Target result opened and displayed",
            ))
            return replanned

        # Route 4: Application launch / focus failed (e.g. icon click didn't trigger launch)
        app_candidates = ["notepad", "calculator", "calc", "chrome", "edge", "paint", "explorer", "cmd", "terminal", "powershell", "settings", "spotify", "slack", "vlc", "word", "excel"]
        found_app = next((a for a in app_candidates if a in desc_lower or a in reason_lower), None)
        if not found_app:
            m_app = re.search(r'(?:launch|open)\s+([a-zA-Z0-9_\-\.]+)(?:\s+application|\s+app)?', desc_lower, re.IGNORECASE)
            if m_app and m_app.group(1).lower() not in ("target", "the", "a", "an", "windows", "active"):
                found_app = m_app.group(1)
            else:
                m_goal = re.search(r'(?:open|launch|start)\s+([a-zA-Z0-9_\-\.]+)', goal, re.IGNORECASE)
                if not m_goal:
                    m_goal = re.search(r'([a-zA-Z0-9_\-\.]+)\s+(?:open|launch|start)', goal, re.IGNORECASE)
                if m_goal and m_goal.group(1).lower() not in ("start", "menu", "do", "and"):
                    found_app = m_goal.group(1)

        if found_app or "launch" in desc_lower or "open" in desc_lower:
            target_cmd = (found_app or "target application").title()
            replanned.append(SubGoal(
                index=idx,
                description="Click Windows Start icon on taskbar to open Start Menu",
                expected_ui_state="Start menu opened and search input visible",
            ))
            replanned.append(SubGoal(
                index=idx + 1,
                description=f"Type '{target_cmd}' into search",
                expected_ui_state=f"Search results for {target_cmd} displayed",
            ))
            replanned.append(SubGoal(
                index=idx + 2,
                description=f"Click {target_cmd} app icon in search results to launch application",
                expected_ui_state=f"{target_cmd} application window active",
            ))
            return replanned

        # Route 5: Generic strategic fallback
        clean_fallback = _sanitize_coordinates(failed_subgoal.description)
        replanned.append(SubGoal(
            index=idx,
            description=f"Visually locate and click target element for '{clean_fallback}'",
            expected_ui_state=failed_subgoal.expected_ui_state,
        ))
        return replanned

    def _replan_with_llm(
        self,
        goal: str,
        failed_subgoal: SubGoal,
        failure_reason: str,
        diagnostic: Optional[ReflectionDiagnostic] = None,
    ) -> List[SubGoal]:
        """Queries local VLA/LLM client for alternative strategic routes given failure diagnostics."""
        if not self._is_local_client_allowed():
            return []
        system_prompt = (
            "You are Layer 1 Cognitive Planner for AVCS Autonomous Desktop OS.\n"
            "A sub-goal failed during desktop execution. Synthesize an alternative strategic route.\n"
            "STRICT INVARIANT: NEVER output (x, y) coordinates. Only output semantic intent.\n"
            "Format your reply ONLY as a valid JSON array of sub-goals:\n"
            '[{"index": 1, "description": "...", "expected_ui_state": "..."}]'
        )
        diag_str = json.dumps(diagnostic.to_dict()) if diagnostic else failure_reason
        user_prompt = (
            f"Overall Goal: '{goal}'\n"
            f"Failed Sub-Goal: {json.dumps(failed_subgoal.to_dict())}\n"
            f"Failure Diagnostics: {diag_str}\n"
            "Provide alternative strategic route sub-goals to bypass or recover from this obstacle."
        )

        response = self.client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )

        text = response.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()

        data = json.loads(text)
        replanned: List[SubGoal] = []
        if isinstance(data, list):
            for i, item in enumerate(data, start=failed_subgoal.index):
                desc = _sanitize_coordinates(str(item.get("description", "")).strip())
                ui_state = _sanitize_coordinates(str(item.get("expected_ui_state", "")).strip())
                _validate_no_coordinates(desc)
                _validate_no_coordinates(ui_state)
                replanned.append(SubGoal(
                    index=i,
                    description=desc,
                    expected_ui_state=ui_state,
                ))
        return replanned
