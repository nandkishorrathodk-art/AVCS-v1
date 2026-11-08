"""Cognitive Intelligence & Sub-Goal Planner (Layer 1).

Decouples high-level strategic reasoning from low-level motor actuation:
1. Operates at low frequency (0.5 - 1.0 Hz).
2. Ingests multilingual user instructions (Hindi, English, Hinglish).
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
    """Enforces strict user requirement: SubGoal must be strictly in English (no Hindi/Devanagari text)."""
    if not text:
        return
    if re.search(r'[\u0900-\u097F]', text):
        raise ValueError(
            f"Invariant violation: SubGoal must be in English only (no Hindi text permitted per user requirements). "
            f"Found Devanagari script in text: {text!r}"
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
    """Translates arbitrary Hindi/Hinglish text to clean English description and expected UI state."""
    translations = [
        (r'(?<!\S)chrome\s+(?:band\s*karo|close\s*karo|khatam\s*karo)(?!\S)', 'Close Google Chrome application'),
        (r'(?<!\S)notepad\s+(?:band\s*karo|close\s*karo)(?!\S)', 'Close Notepad application'),
        (r'(?<!\S)(?:band\s*karo|band\s*kar\s*do|close\s*karo|हटाओ|बंद\s*करो|बंद\s*करें)(?!\S)', 'close active window'),
        (r'(?<!\S)(?:kholo|khol\s*do|kholiye|खोलो|खोलें|खोलिए)(?!\S)', 'open application'),
        (r'(?<!\S)(?:chalao|chalaayein|start\s*karo|चलाओ|शुरू\s*करो)(?!\S)', 'launch application'),
        (r'(?<!\S)(?:dhoondo|khojo|search\s*karo|ढूंढो|खोजो|सर्च\s*करो)(?!\S)', 'search query'),
        (r'(?<!\S)(?:likho|darj\s*karo|type\s*karo|लिखो|टाइप\s*करो|दर्ज\s*करो)(?!\S)', 'type text'),
        (r'(?<!\S)(?:dabao|press\s*karo|click\s*karo|दबाओ|क्लिक\s*करो)(?!\S)', 'interact with element'),
        (r'(?<!\S)(?:bajaao|play\s*karo|बजाओ)(?!\S)', 'play media'),
        (r'(?<!\S)(?:save\s*karo|seve\s*karo|सेव\s*करो|सहेजें)(?!\S)', 'save file'),
        (r'(?<!\S)(?:refresh\s*karo|taaza\s*karo|रिफ्रेश)(?!\S)', 'refresh view'),
        (r'(?<!\S)(?:gaana|music|song|गाना|संगीत)(?!\S)', 'music'),
        (r'(?<!\S)(?:video|वीडियो)(?!\S)', 'video'),
        (r'(?<!\S)(?:tasveer|photo|image|तस्वीर|फोटो)(?!\S)', 'image'),
        (r'(?<!\S)(?:file|document|फ़ाइल|फाइल|दस्तावेज़)(?!\S)', 'document'),
        (r'(?<!\S)(?:panna|page|पेज|पन्ना)(?!\S)', 'page'),
        (r'(?<!\S)(?:karo|kijiye|करो|करें|कीजिए)(?!\S)', ''),
        (r'(?<!\S)(?:par|me|पर|में)(?!\S)', 'on'),
        (r'(?<!\S)(?:aur|और)(?!\S)', 'and'),
        (r'(?<!\S)(?:fir|phir|फिर)(?!\S)', 'then'),
    ]
    s = text
    for pat, repl in translations:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    # Strip any residual Devanagari characters
    s = re.sub(r'[\u0900-\u097F]+', 'action', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if not s or s == 'action':
        s = "Execute target desktop action"
    desc = s[0].upper() + s[1:]
    expected = f"Completed '{desc}'"
    return desc, expected


@dataclass
class SubGoal:
    """Atomic semantic sub-goal directive emitted by Layer 1 Cognitive Planner.

    Strict Invariants:
    1. NEVER contains low-level (x, y) coordinates.
    2. Strictly English only (no Hindi/Devanagari text permitted per user requirements).
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

    def decompose(self, goal: str) -> List[SubGoal]:
        """Decomposes a multilingual user goal into an ordered sequence of atomic sub-goals.

        Supports English, Hindi, and Hinglish.
        Guarantees strict invariant: NEVER outputs low-level (x, y) coordinates.
        """
        self._apply_rate_limit()
        goal = (goal or "").strip()
        if not goal:
            return []
        goal = _sanitize_coordinates(goal)

        # 1. Attempt LLM-assisted decomposition if client is provided and online
        if self.client is not None:
            try:
                llm_subgoals = self._decompose_with_llm(goal)
                if llm_subgoals:
                    return llm_subgoals
            except Exception as e:
                logger.warning(f"CognitivePlanner LLM decomposition fallback to deterministic engine: {e}")

        # 2. Deterministic Multilingual Parser (Hindi, English, Hinglish)
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

        # 1. Attempt LLM-assisted dynamic replanning if client is provided
        if self.client is not None:
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
        """Splits composite instructions on multilingual sequential delimiters."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Delimiters across Hindi (Devanagari), Hinglish (Latin), and English:
        pattern = (
            r'\s*(?:'
            r'[,\u0964]|(?<=[a-zA-Z0-9\u0900-\u097F])\.\s+|'
            r';|\->|\||'
            r'\band\s+then\b|\bafter\s+that\b|\band\b|\bthen\b|'
            r'\baur\s+phir\b|\baur\s+fir\b|\baur\b|\bphir\b|\bfir\b|'
            r'\bkhol\s*kar\s*ke\b|\bkhol\s*kar\b|\bkhol\s*ke\b|\bkholkar\b|\bkholke\b|'
            r'\bkar\s*ke\b|\bke\s+baad\b|'
            r'और\s+फिर|और|फिर|के\s+बाद|\s+तथा\s+|\s+व\s+|\bतथा\b|'
            r'खोल\s*कर\s*के|खोलकर|खोल\s*कर|खोलके|खोल\s*के'
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

        # Known application keywords and their clean canonical names (English + Hindi Devanagari)
        app_catalog = {
            "notepad": ("Notepad", "Notepad application window active and editor focused"),
            "नोटपैड": ("Notepad", "Notepad application window active and editor focused"),
            "calculator": ("Calculator", "Calculator application window active"),
            "calc": ("Calculator", "Calculator application window active"),
            "कैलकुलेटर": ("Calculator", "Calculator application window active"),
            "कैलक": ("Calculator", "Calculator application window active"),
            "chrome": ("Google Chrome", "Google Chrome window active and address bar visible"),
            "google chrome": ("Google Chrome", "Google Chrome window active and address bar visible"),
            "क्रोम": ("Google Chrome", "Google Chrome window active and address bar visible"),
            "गूगल क्रोम": ("Google Chrome", "Google Chrome window active and address bar visible"),
            "edge": ("Microsoft Edge", "Microsoft Edge window active"),
            "browser": ("Web Browser", "Browser window active"),
            "ब्राउज़र": ("Web Browser", "Browser window active"),
            "ब्राउजर": ("Web Browser", "Browser window active"),
            "paint": ("Paint", "Paint application canvas ready"),
            "पेंट": ("Paint", "Paint application canvas ready"),
            "explorer": ("File Explorer", "File Explorer window open"),
            "file explorer": ("File Explorer", "File Explorer window open"),
            "फ़ाइल एक्सप्लोरर": ("File Explorer", "File Explorer window open"),
            "एक्सप्लोरर": ("File Explorer", "File Explorer window open"),
            "cmd": ("Command Prompt", "Command prompt window open"),
            "terminal": ("Terminal", "Terminal window open"),
            "टर्मिनल": ("Terminal", "Terminal window open"),
            "powershell": ("PowerShell", "PowerShell window open"),
            "settings": ("Windows Settings", "Settings window open"),
            "सेटिंग्स": ("Windows Settings", "Settings window open"),
            "सेटिंग": ("Windows Settings", "Settings window open"),
            "start menu": ("Windows Start Menu", "Start menu opened and search input visible"),
            "start": ("Windows Start Menu", "Start menu opened and search input visible"),
            "स्टार्ट मेन्यू": ("Windows Start Menu", "Start menu opened and search input visible"),
            "स्टार्ट": ("Windows Start Menu", "Start menu opened and search input visible"),
            "task manager": ("Task Manager", "Task Manager window open"),
            "word": ("Microsoft Word", "Word document editor open"),
            "वर्ड": ("Microsoft Word", "Word document editor open"),
            "excel": ("Microsoft Excel", "Excel spreadsheet active"),
            "एक्सेल": ("Microsoft Excel", "Excel spreadsheet active"),
            "spotify": ("Spotify", "Spotify application window open"),
            "youtube": ("YouTube", "YouTube application or browser tab active"),
            "यूट्यूब": ("YouTube", "YouTube application or browser tab active"),
            "google": ("Google", "Google search interface visible"),
            "गूगल": ("Google", "Google search interface visible"),
        }

        idx = 1
        for clause in clauses:
            raw_clean = clause.strip()
            # Enforce strict invariant by sanitizing any low-level coordinates into semantic directives
            clean = _sanitize_coordinates(raw_clean)
            lower = clean.lower()

            # Pattern A: Start menu opening (Latin & Devanagari)
            if re.search(r'\b(?:start\s*menu|start)\b|स्टार्ट\s*मेन्यू|स्टार्ट', clean, re.IGNORECASE):
                if re.search(r'(?:open|kholo|start|launch|press|खोलो|खोलें|खोलिए|दबाओ|चालू)', clean, re.IGNORECASE) or lower in ("start", "start menu", "स्टार्ट", "स्टार्ट मेन्यू"):
                    subgoals.append(SubGoal(
                        index=idx,
                        description="Open Windows Start Menu",
                        expected_ui_state="Start menu opened and search input visible",
                    ))
                    idx += 1
                    continue

            # Pattern B0: Close / Exit application or window (must precede launch pattern)
            # e.g., "Chrome band karo", "Notepad close karo", "window band kar do", "विंडो बंद करो", "नोटपैड बंद करो"
            close_verbs = [
                "close", "band karo", "band kar do", "band kijiye", "band", "exit", "quit", "hatao",
                "बंद करो", "बंद करें", "बंद कर दें", "बंद कीजिए", "हटाओ", "बंद"
            ]
            if any(cv in lower for cv in close_verbs):
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
            # e.g., "Google par OpenAI search karo", "search for OpenAI on Google", "गूगल पर OpenAI सर्च करो"
            search_match = re.search(
                r'^\s*(?:search|dhoondo|find)\s+(?:for\s+)?([a-zA-Z0-9_\-\.\s\u0900-\u097F]+?)(?:\s+(?:on|in|par|me|पर|में)\s+([a-zA-Z0-9_\-\.\s\u0900-\u097F]+))?$',
                clean,
                re.IGNORECASE,
            )
            hinglish_search = re.search(
                r'([a-zA-Z0-9_\-\.\s\u0900-\u097F]+?)\s+(?:par|me|पर|में)\s+([a-zA-Z0-9_\-\.\s\u0900-\u097F]+?)\s+(?:search|dhoondo|सर्च|खोज|ढूंढो)\s*(?:karo|kijiye|करो|करें|कीजिए)?',
                clean,
                re.IGNORECASE,
            )

            if hinglish_search:
                platform_raw = hinglish_search.group(1).strip()
                query_raw = hinglish_search.group(2).strip()
                platform = "Google" if "गूगल" in platform_raw or "google" in platform_raw.lower() else platform_raw.title()
                query = re.sub(r'[\u0900-\u097F]+', 'target query', query_raw)
                subgoals.append(SubGoal(
                    index=idx,
                    description=f"Search for '{query}' on {platform}",
                    expected_ui_state=f"Search results for '{query}' displayed on {platform}",
                ))
                idx += 1
                continue
            elif search_match and search_match.group(1).strip().lower() not in ("karo", "kijiye", "करो", "करें"):
                query_raw = search_match.group(1).strip()
                platform_raw = search_match.group(2).strip() if search_match.group(2) else ""
                platform = "Google" if "गूगल" in platform_raw or "google" in platform_raw.lower() else platform_raw.title()
                query = re.sub(r'[\u0900-\u097F]+', 'target query', query_raw)
                dest = f" on {platform}" if platform else ""
                subgoals.append(SubGoal(
                    index=idx,
                    description=f"Search for '{query}'{dest}",
                    expected_ui_state=f"Search results for '{query}' displayed{dest}",
                ))
                idx += 1
                continue

            # Pattern B2: Direct search: e.g. "Notepad search karo", "नोटपैड सर्च करो", "Notepad search karo open karo"
            direct_search = re.search(
                r'([a-zA-Z0-9_\-\.\s\u0900-\u097F]+?)\s+(?:search|dhoondo|khojo|सर्च|खोज|ढूंढो)\s*(?:karo|kar\s*do|kijiye|chalayein|करो|करें|कीजिए)?(?:\s+(?:open\s*karo|khol\s*do|kholo|chalao|chalayein|launch\s*karo|open))?$',
                clean,
                re.IGNORECASE,
            )
            if direct_search:
                query = direct_search.group(1).strip()
                if query and query.lower() not in ("aur", "fir", "phir", "and", "और", "फिर"):
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
                        clean_query = re.sub(r'[\u0900-\u097F]+', 'target query', query)
                        subgoals.append(SubGoal(
                            index=idx,
                            description=f"Search for '{clean_query}'",
                            expected_ui_state=f"Search results for '{clean_query}' displayed",
                        ))
                    idx += 1
                    continue

            # Pattern C: Application launch / open (Latin & Devanagari)
            matched_app = None
            for app_key, (app_name, expected_ui) in app_catalog.items():
                if re.search(r'(?:\b|^)' + re.escape(app_key) + r'(?:\b|$)', lower):
                    matched_app = (app_name, expected_ui)
                    break

            launch_verbs = ["open", "launch", "start", "run", "kholo", "khol do", "kholiye", "chalao", "chalayein", "chalaayein", "kar do", "खोलो", "खोलें", "खोलिए", "चलाओ", "चलाएं", "शुरू"]
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

            # Pattern D1: SOV Typing text (Hindi/Hinglish: [payload] [type karo/likho/दर्ज करो])
            sov_type_m = re.search(
                r'^\s*(.+?)\s+(?:type\s*karo|type\s*kar\s*do|type\s*kijiye|type\s*karein|type|likho|likh\s*do|likhein|darj\s*karo|enter\s*karo|टाइप\s*करो|टाइप\s*करें|लिखो|लिखें|दर्ज\s*करो)\s*$',
                clean,
                re.IGNORECASE,
            )
            if sov_type_m and sov_type_m.group(1).strip().lower() not in ("aur", "and", "fir", "phir", "open", "kholo", "start", "close", "band"):
                text_to_type = sov_type_m.group(1).strip()
                if (text_to_type.startswith('"') and text_to_type.endswith('"')) or (text_to_type.startswith("'") and text_to_type.endswith("'")):
                    text_to_type = text_to_type[1:-1].strip()
                clean_type_text = re.sub(r'[\u0900-\u097F]+', 'target text', text_to_type)
                subgoals.append(SubGoal(
                    index=idx,
                    description=f"Type text '{clean_type_text}' into active target field",
                    expected_ui_state=f"Text '{clean_type_text}' visible in target field",
                ))
                idx += 1
                continue

            # Pattern D2: SVO Typing text (English: type [payload])
            type_m = re.search(
                r'^\s*(?:type|write|likho|enter|टाइप\s*करो|टाइप|लिखो|दर्ज)\s+(?:karo|kar\s*do|kijiye|chalayein|करो|करें|कीजिए)?\s*(.+)',
                clean,
                re.IGNORECASE,
            )
            if type_m:
                text_to_type = type_m.group(1).strip()
                text_to_type = re.sub(r'\s+(?:karo|kar\s*do|kijiye|chalayein|करो|करें|कीजिए)$', '', text_to_type, flags=re.IGNORECASE).strip()
                if (text_to_type.startswith('"') and text_to_type.endswith('"')) or (text_to_type.startswith("'") and text_to_type.endswith("'")):
                    text_to_type = text_to_type[1:-1].strip()
                clean_type_text = re.sub(r'[\u0900-\u097F]+', 'target text', text_to_type)
                subgoals.append(SubGoal(
                    index=idx,
                    description=f"Type text '{clean_type_text}' into active target field",
                    expected_ui_state=f"Text '{clean_type_text}' visible in target field",
                ))
                idx += 1
                continue

            # Pattern E: Calculation
            calc_m = re.search(r'(?:calculate|hisab|गणना|हिसाब)\s*(.+)?', clean, re.IGNORECASE)
            if any(k in lower for k in ["calculate", "hisab", "गणना", "हिसाब"]):
                expr = clean
                for w in ["calculate", "hisab karo", "hisab", "karo", "गणना करो", "गणना", "हिसाब"]:
                    expr = expr.replace(w, "")
                expr = re.sub(r'[\u0900-\u097F]+', '', expr).strip()
                subgoals.append(SubGoal(
                    index=idx,
                    description=f"Calculate {expr or 'expression'}",
                    expected_ui_state="Calculation result displayed",
                ))
                idx += 1
                continue

            # Pattern F: Save file
            if any(k in lower for k in ["save", "save karo", "सेव करो", "सेव", "सहेजें"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Save document or current file",
                    expected_ui_state="File saved confirmation dialog or status",
                ))
                idx += 1
                continue

            # Pattern G: Media playback (audio, song, video)
            if any(k in lower for k in ["play", "bajaao", "बजाओ", "गाना", "video chalao", "वीडियो चलाओ", "music", "song", "वीडियो"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Play media or audio playback",
                    expected_ui_state="Media player or browser playback active",
                ))
                idx += 1
                continue

            # Pattern H: Refresh page or view
            if any(k in lower for k in ["refresh", "reload", "रिफ्रेश", "रीलोड"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Refresh current page or view",
                    expected_ui_state="Page reloaded with refreshed content",
                ))
                idx += 1
                continue

            # Pattern I: Window minimize / maximize
            if any(k in lower for k in ["minimize", "chhota karo", "छोटा करो"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Minimize active window",
                    expected_ui_state="Active window minimized to taskbar",
                ))
                idx += 1
                continue
            if any(k in lower for k in ["maximize", "bada karo", "बड़ा करो"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Maximize active window",
                    expected_ui_state="Active window maximized to full screen",
                ))
                idx += 1
                continue

            # Pattern J: Clipboard copy / paste / select all
            if any(k in lower for k in ["copy", "copy karo", "कॉपी करो"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Copy selected content",
                    expected_ui_state="Selected text or element copied to clipboard",
                ))
                idx += 1
                continue
            if any(k in lower for k in ["paste", "paste karo", "पेस्ट करो"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Paste clipboard content",
                    expected_ui_state="Clipboard content pasted into active field",
                ))
                idx += 1
                continue
            if any(k in lower for k in ["select all", "sab select", "सब सेलेक्ट"]):
                subgoals.append(SubGoal(
                    index=idx,
                    description="Select all content",
                    expected_ui_state="All items or text in active view selected",
                ))
                idx += 1
                continue

            # Default fallback for arbitrary clause (guaranteed translated to pure English)
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
        """Calls frontier LLM via CustomLLMClient to produce coordinate-free sub-goals."""
        system_prompt = (
            "You are Layer 1 Cognitive Intelligence for AVCS Autonomous Desktop OS. "
            "Decompose user goals (Hindi, English, Hinglish) into an ordered sequence of atomic sub-goals.\n"
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
        if "```" in text:
            parts = text.split("```")
            for p in parts:
                if p.startswith("json"):
                    text = p[4:].strip()
                    break
                elif p.strip().startswith("["):
                    text = p.strip()
                    break

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

        # Route 1: Loop deadlock or modal alert blocking UI (repetition >= 2)
        if stuck_reps >= 2 or "loop" in reason_lower or "stuck" in reason_lower:
            clean_desc = failed_subgoal.description
            while True:
                m = re.match(r"^Retry\s*['\"](.*?)['\"](?:\s+(?:visually|using\s+[\w\s]+))?$", clean_desc, re.IGNORECASE)
                if m:
                    clean_desc = m.group(1).strip()
                else:
                    break

            replanned.append(SubGoal(
                index=idx,
                description="Click blank area of desktop or active window to dismiss temporary overlay and refocus",
                expected_ui_state="Screen stabilized and active workspace refocused",
            ))
            replanned.append(SubGoal(
                index=idx + 1,
                description=f"Retry '{clean_desc}' visually",
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
                m_goal = re.search(r'(?:open|launch|kholo|chalao|start)\s+([a-zA-Z0-9_\-\.]+)', goal, re.IGNORECASE)
                if not m_goal:
                    m_goal = re.search(r'([a-zA-Z0-9_\-\.]+)\s+(?:open|launch|kholo|chalao|start)', goal, re.IGNORECASE)
                if m_goal and m_goal.group(1).lower() not in ("start", "menu", "karo", "do", "kijiye", "aur", "and"):
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
        """Queries frontier LLM for alternative strategic routes given failure diagnostics."""
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
        if "```" in text:
            parts = text.split("```")
            for p in parts:
                if p.startswith("json"):
                    text = p[4:].strip()
                    break
                elif p.strip().startswith("["):
                    text = p.strip()
                    break

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
