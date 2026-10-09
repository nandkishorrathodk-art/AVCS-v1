"""AVCS Universal Ecosystem Adapters.

Re-exports adapters for Anthropic Claude, Playwright, OpenAI Operator, and LangChain.
"""
from __future__ import annotations

from adapters.claude_computer_use import AVCSClaudeStreamFilter, StreamFilterStats
from adapters.playwright_eye import AVCSPlaywrightEye
from adapters.openai_operator import AVCSOperatorHarness, OperatorVisualObservation
from adapters.langchain_tool import AVCSScreenTools

__all__ = [
    "AVCSClaudeStreamFilter",
    "StreamFilterStats",
    "AVCSPlaywrightEye",
    "AVCSOperatorHarness",
    "OperatorVisualObservation",
    "AVCSScreenTools",
]

