"""End-to-End System 2 (LLM) + System 1 (AVCS Eyes & Hands) Live Closed-Loop Demo.

Re-exports and launches the canonical demo in avcs.demo_system2.
"""
from __future__ import annotations

from avcs.demo_system2 import main, run_avcs_system2_demo

__all__ = ["main", "run_avcs_system2_demo"]


if __name__ == "__main__":
    main()
