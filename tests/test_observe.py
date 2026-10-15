"""Tests for avcs.observe desktop observation narrative synthesis and CLI."""
from __future__ import annotations

import json
from unittest.mock import patch

from avcs.observe import (
    main,
    synthesize_natural_language_report,
)


def test_synthesize_report_with_windows_and_still_screen():
    mock_obs = {
        "desktop_context": {
            "status": "ok",
            "count": 3,
            "windows": [
                {
                    "title": "Document - Word",
                    "process_name": "WINWORD.EXE",
                    "is_foreground": True,
                },
                {
                    "title": "Chrome",
                    "process_name": "chrome.exe",
                    "is_foreground": False,
                },
            ],
        },
        "visual_perception": {
            "description": "A dark, flat screen.",
            "signals": {
                "peak_motion": 0.01,
                "changed_area_fraction": 0.005,
            },
        },
    }
    report = synthesize_natural_language_report(mock_obs)
    assert "3 visible window(s) open." in report
    assert "Foreground window: 'Document - Word' (WINWORD.EXE)." in report
    assert "Screen is essentially still. Static document, code editor, or idle desktop." in report
    assert "Motor actions authorized: False" in report


def test_synthesize_report_with_high_motion_game_media():
    mock_obs = {
        "desktop_context": {
            "status": "ok",
            "count": 5,
            "windows": [
                {
                    "title": "Elden Ring",
                    "process_name": "eldenring.exe",
                    "is_foreground": True,
                }
            ],
        },
        "visual_perception": {
            "description": "A bright, changing scene.",
            "signals": {
                "peak_motion": 0.28,
                "changed_area_fraction": 0.45,
            },
        },
    }
    report = synthesize_natural_language_report(mock_obs)
    assert "5 visible window(s) open." in report
    assert "Continuous high visual motion detected" in report
    assert "video playback or active gameplay" in report


def test_synthesize_report_without_os_context():
    mock_obs = {
        "desktop_context": {
            "status": "not_configured",
        },
        "visual_perception": {
            "description": "A neutral desktop view.",
            "signals": {
                "peak_motion": 0.0,
                "changed_area_fraction": 0.0,
            },
        },
    }
    report = synthesize_natural_language_report(mock_obs)
    assert "OS window context: not available or not configured." in report
    assert "Screen is essentially still" in report


def test_observe_cli_json_flag(capsys):
    dummy_obs = {
        "schema_version": "avcs.desktop_observation.v1",
        "desktop_context": {"status": "ok", "count": 1, "windows": []},
        "visual_perception": {"description": "test", "signals": {}},
    }
    with patch("avcs.observe.capture_desktop_observation", return_value=dummy_obs):
        code = main(["--json"])
        assert code == 0
        captured = capsys.readouterr()
        loaded = json.loads(captured.out)
        assert loaded["schema_version"] == "avcs.desktop_observation.v1"
