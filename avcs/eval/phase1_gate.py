"""Phase 1 Gate 1 Verification & Baseline Leaderboard Publisher.

Validates all Phase 1 Completion Gate criteria from the Pure-RGB VLA Roadmap:
1. Deterministic JSON & dict round-trip across all 10 desktop action primitives.
2. Zero-error stateful mock AceAgent lifecycle round-trip (start -> plan -> reset -> close).
3. Showdown-Clicks evaluation (real dev CSV if available, else explicit cleanroom synthetic).
4. ScreenSpot-Pro dense high-res pure-RGB evaluation (with <24px icon recall).
5. OSWorld-Verified 30-task strictly pure-RGB smoke rollout (accessibility tree disabled).
6. Publishes structured Gate 1 baseline leaderboard JSON.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from avcs.eval.osworld_runner import (
    OSWorldObservationContract,
    generate_osworld_smoke_tasks,
    load_osworld_tasks,
    run_osworld_eval,
)
from avcs.eval.screenspot_pro import (
    generate_synthetic_screenspot_pro_suite,
    load_screenspot_pro,
    run_screenspot_pro,
)
from avcs.eval.showdown import (
    generate_synthetic_showdown_suite,
    load_showdown_clicks_dev,
    run_benchmark as run_showdown_benchmark,
)
from avcs.models import LocalVAMClient
from avcs.os_io.computer import Observation
from avcs.protocol.actions import Action, ActionKind
from avcs.runtime.agent import AceAgent


PHASE1_ACTION_PRIMITIVES: List[ActionKind] = [
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

# Published reference baselines cited directly from the Ace / Showdown-Clicks-Dev card.
# Kept explicitly separated from local measured runs under `published_literature_baselines`.
PUBLISHED_SHOWDOWN_BASELINES: List[Dict[str, Any]] = [
    {"model": "ace-control-medium", "accuracy": 77.56, "latency_ms": 533, "source": "published_vendor_report"},
    {"model": "ace-control-small", "accuracy": 72.89, "latency_ms": 324, "source": "published_vendor_report"},
    {"model": "Operator", "accuracy": 64.27, "latency_ms": 6385, "source": "published_vendor_report"},
    {"model": "UI-TARS-72B-SFT", "accuracy": 54.40, "latency_ms": 1977, "source": "published_vendor_report"},
    {"model": "Claude 3.7 Sonnet", "accuracy": 53.68, "latency_ms": 9656, "source": "published_vendor_report"},
    {"model": "OmniParser V2 + GPT-4o", "accuracy": 51.71, "latency_ms": 12642, "source": "published_vendor_report"},
    {"model": "Qwen2.5-VL-72B-Instruct", "accuracy": 24.78, "latency_ms": 3790, "source": "published_vendor_report"},
    {"model": "GPT-4o", "accuracy": 5.21, "latency_ms": None, "source": "published_vendor_report"},
]


def verify_protocol_roundtrip() -> Dict[str, Any]:
    """Verifies deterministic JSON serialization and parsing across all 10 desktop primitives."""
    sample_payloads = {
        ActionKind.LEFT_CLICK: Action(kind=ActionKind.LEFT_CLICK, coordinate=(640, 400), confidence=0.95),
        ActionKind.RIGHT_CLICK: Action(kind=ActionKind.RIGHT_CLICK, coordinate=(120, 45), confidence=0.9),
        ActionKind.DOUBLE_CLICK: Action(kind=ActionKind.DOUBLE_CLICK, coordinate=(300, 220), confidence=0.9),
        ActionKind.MOUSE_MOVE: Action(kind=ActionKind.MOUSE_MOVE, coordinate=(800, 500)),
        ActionKind.LEFT_CLICK_DRAG: Action(
            kind=ActionKind.LEFT_CLICK_DRAG,
            start_coordinate=(200, 300),
            end_coordinate=(700, 300),
            duration=0.4,
        ),
        ActionKind.SCROLL: Action(kind=ActionKind.SCROLL, coordinate=(640, 400), scroll_delta=(0, -120)),
        ActionKind.KEY: Action(kind=ActionKind.KEY, keys=["ctrl", "s"]),
        ActionKind.TYPE_TEXT: Action(kind=ActionKind.TYPE_TEXT, text="hello world", press_enter=True),
        ActionKind.WAIT: Action(kind=ActionKind.WAIT, duration=0.5, reason="spinner active"),
        ActionKind.STOP: Action(kind=ActionKind.STOP, reason="goal verified"),
    }

    verified = 0
    for kind in PHASE1_ACTION_PRIMITIVES:
        act = sample_payloads[kind]
        j1 = act.to_json()
        restored = Action.from_json(j1)
        j2 = restored.to_json()
        if j1 != j2 or restored.kind != kind:
            raise AssertionError(f"Non-deterministic round-trip for {kind.value}: {j1} != {j2}")
        verified += 1

    return {"passed": True, "primitives_verified": verified}


def verify_mock_agent_lifecycle() -> Dict[str, Any]:
    """Verifies zero-error start -> plan -> reset -> plan -> close lifecycle on offline mock agent."""
    img = Image.new("RGB", (1280, 800), color=(240, 240, 245))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    obs = Observation(
        image_bytes=buf.getvalue(),
        format="jpeg",
        width=1280,
        height=800,
        native_width=1280,
        native_height=800,
    )

    client = LocalVAMClient(endpoint="http://127.0.0.1:0/v1", model="phase1-offline-mock", allow_offline_mock=True)
    agent = AceAgent(client=client)  # type: ignore[arg-type]
    session = agent.start(instruction="Click Save button", max_steps=10, dry_run=True)

    act1 = session.plan(obs)
    if session.step_count != 1 or not isinstance(act1, Action):
        raise AssertionError(f"Expected step_count==1 after first plan(), got {session.step_count}")

    session.reset()
    if session.step_count != 0 or session.is_closed:
        raise AssertionError("Session reset() failed to clear step_count or reopen state")

    act2 = session.plan(obs)
    if session.step_count != 1 or not isinstance(act2, Action):
        raise AssertionError("Session plan() after reset() failed")

    session.close()
    if not session.is_closed:
        raise AssertionError("Session close() did not mark session closed")

    closed_rejected = False
    try:
        session.plan(obs)
    except RuntimeError:
        closed_rejected = True
    if not closed_rejected:
        raise AssertionError("Closed session must reject further plan() calls")

    return {"passed": True, "errors": 0, "closed_guard_verified": True}


def load_phase1_config(config_path: str = "configs/phase1_eval.yaml") -> Dict[str, Any]:
    """Loads Phase 1 evaluation config schema (PyYAML when available, stdlib fallback)."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        return {}
    raw_text = cfg_file.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]
        loaded = yaml.safe_load(raw_text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        # Lightweight fallback parser for nested Phase 1 YAML scalars
        root: Dict[str, Any] = {}
        stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
        for raw_line in raw_text.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip() or line.lstrip().startswith("- "):
                continue
            indent = len(line) - len(line.lstrip())
            if ":" not in line:
                continue
            key, val = [s.strip() for s in line.split(":", 1)]
            while len(stack) > 1 and stack[-1][0] >= indent:
                stack.pop()
            cur = stack[-1][1]
            if not val:
                child: Dict[str, Any] = {}
                cur[key] = child
                stack.append((indent, child))
            else:
                v_clean = val.strip("\"'")
                if v_clean.lower() == "true":
                    cur[key] = True
                elif v_clean.lower() == "false":
                    cur[key] = False
                else:
                    try:
                        cur[key] = float(v_clean) if "." in v_clean else int(v_clean)
                    except ValueError:
                        cur[key] = v_clean
        return root


def run_phase1_gate(
    output_path: str = "benchmarks/results/phase1_gate1_leaderboard.json",
    showdown_limit: int = 557,
    screenspot_limit: int = 30,
    osworld_tasks_limit: int = 30,
    smoke: bool = False,
    config_path: str = "configs/phase1_eval.yaml",
) -> Dict[str, Any]:
    """Runs complete Phase 1 Gate 1 checks and persists the baseline leaderboard artifact."""
    cfg = load_phase1_config(config_path)
    obs_cfg = cfg.get("observation", {}) if isinstance(cfg.get("observation"), dict) else {}
    bench_cfg = cfg.get("benchmarks", {}) if isinstance(cfg.get("benchmarks"), dict) else {}
    sd_cfg = bench_cfg.get("showdown_clicks", {}) if isinstance(bench_cfg.get("showdown_clicks"), dict) else {}
    ss_cfg = bench_cfg.get("screenspot_pro", {}) if isinstance(bench_cfg.get("screenspot_pro"), dict) else {}
    osw_cfg = bench_cfg.get("osworld_verified", {}) if isinstance(bench_cfg.get("osworld_verified"), dict) else {}

    tolerance_px = float(sd_cfg.get("tolerance_px", 15.0))
    if smoke:
        showdown_limit = min(showdown_limit, 30)
        screenspot_limit = min(screenspot_limit, 12)
        osworld_tasks_limit = min(osworld_tasks_limit, 30)

    proto_res = verify_protocol_roundtrip()
    lifecycle_res = verify_mock_agent_lifecycle()
    contract = OSWorldObservationContract(
        observation_mode=str(obs_cfg.get("mode", "pure_rgb")),
        use_a11y_tree=bool(obs_cfg.get("use_a11y_tree", False) or osw_cfg.get("disable_a11y_tree") is False),
        use_dom=bool(obs_cfg.get("use_dom", False) or osw_cfg.get("disable_dom") is False),
        use_ocr=bool(obs_cfg.get("use_ocr", False)),
    )

    client = LocalVAMClient(endpoint="http://127.0.0.1:0/v1", model="phase1-offline-baseline", allow_offline_mock=True)
    agent = AceAgent(client=client)  # type: ignore[arg-type]

    showdown_csv = Path(str(sd_cfg.get("data_path", "data/showdown-clicks-dev/data.csv")))
    if showdown_csv.exists():
        sd_samples = load_showdown_clicks_dev(str(showdown_csv), limit=showdown_limit)
        sd_source = str(showdown_csv)
    else:
        sd_samples = generate_synthetic_showdown_suite(count=min(showdown_limit, 50 if not smoke else 30))
        sd_source = "cleanroom_synthetic"
    sd_report = run_showdown_benchmark(agent=agent, samples=sd_samples, tolerance_px=tolerance_px)
    sd_report["dataset_source"] = sd_source
    sd_report.pop("results", None)

    ss_dir = Path(str(ss_cfg.get("data_dir", "data/screenspot-pro")))
    if ss_dir.exists() and list(ss_dir.glob("*.json*")):
        ss_samples = load_screenspot_pro(str(ss_dir), limit=screenspot_limit)
        ss_source = str(ss_dir)
    else:
        ss_samples = generate_synthetic_screenspot_pro_suite(count=screenspot_limit)
        ss_source = "cleanroom_synthetic"
    ss_report = run_screenspot_pro(agent=agent, samples=ss_samples, tolerance_px=float(ss_cfg.get("tolerance_px", tolerance_px)))
    ss_report["dataset_source"] = ss_source
    ss_report.pop("results", None)

    osw_dir = Path(str(osw_cfg.get("manifest_dir", "data/osworld-verified")))
    if osw_dir.exists() and list(osw_dir.rglob("*.json")):
        osw_tasks = load_osworld_tasks(str(osw_dir), limit=osworld_tasks_limit)
        osw_source = str(osw_dir)
    else:
        osw_tasks = generate_osworld_smoke_tasks(count=osworld_tasks_limit)
        osw_source = "cleanroom_synthetic_smoke"
    rollout_log = str(Path(output_path).parent / "phase1_osworld_smoke_rollouts.jsonl")
    osw_report = run_osworld_eval(
        agent=agent,
        tasks=osw_tasks,
        contract=contract,
        rollout_log_path=rollout_log,
        max_steps_override=int(osw_cfg.get("max_steps_per_task", 15)),
    )
    osw_report["dataset_source"] = osw_source
    osw_report.pop("tasks", None)

    harness_passed = bool(
        proto_res["passed"]
        and lifecycle_res["passed"]
        and osw_report["error_count"] == 0
        and osw_report["deterministic_reset_rate"] == 100.0
        and osw_report["env_flags"]["require_a11y_tree"] is False
    )
    gate_cfg = cfg.get("gate1", {}) if isinstance(cfg.get("gate1"), dict) else {}
    min_sd_acc = float(gate_cfg.get("min_showdown_accuracy", 0.65))
    sd_strict_acc = float(sd_report.get("strict_accuracy", 0.0))
    accuracy_gate_passed = bool(sd_strict_acc >= min_sd_acc)
    gate_passed = harness_passed if smoke else bool(harness_passed and accuracy_gate_passed)

    out = Path(output_path)
    existing_zero_shot: Optional[Dict[str, Any]] = None
    if out.exists():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
            existing_zero_shot = prev.get("measured_local", {}).get("zero_shot_baselines")
        except Exception:
            existing_zero_shot = None

    measured_local: Dict[str, Any] = {
        "showdown_clicks": sd_report,
        "screenspot_pro": ss_report,
        "osworld_verified_smoke": osw_report,
    }
    if isinstance(existing_zero_shot, dict) and existing_zero_shot:
        measured_local["zero_shot_baselines"] = existing_zero_shot

    leaderboard = {
        "phase": "phase1_gate1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate1_passed": gate_passed,
        "harness_smoke_passed": harness_passed,
        "accuracy_gate_passed": accuracy_gate_passed,
        "smoke": bool(smoke),
        "config_path": config_path if cfg else None,
        "observation_contract": contract.to_env_flags(),
        "checks": {
            "protocol_roundtrip": proto_res,
            "mock_agent_lifecycle": lifecycle_res,
        },
        "measured_local": measured_local,
        "published_literature_baselines": PUBLISHED_SHOWDOWN_BASELINES,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)
    return leaderboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 Gate 1 Verification & Publish Leaderboard JSON")
    parser.add_argument("--output", type=str, default="benchmarks/results/phase1_gate1_leaderboard.json")
    parser.add_argument("--showdown-limit", type=int, default=557)
    parser.add_argument("--screenspot-limit", type=int, default=30)
    parser.add_argument("--osworld-limit", type=int, default=30)
    parser.add_argument("--smoke", action="store_true", help="Run fast CI smoke sizes across all harnesses")
    args = parser.parse_args()

    res = run_phase1_gate(
        output_path=args.output,
        showdown_limit=args.showdown_limit,
        screenspot_limit=args.screenspot_limit,
        osworld_tasks_limit=args.osworld_limit,
        smoke=args.smoke,
    )
    print(f"PHASE1-GATE1: {'PASS' if res['gate1_passed'] else 'FAIL'} -> {args.output}")


if __name__ == "__main__":
    main()
