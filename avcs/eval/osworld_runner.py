"""OSWorld-Verified Strictly Pure-RGB Evaluation Harness.

Provides deterministic task manifest loading, fail-closed pure-RGB observation
enforcement (accessibility tree and DOM modes disabled), deterministic environment
reset, multi-step AceAgentSession rollout recording, and summary telemetry for
Phase 1 Gate 1 (30-task smoke slice and 369-task OSWorld-Verified manifest).
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
from PIL import Image, ImageDraw

from avcs.eval.showdown import bootstrap_confidence_interval, wilson_score_interval
from avcs.os_io.computer import Observation
from avcs.protocol.actions import Action, ActionKind
from avcs.runtime.agent import AceAgent


@dataclasses.dataclass(frozen=True)
class OSWorldObservationContract:
    """Locks OSWorld observation configuration strictly to pure RGB pixels."""

    observation_mode: str = "pure_rgb"
    use_a11y_tree: bool = False
    use_dom: bool = False
    use_ocr: bool = False

    def __post_init__(self) -> None:
        if self.observation_mode != "pure_rgb":
            raise ValueError(
                f"OSWorld pure-RGB contract violation: observation_mode must be 'pure_rgb', got {self.observation_mode!r}"
            )
        if self.use_a11y_tree or self.use_dom or self.use_ocr:
            raise ValueError(
                "OSWorld pure-RGB contract violation: accessibility tree, DOM, and OCR observations are forbidden."
            )

    def to_env_flags(self) -> Dict[str, Any]:
        """Returns deterministic OSWorld VM launch flags enforcing screenshot-only observations."""
        return {
            "observation_type": "screenshot",
            "require_a11y_tree": False,
            "require_terminal": False,
            "use_dom": False,
            "use_ocr": False,
        }


@dataclasses.dataclass
class OSWorldTask:
    """Single OSWorld-Verified desktop task specification."""

    task_id: str
    app_domain: str
    instruction: str
    target_bbox: Tuple[int, int, int, int] = (540, 360, 740, 420)
    max_steps: int = 15
    expected_action: ActionKind = ActionKind.LEFT_CLICK
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


class OSWorldEnvAdapter(Protocol):
    """Abstract environment interface for pure-RGB OSWorld rollouts."""

    def reset(self, task: OSWorldTask) -> Observation:
        ...

    def step(self, action: Action) -> Tuple[Observation, bool, Dict[str, Any]]:
        ...

    def evaluate(self) -> float:
        ...


class DeterministicPureRGBMockEnv:
    """Deterministic offline pure-RGB desktop simulator for Gate 1 CI & smoke verification.

    Maintains canonical 1280x800 pure-RGB frames, validates reset determinism,
    records identical frame hashes for stall detection, and evaluates whether
    executed actions interacted cleanly with the task target region.
    """

    def __init__(self, width: int = 1280, height: int = 800) -> None:
        self.width = width
        self.height = height
        self._task: Optional[OSWorldTask] = None
        self._step_idx: int = 0
        self._task_completed: bool = False

    def _render_frame(self, task: OSWorldTask, completed: bool, step_idx: int) -> Observation:
        img = Image.new("RGB", (self.width, self.height), color=(236, 240, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (self.width, 42)], fill=(36, 41, 48))
        bx1, by1, bx2, by2 = task.target_bbox
        fill_color = (46, 160, 67) if completed else (31, 111, 235)
        draw.rectangle([(bx1, by1), (bx2, by2)], fill=fill_color, outline=(20, 45, 90), width=2)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return Observation(
            image_bytes=buf.getvalue(),
            format="jpeg",
            width=self.width,
            height=self.height,
            native_width=self.width,
            native_height=self.height,
        )

    def reset(self, task: OSWorldTask) -> Observation:
        self._task = task
        self._step_idx = 0
        self._task_completed = False
        return self._render_frame(task, completed=False, step_idx=0)

    def step(self, action: Action) -> Tuple[Observation, bool, Dict[str, Any]]:
        if self._task is None:
            raise RuntimeError("Environment must be reset(task) before step().")
        self._step_idx += 1

        if action.is_click() and action.coordinate is not None:
            px, py = action.coordinate
            bx1, by1, bx2, by2 = self._task.target_bbox
            if min(bx1, bx2) <= px <= max(bx1, bx2) and min(by1, by2) <= py <= max(by1, by2):
                self._task_completed = True
        elif action.kind == self._task.expected_action and action.kind in (ActionKind.KEY, ActionKind.TYPE, ActionKind.TYPE_TEXT, ActionKind.SCROLL):
            self._task_completed = True

        done = bool(action.kind == ActionKind.STOP or self._step_idx >= self._task.max_steps)
        obs = self._render_frame(self._task, completed=self._task_completed, step_idx=self._step_idx)
        return obs, done, {"completed": self._task_completed, "step_index": self._step_idx}

    def evaluate(self) -> float:
        return 1.0 if self._task_completed else 0.0


def load_osworld_tasks(
    manifest_dir: str = "data/osworld-verified",
    limit: Optional[int] = None,
) -> List[OSWorldTask]:
    """Loads OSWorld-Verified JSON task manifests from disk."""
    root = Path(manifest_dir)
    if not root.exists():
        raise FileNotFoundError(f"OSWorld-Verified manifest directory not found: {root}")

    json_files = sorted(root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No OSWorld task JSON files found under {root}")

    tasks: List[OSWorldTask] = []
    for jf in json_files:
        if limit is not None and limit > 0 and len(tasks) >= limit:
            break
        with open(jf, "r", encoding="utf-8") as f:
            payload = json.load(f)
        entries = payload if isinstance(payload, list) else [payload]
        for item in entries:
            if limit is not None and limit > 0 and len(tasks) >= limit:
                break
            bbox = tuple(int(v) for v in item.get("target_bbox", (540, 360, 740, 420))[:4])
            tasks.append(
                OSWorldTask(
                    task_id=str(item.get("id") or jf.stem),
                    app_domain=str(item.get("snapshot") or item.get("domain") or jf.parent.name or "os"),
                    instruction=str(item.get("instruction") or ""),
                    target_bbox=bbox,  # type: ignore[arg-type]
                    max_steps=int(item.get("max_steps", 15)),
                    metadata={"manifest_file": str(jf)},
                )
            )
    return tasks


def generate_osworld_smoke_tasks(count: int = 30) -> List[OSWorldTask]:
    """Generates a 30-task balanced smoke slice spanning all 6 OSWorld desktop domains."""
    domains = [
        ("chrome", "Open bookmark manager and select saved article", (180, 90, 320, 125)),
        ("vscode", "Click Run and Debug sidebar icon in VS Code", (12, 210, 48, 250)),
        ("gimp", "Select Crop tool on GIMP left toolbox", (18, 165, 52, 198)),
        ("libreoffice_calc", "Click Bold formatting icon on LibreOffice toolbar", (260, 72, 294, 102)),
        ("vlc", "Click Play button in VLC media player control bar", (36, 738, 76, 774)),
        ("os_io", "Click Save confirmation button in file dialog", (980, 690, 1090, 730)),
    ]
    tasks: List[OSWorldTask] = []
    for i in range(count):
        domain, instr, bbox = domains[i % len(domains)]
        tasks.append(
            OSWorldTask(
                task_id=f"osworld_smoke_{domain}_{i+1:03d}",
                app_domain=domain,
                instruction=f"{instr} (task #{i+1})",
                target_bbox=bbox,
                max_steps=6,
            )
        )
    return tasks


def run_osworld_eval(
    agent: AceAgent,
    tasks: List[OSWorldTask],
    env: Optional[OSWorldEnvAdapter] = None,
    contract: Optional[OSWorldObservationContract] = None,
    use_a11y_tree: bool = False,
    use_dom: bool = False,
    rollout_log_path: Optional[str] = None,
    max_steps_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs pure-RGB OSWorld rollouts with deterministic reset, stall tracking, and trace logging."""
    if contract is None:
        contract = OSWorldObservationContract(
            observation_mode="pure_rgb",
            use_a11y_tree=use_a11y_tree,
            use_dom=use_dom,
        )
    elif use_a11y_tree or use_dom:
        raise ValueError("Accessibility tree and DOM observations cannot be enabled in pure-RGB OSWorld runs.")

    active_env: OSWorldEnvAdapter = env or DeterministicPureRGBMockEnv()
    task_results: List[Dict[str, Any]] = []
    latencies: List[float] = []
    steps_taken_success: List[int] = []
    stall_events = 0
    error_count = 0

    log_file = Path(rollout_log_path) if rollout_log_path else None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_file, "w", encoding="utf-8")
    else:
        log_f = None

    try:
        for task in tasks:
            max_steps = max_steps_override or task.max_steps
            obs_init_1 = active_env.reset(task)
            obs_init_2 = active_env.reset(task)
            reset_deterministic = (
                hashlib.sha256(obs_init_1.image_bytes).hexdigest()
                == hashlib.sha256(obs_init_2.image_bytes).hexdigest()
            )
            obs = obs_init_2

            session = agent.start(instruction=task.instruction, max_steps=max_steps, dry_run=True)
            trace: List[Dict[str, Any]] = []
            prev_hashes: List[str] = []
            task_stall = False

            for step_idx in range(1, max_steps + 1):
                frame_hash = hashlib.sha256(obs.image_bytes).hexdigest()[:16]
                prev_hashes.append(frame_hash)
                if len(prev_hashes) >= 3 and len(set(prev_hashes[-3:])) == 1:
                    task_stall = True

                t0 = time.perf_counter()
                try:
                    action = session.plan(obs)
                except Exception as exc:
                    error_count += 1
                    action = Action(kind=ActionKind.STOP, reason=f"session_plan_error:{exc}")
                step_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(step_ms)

                obs_next, done, info = active_env.step(action)
                trace.append(
                    {
                        "step": step_idx,
                        "frame_hash": frame_hash,
                        "action": action.to_dict(),
                        "latency_ms": round(step_ms, 2),
                        "info": info,
                    }
                )
                obs = obs_next
                if done:
                    break

            session.close()
            reward = float(active_env.evaluate())
            success = reward >= 1.0
            if success:
                steps_taken_success.append(len(trace))
            if task_stall:
                stall_events += 1

            rec = {
                "task_id": task.task_id,
                "app_domain": task.app_domain,
                "observation_mode": contract.observation_mode,
                "a11y_tree_disabled": not contract.use_a11y_tree,
                "reset_deterministic": reset_deterministic,
                "success": success,
                "reward": reward,
                "steps": len(trace),
                "stall_detected": task_stall,
                "trace": trace,
            }
            task_results.append(rec)
            if log_f is not None:
                log_f.write(json.dumps(rec) + "\n")
    finally:
        if log_f is not None:
            log_f.close()

    total = len(task_results)
    successes = sum(1 for r in task_results if r["success"])
    w_low, w_high = wilson_score_interval(successes, total, confidence=0.95)
    b_low, b_high = bootstrap_confidence_interval([r["success"] for r in task_results], n_resamples=1000, confidence=0.95, seed=42)

    return {
        "benchmark": "osworld-verified",
        "observation_mode": contract.observation_mode,
        "env_flags": contract.to_env_flags(),
        "task_count": total,
        "successes": successes,
        "completion_rate": round((successes / max(1, total)) * 100.0, 2),
        "wilson_95_ci": [round(w_low * 100.0, 2), round(w_high * 100.0, 2)],
        "bootstrap_95_ci": [round(b_low * 100.0, 2), round(b_high * 100.0, 2)],
        "median_steps_to_completion": float(np.median(steps_taken_success)) if steps_taken_success else None,
        "stall_rate": round((stall_events / max(1, total)) * 100.0, 2),
        "deterministic_reset_rate": round(
            (sum(1 for r in task_results if r["reset_deterministic"]) / max(1, total)) * 100.0, 2
        ),
        "error_count": error_count,
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 1) if latencies else 0.0,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else 0.0,
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 1) if latencies else 0.0,
        "rollout_log_path": str(log_file) if log_file is not None else None,
        "tasks": task_results,
    }
