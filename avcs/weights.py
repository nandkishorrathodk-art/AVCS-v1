"""AVCS Pre-Trained Weights Management.

Handles automatic discovery, local caching, and loading of pre-trained
Artificial Visual Cortex neural network checkpoints for both ScreenEye (ArtificialVisualCortex)
and HumanEye (RealTimeCorticalBackbone + FastAffordanceHead + FovealAcuityEngine).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional
import urllib.request

import torch

# Standard release checkpoint URL on Parth-labs-in repository
DEFAULT_WEIGHTS_URL = (
    "https://github.com/Parth-labs-in/AVCS/releases/download/v0.2.0/avcs_cortex_weights.pt"
)

# Standard filenames
WEIGHTS_FILENAME = "avcs_cortex_weights.pt"
HUMAN_EYE_WEIGHTS_FILENAME = "human_eye_weights.pt"
GOAL_GROUNDER_WEIGHTS_FILENAME = "goal_grounder_best.pt"


def get_default_cache_dir() -> Path:
    """Returns the user cache directory for AVCS neural weights."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        cache_dir = Path(base) / "AVCS" / "weights"
    else:
        cache_dir = Path(os.path.expanduser("~/.cache/avcs/weights"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_weights_path(custom_path: Optional[str] = None, filename: Optional[str] = None) -> Optional[str]:
    """Finds the path to pre-trained AVCS weights.

    Search priority:
    1. Explicit custom_path argument
    2. AVCS_WEIGHTS_PATH environment variable
    3. Local workspace runs directory: `runs/weights/{filename}`
    4. Bundled package directory (if installed)
    5. User cache directory: `~/.cache/avcs/weights/{filename}`
    """
    target_filename = filename or WEIGHTS_FILENAME

    if custom_path and os.path.isfile(custom_path):
        return custom_path

    env_var = "AVCS_HUMAN_EYE_WEIGHTS_PATH" if target_filename == HUMAN_EYE_WEIGHTS_FILENAME else "AVCS_WEIGHTS_PATH"
    env_path = os.environ.get(env_var) or (os.environ.get("AVCS_WEIGHTS_PATH") if target_filename == WEIGHTS_FILENAME else None)
    if env_path and os.path.isfile(env_path):
        return env_path

    # Workspace runs/ directory
    workspace_path = Path(__file__).resolve().parent.parent / "runs" / "weights" / target_filename
    if workspace_path.is_file():
        return str(workspace_path)

    # Relative to current working directory
    cwd_path = Path.cwd() / "runs" / "weights" / target_filename
    if cwd_path.is_file():
        return str(cwd_path)

    # Bundled within avcs package
    bundled_path = Path(__file__).resolve().parent / target_filename
    if bundled_path.is_file():
        return str(bundled_path)

    # User cache directory
    cached_path = get_default_cache_dir() / target_filename
    if cached_path.is_file():
        return str(cached_path)

    return None


def get_human_eye_weights_path(custom_path: Optional[str] = None) -> Optional[str]:
    """Finds the path to pre-trained HumanEye cortical affordance & foveal weights."""
    return get_weights_path(custom_path=custom_path, filename=HUMAN_EYE_WEIGHTS_FILENAME)


def get_grounding_weights_path(custom_path: Optional[str] = None) -> Optional[str]:
    """Finds the path to pre-trained GoalConditionedGrounder neural weights."""
    return get_weights_path(custom_path=custom_path, filename=GOAL_GROUNDER_WEIGHTS_FILENAME)


def download_weights(
    dest_path: Optional[str] = None,
    url: str = DEFAULT_WEIGHTS_URL,
    verbose: bool = True,
) -> str:
    """Downloads pre-trained AVCS weights from the official release repository."""
    target = Path(dest_path) if dest_path else get_default_cache_dir() / WEIGHTS_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[AVCS] Downloading pre-trained weights from {url} to {target}...")

    urllib.request.urlretrieve(url, str(target))

    if verbose:
        print(f"[AVCS] Download complete ({target.stat().st_size:,} bytes).")

    return str(target)


def _is_human_eye_model(cortex: Any) -> bool:
    """Detects whether model is a HumanEye instance or contains/is HumanEye submodules."""
    cls_name = cortex.__class__.__name__
    if cls_name in (
        "HumanEye",
        "RealTimeCorticalBackbone",
        "FastAffordanceHead",
        "FovealAcuityHead",
        "FovealAcuityEngine",
    ):
        return True
    if hasattr(cortex, "backbone") and cortex.backbone.__class__.__name__ == "RealTimeCorticalBackbone":
        return True
    if hasattr(cortex, "affordance_head") and cortex.affordance_head.__class__.__name__ == "FastAffordanceHead":
        return True
    if hasattr(cortex, "fovea_head") and cortex.fovea_head.__class__.__name__ == "FovealAcuityHead":
        return True
    if hasattr(cortex, "fovea_engine") and hasattr(cortex.fovea_engine, "head"):
        return True
    return False


def _load_complete_state_dict(
    module: torch.nn.Module,
    state: dict[str, Any],
    strict: bool,
    allowed_missing_prefixes: tuple[str, ...] = (),
) -> bool:
    """Load a complete, shape-compatible module state before reporting readiness."""
    model_state = module.state_dict()
    model_keys = set(model_state)
    checkpoint_keys = set(state)
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    if (
        (strict and (missing_keys or unexpected_keys))
        or any(
            not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
            for key in missing_keys
        )
    ):
        return False

    for key, expected in model_state.items():
        if key not in state:
            continue
        actual = state[key]
        if not torch.is_tensor(actual) or actual.shape != expected.shape:
            return False

    module.load_state_dict(
        {key: state[key] for key in model_state if key in state},
        strict=strict,
    )
    return True


def load_avcs_weights(
    cortex: Any,
    path: Optional[str] = None,
    device: str = "cpu",
    strict: bool = False,
    auto_download: bool = False,
) -> bool:
    """Loads pre-trained weights into the cortex model, HumanEye model, or its submodules.

    Returns True if weights were loaded, False otherwise.
    """
    is_human_eye = _is_human_eye_model(cortex)
    resolved_path = None

    if path:
        if os.path.isfile(path):
            resolved_path = path
        else:
            resolved_path = None
    elif is_human_eye:
        resolved_path = get_human_eye_weights_path()
        if resolved_path is None:
            resolved_path = get_weights_path()
    else:
        resolved_path = get_weights_path()

    if resolved_path is None and auto_download and not is_human_eye:
        try:
            resolved_path = download_weights()
        except Exception as exc:
            print(f"[AVCS] Notice: Could not download weights ({exc}). Proceeding with baseline weights.")
            return False

    if resolved_path and os.path.isfile(resolved_path):
        try:
            try:
                state = torch.load(resolved_path, map_location=device, weights_only=False)
            except TypeError:
                state = torch.load(resolved_path, map_location=device)

            if not isinstance(state, dict):
                return False

            # The fovea engine wraps its head; other modules must load as a whole.
            target_module = (
                cortex.head
                if (
                    cortex.__class__.__name__ == "FovealAcuityEngine"
                    and isinstance(cortex.head, torch.nn.Module)
                )
                else cortex
            )

            target_cls = target_module.__class__.__name__

            # Direct submodule state dictionary extraction
            if target_cls == "FovealAcuityHead" and "fovea_head" in state and isinstance(state["fovea_head"], dict):
                loaded = _load_complete_state_dict(target_module, state["fovea_head"], strict)
                if hasattr(cortex, "weights_loaded"):
                    cortex.weights_loaded = loaded
                return loaded
            elif target_cls == "FastAffordanceHead" and "affordance_head" in state and isinstance(state["affordance_head"], dict):
                loaded = _load_complete_state_dict(target_module, state["affordance_head"], strict)
                if hasattr(cortex, "weights_loaded"):
                    cortex.weights_loaded = loaded
                return loaded
            elif target_cls == "RealTimeCorticalBackbone" and "backbone" in state and isinstance(state["backbone"], dict):
                loaded = _load_complete_state_dict(target_module, state["backbone"], strict)
                if hasattr(cortex, "weights_loaded"):
                    cortex.weights_loaded = loaded
                return loaded

            optional_missing_prefixes = (
                ("grounder.",) if target_cls == "HumanEye" and not strict else ()
            )
            loaded = _load_complete_state_dict(
                target_module,
                state,
                strict,
                allowed_missing_prefixes=optional_missing_prefixes,
            )

            # If cortex is HumanEye, also ensure fovea_engine.head is populated
            if hasattr(cortex, "fovea_engine") and hasattr(cortex.fovea_engine, "head"):
                fov_head = cortex.fovea_engine.head
                fov_dict = state.get("fovea_head")
                if isinstance(fov_dict, dict):
                    cortex.fovea_engine.weights_loaded = _load_complete_state_dict(
                        fov_head, fov_dict, strict
                    )

            return loaded
        except Exception as exc:
            print(f"[AVCS] Warning: Failed to load weights from {resolved_path}: {exc}")
            return False

    return False
