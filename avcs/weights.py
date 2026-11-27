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
COMBINED_GROUNDER_WEIGHTS_FILENAME = "goal_grounder_combined_best.pt"
AVCS_V2_WEIGHTS_FILENAME = "avcs_v2_weights.pt"
OBJECTNESS_WEIGHTS_FILENAME = "avcs_v2_weights.pt"


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
    if custom_path and os.path.isfile(custom_path):
        return custom_path
    combined_path = get_weights_path(filename=COMBINED_GROUNDER_WEIGHTS_FILENAME)
    if combined_path and os.path.isfile(combined_path):
        return combined_path
    path = get_weights_path(custom_path=custom_path, filename=GOAL_GROUNDER_WEIGHTS_FILENAME)
    if path and os.path.isfile(path):
        return path
    candidate_path = Path(__file__).resolve().parent.parent / "runs" / "weights" / "candidates" / "candidate_m3_highres_v2.pt"
    if candidate_path.is_file():
        return str(candidate_path)
    return path


def get_objectness_weights_path(custom_path: Optional[str] = None) -> Optional[str]:
    """Finds the path to pre-trained UIObjectnessModel weights (avcs_v2_weights.pt).

    Search priority:
    1. Explicit custom_path argument
    2. AVCS_V2_WEIGHTS_PATH or AVCS_OBJECTNESS_WEIGHTS_PATH environment variable
    3. Local workspace runs directory: `runs/weights/avcs_v2_weights.pt`
    4. Bundled package directory (if installed)
    5. User cache directory: `~/.cache/avcs/weights/avcs_v2_weights.pt`
    """
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    env_path = os.environ.get("AVCS_V2_WEIGHTS_PATH") or os.environ.get("AVCS_OBJECTNESS_WEIGHTS_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    return get_weights_path(filename=AVCS_V2_WEIGHTS_FILENAME)


def load_objectness_weights(
    model: Any,
    path: Optional[str] = None,
    device: str = "cpu",
    strict: bool = True,
) -> bool:
    """Loads pre-trained avcs_v2_weights.pt cleanly into UIObjectnessModel on CPU/device.

    Returns True if weights were loaded successfully, False otherwise.
    """
    resolved_path = get_objectness_weights_path(custom_path=path)
    if not resolved_path or not os.path.isfile(resolved_path):
        return False

    try:
        try:
            state = torch.load(resolved_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(resolved_path, map_location=device)

        if not isinstance(state, dict):
            return False

        target_module = model.objectness_model if hasattr(model, "objectness_model") else model
        state_dict = state.get("model_state_dict", state)
        if not isinstance(state_dict, dict):
            return False

        loaded = _load_complete_state_dict(target_module, state_dict, strict=strict)
        setattr(model, "weights_loaded", loaded)
        setattr(target_module, "weights_loaded", loaded)
        return loaded
    except Exception as exc:
        print(f"[AVCS] Warning: Failed to load objectness weights from {resolved_path}: {exc}")
        return False


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
        "GoalConditionedGrounder",
        "RealTimeCrossModalGrounder",
        "UIObjectnessModel",
    ):
        return True
    if hasattr(cortex, "objectness_model") and cortex.objectness_model is not None:
        return True
    if hasattr(cortex, "backbone") and cortex.backbone.__class__.__name__ == "RealTimeCorticalBackbone":
        return True
    if hasattr(cortex, "affordance_head") and cortex.affordance_head.__class__.__name__ == "FastAffordanceHead":
        return True
    if hasattr(cortex, "fovea_head") and cortex.fovea_head.__class__.__name__ == "FovealAcuityHead":
        return True
    if hasattr(cortex, "grounder") and cortex.grounder.__class__.__name__ in ("GoalConditionedGrounder", "RealTimeCrossModalGrounder"):
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


def _extract_grounder_metadata(state: dict[str, Any]) -> dict[str, Any]:
    """Extract grounder metadata prioritizing top-level schema keys over legacy cortical metadata.

    Never fabricates schema, provenance, or sample-count defaults when those
    fields are absent, so untrained/partial checkpoints fail the `is_trained`
    gate rather than silently masquerading as trained checkpoints.
    """
    if "schema_version" in state or "model_type" in state:
        meta: dict[str, Any] = {}
        for k in (
            "schema_version",
            "model_type",
            "dataset_provenance",
            "training_samples",
            "validation_samples",
            "target_presence_threshold",
            "held_out_test_samples",
            "negative_samples",
            "split_seed",
            "legacy_schema_version",
            "target_presence_calibrated",
            "best_val_accuracy_pct",
            "best_val_median_error_px",
            "training_meta",
            "hyperparameters",
            "eval_metrics",
        ):
            if k in state:
                meta[k] = state[k]
        return meta

    for sub_key in ("training_meta", "grounder_metadata", "grounder_meta", "metadata"):
        sub = state.get(sub_key)
        if isinstance(sub, dict) and (
            "schema_version" in sub
            or sub.get("model_type") == "avcs_goal_conditioned_grounder"
        ):
            return sub

    return {}


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

            elif target_cls in ("GoalConditionedGrounder", "RealTimeCrossModalGrounder"):
                grounder_dict = (
                    state.get("grounder")
                    or state.get("model_state_dict")
                    or state.get("state_dict")
                )
                if not isinstance(grounder_dict, dict):
                    grounder_dict = {
                        k[9:]: v for k, v in state.items() if k.startswith("grounder.")
                    }
                if isinstance(grounder_dict, dict) and grounder_dict:
                    loaded = _load_complete_state_dict(target_module, grounder_dict, strict=False)
                    if loaded:
                        meta = _extract_grounder_metadata(state)
                        if hasattr(target_module, "_mark_loaded_checkpoint"):
                            target_module._mark_loaded_checkpoint(meta)
                    if hasattr(cortex, "weights_loaded"):
                        cortex.weights_loaded = loaded
                    return loaded

            elif target_cls == "UIObjectnessModel":
                obj_dict = state.get("model_state_dict", state)
                if isinstance(obj_dict, dict):
                    loaded = _load_complete_state_dict(target_module, obj_dict, strict=strict)
                    if hasattr(cortex, "weights_loaded"):
                        cortex.weights_loaded = loaded
                    return loaded

            # HumanEye carries a grounder and an objectness head that are loaded
            # separately below; their absence from the cortical checkpoint must
            # not fail the completeness check for the backbone/affordance/fovea.
            optional_missing_prefixes = (
                ("grounder.", "objectness_model.")
                if target_cls == "HumanEye" and not strict
                else ()
            )
            # Legacy checkpoints predate the fully-learned heads in core/learned.py,
            # so their keys are absent. Treat them as optional: the backbone and
            # original heads load, and the new heads stay at their initialisation
            # until a checkpoint that contains them is trained.
            learned_head_prefixes = (
                "text_encoder.",
                "slot_detector.",
                "scene_head.",
                "action_policy.",
                "stability_learned.",
            )
            optional_missing_prefixes = optional_missing_prefixes + learned_head_prefixes
            loaded = _load_complete_state_dict(
                target_module,
                state,
                strict,
                allowed_missing_prefixes=optional_missing_prefixes,
            )

            # If cortex is HumanEye, also ensure grounder is wired and populated
            if hasattr(cortex, "grounder"):
                try:
                    from avcs.goal_grounder import GoalConditionedGrounder
                    if not isinstance(cortex.grounder, GoalConditionedGrounder):
                        feat_dim = (
                            getattr(cortex.backbone, "feature_dim", 128)
                            if hasattr(cortex, "backbone")
                            else 128
                        )
                        in_sz = (
                            cortex.input_size[0]
                            if hasattr(cortex, "input_size")
                            and isinstance(cortex.input_size, (tuple, list))
                            else 128
                        )
                        device_to_use = getattr(cortex, "device", "cpu")
                        cortex.grounder = GoalConditionedGrounder(
                            feature_dim=feat_dim, input_size=in_sz
                        ).to(device_to_use)

                    grounder_dict = state.get("grounder")
                    if not isinstance(grounder_dict, dict):
                        grounder_dict = {
                            k[9:]: v for k, v in state.items() if k.startswith("grounder.")
                        }
                    if isinstance(grounder_dict, dict) and grounder_dict:
                        grounder_loaded = _load_complete_state_dict(
                            cortex.grounder, grounder_dict, strict=False
                        )
                        if grounder_loaded:
                            meta = _extract_grounder_metadata(state)
                            if hasattr(cortex.grounder, "_mark_loaded_checkpoint"):
                                cortex.grounder._mark_loaded_checkpoint(meta)
                        cortex.grounder.eval()
                except Exception as exc:
                    print(f"[AVCS] Notice: Grounder loading skipped: {exc}")

            # If cortex is HumanEye, also wire locate and predict_point helpers if not present
            if target_cls == "HumanEye" or hasattr(cortex, "grounder"):
                if not hasattr(cortex, "locate"):
                    from avcs.goal_grounder import GroundedTarget

                    def _locate(query: str, frame: Optional[Any] = None) -> GroundedTarget:
                        if frame is None:
                            if hasattr(cortex, "source") and hasattr(cortex.source, "grab"):
                                frame = cortex.source.grab()
                            elif getattr(cortex, "_latest_raw_frame", None) is not None:
                                frame = cortex._latest_raw_frame
                            else:
                                in_sz = (
                                    cortex.input_size[0]
                                    if hasattr(cortex, "input_size")
                                    and isinstance(cortex.input_size, (tuple, list))
                                    else 128
                                )
                                import numpy as _np
                                frame = _np.zeros((in_sz, in_sz, 3), dtype=_np.uint8)
                        return cortex.grounder.locate(query, frame=frame)

                    cortex.locate = _locate

                if not hasattr(cortex, "predict_point"):
                    def _predict_point(image_tensor: torch.Tensor, query: str) -> dict[str, Any]:
                        return cortex.grounder.predict_point(image_tensor, query)

                    cortex.predict_point = _predict_point

            # If cortex is HumanEye, also ensure fovea_engine.head is populated
            if hasattr(cortex, "fovea_engine") and hasattr(cortex.fovea_engine, "head"):
                fov_head = cortex.fovea_engine.head
                fov_dict = state.get("fovea_head")
                if isinstance(fov_dict, dict):
                    cortex.fovea_engine.weights_loaded = _load_complete_state_dict(
                        fov_head, fov_dict, strict
                    )

            # If cortex is HumanEye, also ensure objectness_model is populated
            if hasattr(cortex, "objectness_model") and cortex.objectness_model is not None:
                load_objectness_weights(cortex.objectness_model, device=device, strict=False)

            return loaded
        except Exception as exc:
            print(f"[AVCS] Warning: Failed to load weights from {resolved_path}: {exc}")
            return False

    return False
