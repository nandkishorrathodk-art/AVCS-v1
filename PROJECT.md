# Project: AVCS (Artificial Visual Cortex System)

## Architecture
AVCS is a pure-pixel, CPU-optimized desktop UI perception framework providing fast visual affordance detection and foveal typing focus verification without relying on external OCR engines or accessibility trees.
- **Global / Reflex Vision**: `HumanEye` (`avcs/human_eye.py`) capturing raw desktop pixel buffers via `forge/vision/capture.py` (`Win32FastCapture`, `ScreenCapture`, `SyntheticScreen`).
- **Affordance Detection**: `FastAffordanceHead` (`avcs/human_eye.py`) predicting interactive UI affordance locations (clickable, typeable, static background) with 3x3 2D MaxPool NMS and competitive static margin gating.
- **Foveal Refinement & Focus Verification**: `refine_affordance_with_fovea` and `VisualFocusDetector` (`avcs/focus.py`) performing high-resolution spatial localization (+/-2px) and temporal caret blink frequency (0.8–2.0 Hz) analysis with fail-closed focus gating.
- **Model Storage & Packaging**: Weight discovery from `runs/weights/human_eye_weights.pt` and single-thread CPU inference (p50 = 1.14ms–5.17ms, strictly <= 20.0ms).

## Feature Inventory
| # | Feature | Description | Milestone | Source | Status |
|---|---------|-------------|-----------|--------|--------|
| 1 | Baseline Repo & Test Audit | Discover existing code, tests, APIs, and baseline status | M1: Audit & Baseline | ORIGINAL_REQUEST §2, §4 | DONE (623P/1S/0F) |
| 2 | Affordance Dataset Pipeline | Generate real desktop & synthetic UI element samples with manifest & splits | M2: Dataset & Spec | ORIGINAL_REQUEST §2 Agent B | DONE (270 frames, manifest verified) |
| 3 | Independent E2E Test Suite | 13 comprehensive tests covering R1-R4 in tests/test_independent_eval.py | M2: Test Infra | ORIGINAL_REQUEST §3 R1-R4 | DONE (13/13 passed) |
| 4 | Affordance Model & CPU Training | Implement/train CPU-compatible affordance model, focal loss, checkpoints in runs/weights/ | M3: Model & Training | ORIGINAL_REQUEST §2 Agent C, §3 R1 | DONE (260 steps, 94.5% prec, 0 FP) |
| 5 | Foveal Refinement & Caret Blink | High-res crop refinement (+/-2px) and temporal blink analysis (0.8-2.0Hz, confidence >=0.70) | M4: Foveal & Focus | ORIGINAL_REQUEST §2 Agent C, §3 R2 | DONE (100% acc, 0.45px median err) |
| 6 | Independent Benchmarking (R1-R3) | Evaluate on held-out desktop, blank/wallpaper negative set, host CPU latency measurement | M5: Independent Verification | ORIGINAL_REQUEST §2 Agent D, §3 R1-R3 | DONE (96.4% prec, 100% <=2px, 84.3 FPS) |
| 7 | Weight Discovery & Packaging | HumanEye automatic weight loading, zero test regressions | M6: Packaging & Integration | ORIGINAL_REQUEST §2 Agent E, §3 R4 | DONE (650P/1S/0F, auto-discovery verified) |
| 8 | Forensic Integrity Audit & Release | Independent audit for pure-pixel compliance (no OCR/UIA) and final 10-point report | M7: Forensic Audit & Release | ORIGINAL_REQUEST §2 Agent E, §5 | DONE (CLEAN verdict) |

## Milestones
| # | Name | Scope | Dependencies | Status | Output Artifacts |
|---|------|-------|-------------|--------|------------------|
| M1 | Audit & Baseline | Inspect existing code, test suite, and APIs (Agent A) | none | DONE | `explorer_a1/handoff.md`, `explorer_a2/handoff.md`, `spec_miner_a3/handoff.md` |
| M2 | Dataset & Test Infra | Schema definition, dataset generation, negative sets (Agent B), E2E test suite | M1 | DONE | `data/manifest.json`, `data/screens/`, `tests/test_independent_eval.py` |
| M3 | Model & Training | CPU training on dataset, focal loss, checkpoints in runs/weights/ (Agent C) | M1, M2 | DONE | `runs/weights/human_eye_weights.pt`, `runs/weights/avcs_training_summary.json` |
| M4 | Foveal & Focus | Foveal refinement & temporal caret blink focus detection verification | M1, M3 | DONE | `avcs/focus.py`, `avcs/human_eye.py` |
| M5 | Independent Verification | Verification of R1 (affordances), R2 (focus), R3 (CPU latency) (Agent D) | M3, M4 | DONE | `challenger_d1/handoff.md`, `challenger_d2/handoff.md` |
| M6 | Packaging & Integration | HumanEye weights discovery, project test verification (Agent E1) | M5 | DONE | `worker_packaging/handoff.md` (650P/1S/0F) |
| M7 | Forensic Audit & Release | Integrity forensics, final 10-point evidence report | M6 | DONE | `auditor_integrity/handoff.md` (Verdict: CLEAN) |

## Interface Contracts
- **HumanEye ↔ FastAffordanceHead**: Accepts cortical feature tensor `(B, 128, 16, 16)`, outputs affordance logits `(B, 3, 16, 16)` with static prior bias (+1.0) and interactive bias (-0.5).
- **FastAffordanceHead.find_hotspots**: 2D 3x3 MaxPool NMS + competitive static margin gating `(heat - static_heat) >= 0.05`. Returns `List[Tuple[float, float, str, float]]` (norm_x, norm_y, action_type, confidence).
- **refine_affordance_with_fovea**: Un-downsampled local crop around candidate coordinate, Sobel edge contrast check (`min_contrast >= 0.010`), returns `Optional[FovealInspection]`.
- **VisualFocusDetector.verify_focus**: Temporal difference (0.8–2.0 Hz) + spatial caret geometry + active focus ring. Fails closed (`is_focused=False`) if evidence insufficient or confidence < 0.70.
- **Model Loader ↔ runs/weights/**: Auto-discovery in `avcs/weights.py` loads `human_eye_weights.pt` containing `backbone`, `affordance_head`, `ventral`, `fovea_head`.

## Code Layout
- `avcs/`: Core AVCS implementation (`human_eye.py`, `focus.py`, `grounding.py`, `weights.py`, `sdk.py`)
- `avcs/training/`: Training routines (`train_affordance.py`, `gui_dataset.py`)
- `forge/vision/`: Vision capture and reflex components (`capture.py`, `cortex/`)
- `tests/`: 59 test files, 651 tests (650 passing, 1 skipped)
- `runs/weights/`: Evaluated model weights (`human_eye_weights.pt`), summary (`avcs_training_summary.json`)
- `data/`: Dataset manifests (`manifest.json`), 270 screen images, 270 masks, 140 fovea crops
