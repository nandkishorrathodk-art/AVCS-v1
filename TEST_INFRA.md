# TEST_INFRA — AVCS End-to-End 4-Tier Test Architecture

## 1. Executive Summary & Philosophy

The Artificial Visual Cortex System (AVCS) is a pure-pixel, closed-loop desktop perception and motor control platform. Because AVCS interacts directly with operating system windows, user data, and System 2 (LLM) agents, test verification must be authoritative, empirical, and completely free of mock-induced complacency.

### Core Testing Mandates
1. **Pure-Pixel Grounding**: All visual affordance, gaze, and focus tests operate exclusively on raw RGB pixel arrays. External OCR engines (Tesseract, EasyOCR), accessibility trees (Windows UI Automation, DOM inspectors), and accessibility metadata are strictly forbidden.
2. **Fail-Closed Safety**: Any ambiguous query, low-confidence gaze fixation, unfocused input target, or unvalidated model state must deterministically fail closed (raising `UnvalidatedGroundingError` or `UnfocusedInputElementError`) rather than executing an unverified OS action.
3. **No Prompt-Echo or Hardcoded Oracles**: Expected outputs must derive from documented mathematical properties (e.g. 5th-order minimum jerk, Fitts' law), synthetic UI ground-truth geometries, or physical invariants, never from tautological implementation mirrors.
4. **Hardware & Environment Independence**: Automated test execution must run reliably on CPU (no GPU requirement) with zero host desktop disruption (using `VirtualCursorDriver` and `SyntheticScreen` fixtures).
5. **Progressive Testability**: Features are verified against their declared milestone dependencies. Gated features (e.g., natural-language motor action prior to trained grounding checkpoint validation) must strictly pass by verifying that the fail-closed security boundary holds.

---

## 2. 4-Tier Test Architecture Overview

AVCS adopts a 4-tier hierarchical test pyramid designed to provide complete verification across individual capabilities, edge boundaries, multi-subsystem pipelines, and end-to-end user workflows.

```
+-------------------------------------------------------------------+
|               Tier 4: Real-World Scenarios (E2E)                  |
|  - Full Observe -> Ground -> Validate -> Act -> Verify Loops      |
|  - Form-fill workflows with focus gating                          |
|  - Unvalidated model safety lockout scenarios                     |
|  - Negative background rejection in desktop contexts              |
+-------------------------------------------------------------------+
|             Tier 3: Cross-Feature Interaction Pipelines           |
|  - Perception -> Motor trajectory coupling                        |
|  - Grounding -> Foveal ROI acuity refinement                      |
|  - Temporal caret blink -> Typing execution safety                |
|  - ActionBroker -> Fresh-frame visual ripple confirmation         |
|  - 60 FPS NeuroStreamBus telemetry during motor execution         |
+-------------------------------------------------------------------+
|                Tier 2: Boundary & Corner Cases                    |
|  - Resolution extremes (320x240 to 4K 3840x2160)                  |
|  - Blank & wallpaper negative test sets (zero false-positive rule)|
|  - Edge contrast limits (<0.010 contrast gating)                  |
|  - Caret blink boundary frequencies (0.8 - 2.0 Hz limits)         |
|  - Ambiguous / adversarial / empty queries                        |
+-------------------------------------------------------------------+
|                 Tier 1: Feature Coverage (Core API)               |
|  - Features 1 to 20 functional contracts (>=5 tests per feature)  |
|  - Public API validation: AVCSController / HumanEyeClient         |
|  - Strict interface conformance, parameter validation, return types|
+-------------------------------------------------------------------+
```

---

## 3. Comprehensive Feature Mapping (Features 1 – 20)

Every feature defined in `PROJECT.md` is mapped to its primary test tier, target test suite, authoritative expected output oracle, and concrete pass/fail acceptance criteria.

| Feature ID | Feature Name | Milestone | Primary Tier | Target Test File | Expected Output Oracle | Pass/Fail Acceptance Criteria |
|---|---|---|---|---|---|---|
| **1** | Baseline Test & Checkpoint Integrity | M1 | Tier 1 | `tests/e2e/test_tier1_features.py` | Pytest collector + Checkpoint state dict schema | Zero test regressions (>=666 baseline tests passing); authentic `human_eye_weights.pt` loads valid tensor dict. |
| **2** | Synthetic & Held-Out UI Dataset Pipeline | M1 | Tier 1 | `tests/e2e/test_tier1_features.py` | Geometric rendering formulas (seed-deterministic) | Known UI elements have exact bounding box coordinates $[x_1, y_1, x_2, y_2]$ and interior centroids $(x_c, y_c)$. |
| **3** | Remove SHA-256 Pseudo-Random Text Embedding | M2 | Tier 1 | `tests/e2e/test_tier1_features.py` | Cross-modal text projection / AST inspection | Text embeddings must not derive from SHA-256 seeded `torch.randn`; query variation yields deterministic semantic shift. |
| **4** | Genuine Cross-Modal Grounding Head | M2 | Tier 1 | `tests/e2e/test_tier1_features.py` | Attention map heatmap distribution | Output spatial peak responds to linguistic tokens; `predict()` satisfies confidence $\in [0, 1]$ and valid provenance. |
| **5** | Eliminate DSNT Center-Collapse | M2 | Tier 1 | `tests/e2e/test_tier1_features.py` | Spatial peak clustering / argmax | Uniform or diffuse heatmap does NOT collapse to $(0.5, 0.5)$; peak coordinate matches actual high-activation cluster. |
| **6** | Grounding Accuracy & Localization | M2 | Tier 1 | `tests/e2e/test_tier1_features.py` | Ground truth UI bounding box annotations | Top-1 accuracy $\ge 85\%$, median localization error $e_{L2} \le 5\text{ px}$ on held-out benchmark set. |
| **7** | Weight Packaging & Auto-Discovery | M2 | Tier 1 | `tests/e2e/test_tier1_features.py` | Filesystem auto-discovery contract | `HumanEye(load_weights=True)` and `HumanEyeClient` locate `runs/weights/human_eye_weights.pt` without explicit path. |
| **8** | Foveal Perception & ROI Inspection | M3 | Tier 1 | `tests/e2e/test_tier1_features.py` | Un-downsampled Sobel/Scharr edge filter | `contrast_score < 0.010` returns `None` (fail-closed); structured elements return `FovealInspection` with valid crop. |
| **9** | Zero-Hallucination Natural Language Observation | M3 | Tier 1 | `tests/e2e/test_tier1_features.py` | Scene graph / Visual memory diff | Generated description mentions only confirmed entities; blank screens yield empty entity set; reports uncertainty. |
| **10** | 60 FPS Continuous Telemetry Bus | M3 | Tier 1 | `tests/e2e/test_tier1_features.py` | C-ABI `COMPACT_FMT_40` struct binary layout | Telemetry packet unpacks to exactly 40 bytes; circular buffer maintains 120 slots; EMA FPS tracks 60 Hz. |
| **11** | 5th-Order Minimum-Jerk Kinematics | M4 | Tier 1 | `tests/e2e/test_tier1_features.py` | Analytical polynomial $10\tau^3 - 15\tau^4 + 6\tau^5$ | Velocity and acceleration are 0 at $\tau=0$ and $\tau=1$; duration scales with Fitts' Law $T \propto \log_2(1 + D/W)$. |
| **12** | Configurable 10.5 Hz Micro-Tremor | M4 | Tier 1 | `tests/e2e/test_tier1_features.py` | AR(2) biological resonator transfer function | Power spectrum peaks at $10.5 \pm 1.0\text{ Hz}$; explicit toggle disables jitter completely; final landing step error = 0. |
| **13** | Fail-Closed Typing Safety | M4 | Tier 1 | `tests/e2e/test_tier1_features.py` | Temporal difference caret analysis (0.8–2.0 Hz) | Typing with `enforce_focus=True` into static or unfocused element raises `UnfocusedInputElementError`; zero blind `Ctrl+A`. |
| **14** | Fresh Post-Action Frame Verification | M4 | Tier 1 | `tests/e2e/test_tier1_features.py` | Fresh-frame photometric diff & postcondition spec | Action success is confirmed only after fresh frame capture validates expected state delta or optical ripple $\ge 0.015$. |
| **15** | Central `AVCSController` Orchestrator | M5 | Tier 1 | `tests/e2e/test_tier1_features.py` | Unified controller lifecycle interface | Coordinates Eyes, Fovea, Hands, Verifier, and Telemetry with proper start/stop/close and usage metering. |
| **16** | Coordinate Encapsulation & Intent APIs | M5 | Tier 1 | `tests/e2e/test_tier1_features.py` | API signature contract inspection | Public intent methods accept string queries/names, not raw coordinates; internal coordinates remain encapsulated. |
| **17** | Dynamic Grounding Validation Gate | M5 | Tier 1 | `tests/e2e/test_tier1_features.py` | Security gate invariant | Natural-language action raises `UnvalidatedGroundingError` until verified weights pass validation gate. |
| **18** | Observe-to-Replan Execution Loop | M5 | Tier 1 | `tests/e2e/test_tier1_features.py` | State machine transition sequence | Loop strictly executes `observe -> ground -> validate -> act -> verify -> describe -> replan`. |
| **19** | Single-Thread CPU Latency Benchmark | M6 | Tier 1 | `tests/e2e/test_tier1_features.py` | Hardware timer (`time.perf_counter`) | Neural forward pass median latency $p50 \le 20.0\text{ ms}$ (target $\le 15.0\text{ ms}$) on 1 CPU thread. |
| **20** | Full Test Suite & Release Gate Verification | M6 | Tier 1 | `tests/e2e/test_tier1_features.py` | Multi-gate release checklist oracle | Zero failures across full suite, clean forensic audit (no OCR/UIA), authentic weights present, packaging complete. |

---

## 4. Tier Breakdown & Detailed Test Catalog

### Tier 1: Feature Coverage (`tests/e2e/test_tier1_features.py`)
Focuses on unit and functional contracts of all 20 features exercising the public APIs (`AVCSController` / `HumanEyeClient`, `HumanEye`, `NeuromuscularServoingController`, `VisualFocusDetector`, `NeuroStreamBus`).
- **Baseline & Checkpoint Integrity (Feature 1)**: Checkpoint discovery, state dict key validation (`backbone`, `affordance_head`, `ventral`, `fovea_head`), strict vs non-strict weight loading.
- **Dataset Pipeline (Feature 2)**: Synthetic screen generation across UI categories (buttons, text inputs, tabs, icons, modals) with deterministic bounding boxes and centroid coordinates.
- **Visual Grounding & Safety Gating (Features 3, 4, 5, 6, 7, 16, 17)**: Testing spatial peak clustering, elimination of random embeddings, Top-1 accuracy, and strict enforcement of `UnvalidatedGroundingError` on unvalidated query dispatches.
- **Foveal Acuity & Observations (Features 8, 9, 10)**: Edge contrast gating threshold ($<0.010$), un-downsampled local ROI inspection, zero-hallucination UIML scene tree generation, 40-byte C-ABI `NeuroPacket` binary packing.
- **Motor Control & Trajectory Kinematics (Features 11, 12, 13, 14)**: 5th-order minimum-jerk kinematics, Fitts' law duration scaling, 10.5 Hz AR(2) biological resonator jitter, temporal caret blink detection (0.8–2.0 Hz), fail-closed typing focus gating (`UnfocusedInputElementError`), fresh post-action frame verification.
- **Orchestration & Performance (Features 15, 18, 19, 20)**: Controller lifecycle (`observe`, `perceive`, `see`, `close`), observe-replan loop transitions, single-thread CPU forward-pass benchmark, release checklist validation.

### Tier 2: Boundary & Corner Cases (`tests/e2e/test_tier2_boundaries.py`)
Stress-tests extreme operating conditions, boundary thresholds, adversarial inputs, and zero-signal negative sets.
- **Resolution & Aspect Ratio Extremes**:
  - Minimum resolution: $320 \times 240$ (QVGA).
  - Standard desktop resolutions: $1280 \times 720$ (HD), $1920 \times 1080$ (FHD).
  - High-resolution / Ultrawide: $2560 \times 1440$ (QHD), $3840 \times 2160$ (4K UHD), non-standard $1600 \times 600$.
- **Negative Control & Zero False-Positive Sets**:
  - Pure uniform black screen ($R=G=B=0$).
  - Pure uniform white screen ($R=G=B=255$).
  - Uniform middle-gray screens with subtle Gaussian sensor noise.
  - Multi-stop smooth luminance gradients (horizontal, vertical, diagonal).
  - Requirement: Exactly **zero** false-positive hotspot affordances detected across all negative screens.
- **Contrast Threshold Boundaries**:
  - Zero-contrast flat ROI (contrast $= 0.000$).
  - Sub-threshold subtle texture (contrast $= 0.008 < 0.010$).
  - Boundary-edge contrast (contrast $= 0.010$).
  - High-contrast UI element (contrast $\ge 0.050$).
  - Requirement: Sub-threshold crops must return `None` (strict fail-closed rejection).
- **Temporal Caret Blink Frequency Boundaries**:
  - Sub-biological frequency: $0.2\text{ Hz}$ (too slow, e.g. slow pulse) $\rightarrow$ `is_focused == False`.
  - Lower boundary: $0.75\text{ Hz}$ (below $0.8\text{ Hz}$) $\rightarrow$ `is_focused == False`.
  - Valid biological range: $0.85\text{ Hz}, 1.0\text{ Hz}, 1.5\text{ Hz}, 1.95\text{ Hz}$ $\rightarrow$ `is_focused == True, confidence >= 0.70`.
  - Upper boundary: $2.15\text{ Hz}$ (above $2.0\text{ Hz}$) $\rightarrow$ `is_focused == False`.
  - High-frequency flicker noise: $10.0\text{ Hz}$ $\rightarrow$ `is_focused == False`.
- **Query & Coordinate Boundary Stress**:
  - Empty string query `""`.
  - Ultra-long string query (1,000+ characters).
  - Special characters, unicode emojis, shell injection tokens (`rm -rf`, `<<<`, `\x00`).
  - Extreme normalized coordinates: $(0.0, 0.0)$, $(1.0, 1.0)$, $(0.5, 0.5)$, out-of-bounds $(-0.2, 1.5)$.

### Tier 3: Cross-Feature Interaction Pipelines (`tests/e2e/test_tier3_interactions.py`)
Validates that multiple subsystems coordinate synchronously without dropped events, race conditions, or state contamination.
- **Perception-to-Motor Closed Loop**:
  - `observe()` generates scene affordance hotspots $\rightarrow$ candidate centroid is selected $\rightarrow$ `move_to()` plans minimum-jerk trajectory $\rightarrow$ virtual motor driver steps along trajectory $\rightarrow$ final cursor position strictly matches target within floating tolerance.
- **Grounding & Foveal ROI Acuity Pipeline**:
  - Candidate target coordinate is passed $\rightarrow$ un-downsampled high-resolution crop extracted $\rightarrow$ edge contrast evaluated $\rightarrow$ fine bounding box and centroid refined with subpixel precision ($e_{L2} \le 2\text{ px}$).
- **Focus Verification & Typing Execution Safety Pipeline**:
  - Target input box identified $\rightarrow$ temporal frame buffer inspected $\rightarrow$ caret blink verified $\rightarrow$ only upon positive focus is keystroke dispatch permitted $\rightarrow$ typing into non-blinking elements raises `UnfocusedInputElementError` with zero sent keystrokes.
- **Action Broker State Machine & Verification Loop**:
  - Motor action requested $\rightarrow$ state moves `PENDING` $\rightarrow$ `DISPATCHED` $\rightarrow$ fresh post-action frame captured $\rightarrow$ photometric diff or optical ripple calculated $\rightarrow$ state transitions to `CONFIRMED` or escalates through recovery tiers.
- **Continuous 60 FPS Telemetry Bus During Motor Execution**:
  - `NeuroStreamBus` running in background $\rightarrow$ motor executes multi-step trajectory $\rightarrow$ bus publishes 40-byte `NeuroPacket` for each tick $\rightarrow$ subscriber records continuous stream of gaze, stability, and motor flow.

### Tier 4: Real-World Scenarios & Workflows (`tests/e2e/test_tier4_scenarios.py`)
Exercises high-level end-to-end agent computer-use workflows matching actual operational deployments.
- **Scenario 1: Desktop Workspace Exploration & Scene Comprehension**:
  - Agent initializes client $\rightarrow$ captures full desktop $\rightarrow$ extracts UIML scene tree and visual diff $\rightarrow$ identifies all interactive buttons and menus $\rightarrow$ verifies memory snapshot holds multi-turn episodic history.
- **Scenario 2: Safe Form-Fill Workflow with Focus Protection**:
  - Agent targets a text input field $\rightarrow$ inspects foveal acuity $\rightarrow$ evaluates temporal caret blink $\rightarrow$ safely types text $\rightarrow$ verifies text appears on fresh frame without executing destructive blind `Ctrl+A` + `Backspace`.
- **Scenario 3: Unvalidated Grounding Safety Lockout**:
  - Agent receives natural-language command `"click Submit button"` with unvalidated grounding weights active $\rightarrow$ controller strictly intercepts dispatch and raises `UnvalidatedGroundingError` $\rightarrow$ zero motor motion dispatched $\rightarrow$ safety log captures blocked attempt.
- **Scenario 4: Negative Background Rejection in Live Context**:
  - Agent is presented with a desktop showing empty wallpaper and blank canvas windows $\rightarrow$ affordance detector outputs zero candidate hotspots $\rightarrow$ client reports clean state $\rightarrow$ no phantom clicks dispatched.
- **Scenario 5: Closed-Loop Observe-Ground-Validate-Act-Verify-Replan**:
  - Complete execution cycle following the mandate: `Observe -> Ground -> Validate -> Act -> Verify -> Describe -> Replan`. If action verification detects no state change, the controller enters `Replan` status rather than blindly proceeding.

---

## 5. Test Harness & Fixtures Architecture

All E2E tests leverage dedicated, isolated fixtures located in `tests/e2e/conftest.py`:

1. **`synthetic_screen`**: Provides synthetic desktop image generation without requiring OS screen capture or a physical monitor. Supports configurable seed, resolution, and element placement.
2. **`virtual_motor`**: Emulates OS mouse and keyboard input via `VirtualCursorDriver`, recording all coordinate trajectories, clicks, and keystrokes into an inspectable memory buffer. Zero real mouse or keyboard events are sent to Windows.
3. **`test_client`**: Instantiates `HumanEyeClient(source="synthetic", force_virtual_motor=True)` (or `AVCSController` when present), providing a clean, fully initialized public interface for opaque-box testing. Automatically closed on fixture teardown.
4. **`negative_screen_generator`**: Produces parameterized pure black, pure white, uniform gray, and linear gradient screens for negative control verification.
5. **`caret_frame_sequence`**: Generates synthetic frame buffers simulating a blinking text caret at arbitrary frequencies ($f \in [0.1, 10.0]\text{ Hz}$) with timestamp metadata.

---

## 6. Execution Commands & Pytest Invocations

All tests conform strictly to standard `pytest` conventions:

```bash
# Run all E2E test suites
python3 -m pytest tests/e2e/ -v

# Run individual tiers
python3 -m pytest tests/e2e/test_tier1_features.py -v
python3 -m pytest tests/e2e/test_tier2_boundaries.py -v
python3 -m pytest tests/e2e/test_tier3_interactions.py -v
python3 -m pytest tests/e2e/test_tier4_scenarios.py -v

# Run with tier markers
python3 -m pytest tests/e2e/ -m tier1 -v
python3 -m pytest tests/e2e/ -m tier2 -v
python3 -m pytest tests/e2e/ -m tier3 -v
python3 -m pytest tests/e2e/ -m tier4 -v

# Run full project regression suite (must maintain 666+ passed, 0 failed)
python3 -m pytest tests/ -q
```

---

## 7. Progressive Testability & Milestone Dependencies

The following table documents the verification strategy across implementation milestones:

| Milestone | Target Deliverables | E2E Test Strategy & Invariants |
|---|---|---|
| **M1** (Current) | Dataset pipeline, ground truth generation, baseline verification | Verify baseline tests pass (666+); verify synthetic dataset generator produces exact bounding boxes/centroids; verify blank screen zero-FP negative rejection; verify fail-closed grounding errors. |
| **M2** | Trainable cross-attention grounding head, weights packaging | Verify removal of SHA-256 random features; verify DSNT center-collapse is eliminated; verify grounding accuracy $\ge 85\%$ and localization error $\le 5\text{ px}$. |
| **M3** | Foveal dialogue, zero-hallucination observation, 60 FPS bus | Verify foveal acuity contrast gating ($<0.010$); verify zero-hallucination UIML scene tree generation; verify 40-byte C-ABI `NeuroStreamBus` live broadcast. |
| **M4** | Closed-loop motor control, typing safety, fresh verification | Verify 5th-order minimum-jerk kinematics, 10.5 Hz micro-tremor toggle, fail-closed typing focus gating, fresh post-action frame verification. |
| **M5** | Central `AVCSController`, System 2 observe-replan loop | Verify `AVCSController` coordination; verify coordinate encapsulation; verify dynamic validation gate lifts `UnvalidatedGroundingError` only when validated weights are active. |
| **M6** | Full integration, CPU latency benchmark, release packaging | Verify CPU forward pass latency $\le 20.0\text{ ms}$; verify full 666+ test suite passes with zero regressions; forensic integrity audit verification. |
