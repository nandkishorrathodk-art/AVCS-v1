# Original User Request

## 2026-10-15T02:33:11Z

# AVCS — Audit-First Multi-Agent Implementation & Verification

**Project:** Artificial Visual Cortex System (AVCS) — System 1: Eyes + Hands  
**Working directory:** `C:\Users\nandk\Documents\antigravity\splendid-babbage`  
**Execution target:** Local Windows CPU, standard 4-core machine, no GPU requirement  
**Integrity mode:** Benchmark, evidence-first, no fabricated results

## Mission

Inspect, improve, train, evaluate, and package the existing AVCS implementation into a functioning pure-pixel desktop UI perception system.

Use the available multi-agent orchestration system to coordinate specialized agents. Begin with a repository audit, establish interfaces and baselines, then perform compatible tasks in parallel.

**Do not claim readiness, successful training, test completion, or achieved metrics without verifiable evidence from the host machine.**

## 1. Non-Negotiable Constraints

- Use raw pixel buffers and image-frame sequences for visual affordance detection and typing-focus verification.
- Do not use Tesseract, EasyOCR, external OCR engines, Windows UI Automation/accessibility trees, DOM inspectors, or accessibility metadata as visual detection inputs.
- Keep model training and inference CPU-compatible. GPU acceleration must not be required.
- If visual evidence is insufficient to establish an interactive target or typing focus, fail closed.
- Preserve existing APIs and project behavior wherever practical.
- Do not fabricate screenshots, annotations, trained weights, benchmark results, or passing tests.
- Synthetic data may supplement training, but synthetic-only evaluation does not satisfy real desktop validation.
- Never weaken tests, suppress failures, or tune the model directly against the final held-out test set to manufacture passing results.

## 2. Multi-Agent Responsibilities

**Agent A — Repository Auditor and Integration Lead**
- Inspect the repository, existing architecture, dependencies, tests, and current Git state.
- Locate `HumanEye`, `FastAffordanceHead`/`AffordanceHead`, `refine_affordance_with_fovea`, and existing training/evaluation code where present.
- Verify whether the named APIs and expected interfaces actually exist.
- Establish a baseline test report and document existing failures.
- Define shared interfaces, task dependencies, and integration ownership.

**Agent B — Dataset Generator**
- Build a reproducible dataset pipeline for buttons, text fields, tabs, icons, other relevant interactive UI elements, and static/blank desktop backgrounds.
- Include real desktop screenshots when available and synthetic samples where useful.
- Produce documented labels, bounding boxes or heatmaps, negative examples, and a dataset manifest.
- Split datasets by appropriate application, source, or capture session to reduce data leakage.
- Document data provenance, limitations, and annotation quality.

**Agent C — Model Trainer**
- Implement or improve the existing backbone and affordance head without unnecessarily duplicating project components.
- Train on CPU using an appropriate multi-task objective and focal loss where justified by the existing design and class imbalance.
- Save reproducible checkpoints, configuration, preprocessing parameters, and training logs under `runs/weights/`.
- Optimize single-thread CPU inference and test any quantization or other optimizations for accuracy regressions.
- Never report a successful run unless the actual training process completed.

**Agent D — Independent Verifier**
- Evaluate the final candidate on untouched held-out frames.
- Independently measure detection precision, recall, false positives, spatial localization error, and inference latency.
- Test blank screens and static wallpapers separately.
- Verify temporal cursor-blink detection and fail-closed focus behavior.
- Run the complete discovered test suite before and after changes and report actual results.

**Agent E — Packaging and Integrity Auditor**
- Verify weight discovery and loading by `HumanEye`.
- Audit dependencies and code for prohibited visual-input mechanisms.
- Check CPU-only execution, configuration portability, reproducibility, and missing artifacts.
- Confirm release packaging does not silently fall back to an untrained model while claiming trained-model operation.

Agents may work concurrently on independent tasks. Training must use agreed data and model interfaces. Avoid concurrent conflicting edits; assign an integration owner and use isolated branches/worktrees or separate artifacts when supported.

## 3. Acceptance Criteria

Treat the following as **measurable target thresholds, not assumed accomplishments**.

### R1 — Affordance Detection
- Target at least 80% precision on a predefined held-out real desktop UI evaluation set.
- Zero false-positive hotspots on the explicitly defined blank-screen and static-wallpaper negative test set.
- Apply 2D spatial non-maximum suppression using a 3×3 neighborhood to remove redundant clustered detections.
- Report precision, recall, F1, true positives, false positives, false negatives, and the number of evaluated frames.
- State the test-set size and composition. Zero false positives on a finite test set is not a guarantee of zero false positives in all future environments.

### R2 — Foveal Refinement and Typing Focus
- Integrate `refine_affordance_with_fovea` or its existing project equivalent using un-downsampled local crops.
- Target localization error within ±2 pixels against suitable ground-truth annotations; define the exact error metric and report the distribution.
- Detect caret blinking in the 0.8–2.0 Hz target range using actual timestamped temporal frames.
- Target focus confidence of at least 0.70 only when adequate visual evidence supports it.
- Static, ambiguous, unfocused, or non-input elements must produce `is_focused=False`.
- Verify that typing cannot proceed when confidence, temporal evidence, or target identification is insufficient.

### R3 — CPU Performance
- Target a neural forward-pass latency of ≤20.0 ms on one CPU thread.
- Measure on the actual host using representative input resolutions and realistic inference conditions.
- Report warm-up procedure, thread configuration, sample count, median latency, p95 latency, and achieved FPS.
- Separate neural forward-pass latency from preprocessing, foveal refinement, and end-to-end latency.
- If the threshold is missed, report the measured result and bottleneck honestly; do not invent compliance.

### R4 — Tests and Release
- Discover the actual project test count instead of assuming that 613 tests exist.
- Run the baseline suite and the final suite. Target zero failures and zero regressions.
- Preserve and document pre-existing failures separately.
- Verify automatic weight discovery and real inference through `HumanEye`.
- Save the final evaluated weights, metadata, and reproducible configuration under `runs/weights/`.
- Ensure the release can be loaded and tested on the target CPU without a GPU.

## 4. Execution Plan

**Phase 1 — Audit:** Inspect code, establish test and performance baselines, inventory data and existing checkpoints, and identify blockers.

**Phase 2 — Prepare:** Define dataset schemas, model interfaces, labels, train/validation/test splits, and agent ownership.

**Phase 3 — Implement and Train:** Execute independent work in parallel where safe. Train on CPU and log all meaningful experiments.

**Phase 4 — Independent Verification:** Freeze the candidate checkpoint and evaluate it on untouched test data. Run focus-safety checks, performance benchmarks, and the full test suite.

**Phase 5 — Integrate and Package:** Fix demonstrated regressions, verify automatic weight loading, audit prohibited dependencies, and assemble the release.

**Phase 6 — Final Report:** Provide actual commands, artifacts, test results, benchmark methodology, failures, and limitations.

## 5. Required Final Evidence

Deliver a final report containing:

1. Repository audit and baseline test results.
2. Files and components changed, with reasons.
3. Dataset provenance, counts, label definitions, and split methodology.
4. Training configuration, logs, checkpoint locations, and reproducibility instructions.
5. Held-out precision, recall, false-positive counts, localization error, and dataset size.
6. Focus-verification measurements and fail-closed test results.
7. CPU latency statistics and achieved FPS, measured on the host.
8. Baseline-versus-final test counts and all remaining failures.
9. Actual weight-loading verification through `HumanEye`.
10. Release artifacts, known limitations, and any acceptance criteria not met.

**Completion rule:** Report AVCS as release-ready only when the required evidence supports that conclusion. Otherwise, provide the completed work, measured results, remaining blockers, and an explicit pass/partial-pass/fail status for every criterion.

Begin with Phase 1. Do not presume that the repository, dataset, APIs, or test suite match this specification; discover and verify them first.


## 2026-10-15T14:51:24Z

# AVCS — End-to-End Artificial Visual Cortex
## Pixel-to-Perception, Natural Language Grounding & Autonomous Computer Use

**Project status:** Launched  
**Working directory:** `C:\Users\nandk\Documents\antigravity\splendid-babbage`  
**Execution mode:** Local CPU-first development  
**Primary goal:** Build, train, empirically verify and package a real visual perception and motor-control system that operates a desktop from raw screen pixels and communicates with System 2 through natural language.

## 1. Mission

Build AVCS as a closed-loop computer-use system with three core capabilities:

1. **Eyes:** Perceive desktop pixels, identify applications and UI elements, and localize natural-language targets.
2. **Visual language:** Convert observed visual scenes and state changes into accurate, human-readable observations for System 2 (LLM).
3. **Hands:** Translate high-level intent into grounded mouse and keyboard actions, verify their postconditions, and report the actual outcome.

System 2 must express intent such as “Open VLC Player and inspect its playlist.” It must not calculate, invent or manage screen coordinates.

AVCS owns visual grounding, internal target geometry, motor planning, action execution and verification.

## Core Architectural Mandates & User Corrections

1. **Central AVCS Controller / Orchestrator:**
   System 2 communicates directly with a central `AVCSController` / `HumanEyeClient` orchestrator. The controller coordinates Eyes (Perception & Grounding), Fovea, Hands (Neuromuscular Motor Actuation), and Verifier. System 2 never talks directly to isolated sensory endpoints.

2. **Zero Hallucinated Visual Observations:**
   Visual descriptions emitted to System 2 must strictly derive from actual neural features, optical edges, and real pixel evidence. Never fabricate filenames, movie titles, or UI states. When visual evidence is ambiguous, report uncertainty.

3. **Zero Hardcoded Coordinates & Strict Geometry Encapsulation:**
   All coordinates must be dynamic outputs from the trained neural grounding head (spatial peak clustering). No hardcoded coordinates. Geometry is strictly encapsulated within AVCS; System 2 receives natural-language observations and emits high-level intent.

4. **Fresh Post-Action Verification:**
   After any motor execution, AVCS captures a fresh screen frame and verifies the postcondition (e.g. window appearance, button state transition, optical delta) before confirming success.

5. **Real-Time Continuous Telemetry & Streaming:**
   AVCS must provide a continuous real-time streaming pipeline (60 FPS telemetry bus) broadcasting live gaze fixation, scene stability, surprise alerts, and motor flow to allow real-time monitoring and fluid closed-loop control.

## 2. Mandatory repository audit

Before modifying code:

- Inspect the repository structure, existing architecture, dependencies, tests, training scripts and current model checkpoints.
- Locate `HumanEye`, `UnvalidatedGroundingError`, existing `locate()` and `click_target()` implementations, and all motor-control interfaces.
- Trace the current pixel-to-prediction pipeline and identify any SHA-256 pseudo-random feature generation, hard-coded coordinates, center-collapse behavior, mocked inference, or untrained model paths.
- Record the baseline test results before making changes.
- Preserve working APIs and backward compatibility wherever practical.
- Do not overwrite valuable checkpoints or silently change existing contracts.
- Create a written implementation plan with explicit dependencies and measurable acceptance criteria.

Do not assume the repository matches this specification. Adapt to the actual code.

## 3. Multi-agent teamwork

Use the available, supported multi-agent orchestration mechanism if present. Assign independent tasks with explicit file ownership and integration contracts.

**Integration Lead**
- Own architecture, interfaces, integration and release decisions.
- Prevent conflicting edits.
- Maintain the task ledger and evidence report.

**Dataset Generator**
- Build reproducible synthetic desktop scenes and collect permitted real desktop screenshots.
- Generate accurate target identities and ground-truth bounding boxes or centroids.
- Include varied layouts, DPI scaling, icon sizes, themes, clutter, occlusion and negative queries.
- Split train, validation and test data by scene or layout identity to reduce leakage.

**Cross-Attention Grounding Trainer**
- Implement a genuine trainable language-vision grounding model.
- Train against ground-truth annotations.
- Save checkpoints, training configuration and evaluation results.
- Compare against simple baselines.

**Foveal Dialogue Engineer**
- Implement region-of-interest inspection, visual state tracking and natural-language observation generation.
- Separate directly observed facts from uncertain interpretations.
- Build observation and intent interfaces for System 2.

**Independent Verifier**
- Audit dataset leakage, target accuracy, localization error, latency, safety and test integrity.
- Independently reproduce reported metrics.
- Reject fabricated evidence, mocked success and unsupported release claims.

If agent orchestration is unavailable, execute these workstreams sequentially and document the limitation. Do not claim parallel delegation occurred unless it actually did.

## 4. R1 — Genuine visual grounding

Implement a trainable grounding model that maps a natural-language query and raw screen pixels to a target region.

Example queries:
- “Chrome browser”
- “VLC Player”
- “Notepad”
- “Terminal”
- “Settings”
- “The Play button in the active media player”

Requirements:

- Remove pseudo-random SHA-256-generated visual features from the learned inference path.
- Use a real visual encoder and a text encoder with a trainable cross-modal interaction or cross-attention grounding head.
- Train using annotated target regions and appropriate localization losses.
- Use spatially calibrated heatmaps, candidate regions or bounding-box predictions.
- Avoid center-collapse caused by averaging ambiguous multimodal spatial distributions.
- Support multiple instances, ambiguous names, missing targets and low-confidence predictions.
- Return an internal grounded target containing identity, region geometry, confidence, uncertainty and frame identity.
- Ensure all geometry corresponds to the current frame and current display configuration.

Required public API:

- `locate(query)` — resolve a natural-language target into a verified internal target.
- `click_target(query)` — locate, validate, execute and verify a click.
- Provide a structured failure result when grounding is ambiguous, stale or unsuccessful.

Use the existing project's API conventions wherever possible.

### Training and evaluation

Build a reproducible training pipeline with deterministic configuration, checkpoint saving, validation and held-out evaluation.

Report:
- Top-1 target identification accuracy.
- Recall and precision where appropriate.
- Median, mean and 95th-percentile centroid localization error in physical screen pixels.
- Bounding-box IoU where bounding boxes are available.
- Results by target class, screen layout and difficulty.
- Confidence calibration and false-positive rates.
- Results for ungroundable queries.

Acceptance targets:
- Top-1 grounding accuracy ≥ 85%.
- Median centroid localization error ≤ 5 physical pixels on the explicitly defined supported benchmark.
- No systematic center-collapse.
- No fabricated or hard-coded predictions.

A failed threshold must remain a failed threshold. Never tune against the held-out test set.

## 5. R2 — Foveal perception and natural visual dialogue

Implement a multistage perception pipeline:

1. Capture the current desktop frame.
2. Detect candidate applications, windows and UI regions.
3. Ground the requested target.
4. Allocate higher-resolution inspection to relevant regions when needed.
5. Inspect visible content using appropriate visual recognition, OCR and temporal comparison.
6. Generate a natural-language observation with evidence-backed statements.
7. Send the observation and available action capabilities to System 2.

Example desktop observation:

“The desktop is visible. Chrome appears on the left, and VLC Player is near the lower center. No obvious error dialog is visible.”

Example after opening VLC:

“VLC Player is open. A playlist containing two visible media entries is displayed. Playback appears stopped, and the Play control is visible.”

These are illustrative outputs, not assumptions about the actual screen.

Rules:
- Never invent application contents, filenames, movie titles or UI states.
- Distinguish visible facts from inferred facts.
- Report uncertainty when text is unreadable or an element is occluded.
- Track state transitions using fresh frames.
- Avoid repeating unchanged observations unnecessarily.
- Describe failures and unexpected dialogs in natural language.

Define a structured observation contract with fields for scene description, visible entities, state changes, confidence and timestamp/frame identity. Keep raw coordinate geometry internal to AVCS; do not expose it as a requirement for System 2 planning.

## 6. R3 — Closed-loop motor execution

Implement a motor controller that receives grounded targets rather than LLM-generated coordinates.

Requirements:
- Use fifth-order minimum-jerk trajectories where appropriate.
- Make the 10.5 Hz micro-tremor experiment configurable and disableable.
- Prevent tremor from degrading precise clicks or safety-critical interactions.
- Respect current screen geometry, DPI scaling and target movement.
- Revalidate the target before committing an action.
- Stop or re-ground when the screen changes unexpectedly.
- Support an emergency motor freeze and cancellation path.
- Use UI Automation or semantic control invocation when available and appropriate, with a separately tested visual fallback.

### Postcondition verification

After each action:
- Capture a fresh observation when available.
- Check the expected state transition using independent evidence, such as target-window appearance, control state, text changes or image differences.
- Use optical-flow or photometric-difference signals only as supporting evidence, not as universal proof of a successful click.
- Return success only when the specified postcondition is observed.
- Return failure, timeout or uncertainty otherwise.

Do not claim 15-microsecond verification on an ordinary desktop display. Measure capture, inference, action dispatch and postcondition-verification latency separately, and report end-to-end timing.

### Typing safety

Before typing:
- Verify the intended foreground application.
- Verify the intended text control or an equivalent reliable focus signal.
- Confirm the target is still valid.
- Fail closed if focus cannot be established.
- Do not use caret blinking alone as proof of focus.
- Never type into an unexpected window.
- Require explicit user confirmation for destructive, sensitive or externally consequential operations when appropriate.

## 7. R4 — System 2 interface

Define a high-level intent interface such as:

- `open_application("VLC Player")`
- `inspect_active_window()`
- `describe_visible_content()`
- `click_target("Play button")`
- `type_text("...")` — only after focus and authorization checks.

These examples describe intended interfaces; adapt them to the existing architecture.

The execution loop must be:

**Observe → Ground → Validate → Act → Verify → Describe → Replan**

System 2 receives natural-language observations and action outcomes, not raw coordinate-management tasks.

AVCS must never interpret a generated natural-language observation as proof that an action succeeded. Only verified postconditions can establish success.

## 8. Performance targets

Benchmark on the actual host CPU and record processor model, core/thread configuration, RAM, operating system, Python/PyTorch versions, model size, input resolution and thread settings.

Measure:
- Pixel capture latency.
- Visual encoder latency.
- Grounding-head latency.
- Complete perception latency.
- Natural-language synthesis latency.
- Motor dispatch latency.
- End-to-end action verification latency.
- Peak memory usage.

Treat ≤ 15 ms for the neural perception forward pass on one CPU thread as a target to measure, not a guaranteed outcome.

Measure emergency freeze latency separately. Prioritize immediate cancellation and motor inhibition rather than running another expensive perception pass before stopping movement.

Use warm-up runs, repeated measurements, median and tail percentiles. Report failures and hardware limitations honestly.

## 9. Tests and benchmark integrity

Build a dedicated held-out benchmark containing synthetic and real desktop screens, with documented data provenance and ground truth.

Test:
- Distinct application positions.
- Multiple instances of the same application.
- Unseen layouts and themes.
- DPI scaling and different resolutions.
- Occlusion, small icons and clutter.
- Missing or ambiguous targets.
- Stale frames and moving windows.
- Failed clicks and unexpected dialogs.
- Incorrect foreground windows.
- Missing typing focus.
- Emergency freeze and cancellation.
- Natural-language observation correctness.
- CPU latency and memory limits.

Run the full existing test suite before and after implementation.

Separate:
1. Unit tests.
2. Integration tests.
3. Held-out model benchmarks.
4. Live desktop smoke tests.
5. Safety and failure-injection tests.

Never replace a real integration test with a mock and report it as a live end-to-end test.

## 10. Checkpoint discovery and release packaging

Save trained weights under `runs/weights/`, using the project's established checkpoint conventions. For example:

`runs/weights/human_eye_weights.pt`

Ensure `HumanEye(load_weights=True)` discovers and loads compatible checkpoints automatically.

Include:
- Model architecture and configuration.
- Checkpoint metadata and versioning.
- Dataset manifests and split definitions.
- Training and evaluation scripts.
- Grounding benchmark results.
- CPU performance measurements.
- Test logs and failure summaries.
- Reproduction instructions.
- Known limitations and supported operating conditions.

Never load untrusted checkpoint files. Validate checkpoint compatibility, expected architecture and supported serialization format.

## 11. Removing the validation guard

Do not remove `UnvalidatedGroundingError` merely because a checkpoint exists or a training script completes.

Replace the blanket restriction only when a reproducible validation gate confirms:
- Checkpoint integrity and architecture compatibility.
- Successful held-out benchmark evaluation.
- Required grounding thresholds.
- Proper low-confidence and missing-target rejection.
- Successful regression and safety tests.

Unvalidated models must remain blocked from unrestricted live actions. Keep the validation gate active in development and fail closed if model provenance or evaluation evidence is missing.

## 12. Final deliverables

Deliver:
1. Integrated source implementation.
2. Reproducible training pipeline.
3. Real trained checkpoints, if training succeeds.
4. Held-out benchmark dataset manifest and evaluation results.
5. CPU latency report with actual measurements.
6. Full test-suite report.
7. Independent verification report.
8. Packaging and model-loading verification.
9. Explicit list of incomplete requirements and known limitations.

## Final execution rule

First inspect the repository and report the baseline. Then implement, train, evaluate, integrate and verify in controlled stages.

Do not stop after writing a plan. Do not claim training, testing, CPU measurements or checkpoint loading occurred unless they actually did.

**Release decision:** PASS only when all mandatory release gates are supported by reproducible evidence. Otherwise report BLOCKED or PARTIAL, identify the failed gates and provide the exact evidence needed to proceed.


## 2026-10-17T06:36:58Z

# Teamwork Project Prompt — AVCS Pure-Pixel Neural Sensorimotor Engineering

**Status:** Launched  
**Integrity mode:** Strict benchmark  
**Project type:** Full-team engineering, training, integration, verification and hardening  
**Working directory:** `C:\Users\nandk\Documents\antigravity\splendid-babbage`

## Mission

Build, train, evaluate and package AVCS (Artificial Visual Cortex System), a genuine end-to-end, pure-pixel neural sensorimotor system for desktop interaction.

AVCS must transform raw desktop RGB pixels and a natural-language target description into calibrated spatial localization, target-presence confidence, safe motor actions and closed-loop visual verification.

The system must learn visual grounding through neural representations rather than external semantic shortcuts.

The objective is not to manufacture a passing benchmark result. The objective is to implement a reproducible system, identify its real limitations and demonstrate its actual performance on held-out data and a live desktop.

**Approval gate:** Do not begin irreversible desktop actions or delegate execution through `invoke_subagent` until the required user approval has been obtained.

## Team Structure

Assign independent responsibilities to the following agents, where the orchestration environment supports parallel delegation.

1. **Neural Grounding Engineer:** Own the pixel encoder, language-query conditioning, spatial heatmaps, boundary estimation and centroid localization.
2. **Presence & Adversarial Evaluation Engineer:** Own confidence calibration, absent-target detection, hard-negative datasets and hallucination rejection.
3. **Foveal Resolution & Coordinate Engineer:** Own high-resolution patch refinement, aspect ratios, DPI conversion and multi-monitor coordinate mapping.
4. **Sensorimotor & Verification Engineer:** Own guarded input execution, cursor trajectories, fresh-frame perception and post-action verification.
5. **Independent Verifier & Packaging Auditor:** Independently inspect implementation integrity, benchmark methodology, runtime behavior and reproducibility.

The integration lead must coordinate interfaces, resolve conflicting assumptions and maintain one authoritative acceptance report. If multi-agent execution is unavailable, implement these responsibilities sequentially without pretending that separate agents ran.

---

## Architectural Contract: System 2 ↔ AVCS System 1

System 2 is the reasoning and planning layer. It provides a task objective, authorized action instructions and expected postconditions.

AVCS System 1 is the local pixel-based sensorimotor layer. It perceives the current desktop, grounds targets, controls mouse and keyboard input, detects visual state transitions and reports evidence-backed results to System 2.

### Execution pipeline

1. **Peripheral perception:** Scan the current RGB frame to identify candidate regions without relying on hardcoded target coordinates.
2. **Foveal saccade:** Extract a high-resolution patch around the selected candidate and refine target boundaries, centroid and uncertainty.
3. **Motor execution:** Generate a smooth, bounded cursor trajectory. Support minimum-jerk motion and evaluate optional micro-tremor through controlled ablation tests.
4. **Motion and state perception:** Classify observed application states, including transitions, loading, stable rendering, errors, completion and uncertainty.
5. **Interaction verification:** Capture fresh frames and verify the task-specific postcondition.
6. **System 2 feedback:** Return a concise report containing the observed state, evidence, result and recommended next step.

---

## R1 — Pure-Pixel Neural Target Grounding

Input:
- The current raw RGB desktop frame.
- A natural-language target description.

Output:
- Target-presence confidence.
- Spatial target heatmap.
- Predicted target boundary or region, where supported.
- Predicted centroid in screenshot pixel coordinates.
- Localization uncertainty.
- A machine-readable decision indicating `PRESENT`, `ABSENT` or `UNCERTAIN`.

The grounding pipeline must learn target identity, visual appearance, position and geometry from pixels and learned language-vision representations.

Strictly prohibit:
- Tesseract, EasyOCR, PaddleOCR, Windows OCR and any other OCR engine.
- Windows UIAutomation, MSAA and accessibility-tree extraction.
- Browser DOM inspection, Selenium, Playwright DOM APIs and hidden web-page metadata.
- Hardcoded target coordinates, application-specific coordinate tables or benchmark-specific output overrides.
- Remote vision APIs or external services that bypass the local neural perception pipeline.

Permitted infrastructure must be declared explicitly. Ordinary numerical libraries, model frameworks, dataset tooling, raw-pixel capture and low-level input injection may be used when required. OS interfaces may provide physical display geometry and input delivery, but must never provide semantic UI information.

Do not claim arbitrary text recognition unless the trained pixel model has been tested on the relevant text, fonts, scales and unseen layouts.

---

## R2 — Calibrated Presence and Adversarial Rejection

Replace keyword-only matching with full-description visual grounding.

For example, a query such as “click the unicorn rainbow button” must not match an ordinary button merely because a button exists or the keyword is associated with familiar UI patterns.

Use the following decision policy:

- Presence confidence below `0.35`: classify the requested target as `ABSENT`.
- Presence confidence from `0.35` through `0.69`: classify it as `UNCERTAIN`; prohibit action.
- Presence confidence at or above `0.70`: eligible for further authorization checks, but never sufficient on its own to authorize execution.

Before authorizing an action, independently validate localization quality, target-region plausibility, current focus and the freshness of the supporting frame. Reject ambiguous, partially matched or contradictory descriptions.

Calibrate presence scores against held-out examples. Report calibration quality, false-positive rates and precision-recall trade-offs. Do not assume that a raw neural score is a calibrated probability.

The authorization flag must fail closed:

`action_authorized = False`

whenever target presence is below `0.70`, the target is uncertain, focus is unconfirmed, the frame is stale, or localization confidence falls below a documented safe threshold.

---

## R3 — Aspect-Ratio, DPI and Foveal Coordinate Accuracy

Support non-square displays, including `1920×1200`, as well as different resolutions, window layouts, multi-monitor arrangements and high-DPI scaling.

The model must preserve spatial geometry throughout preprocessing, feature extraction, heatmap decoding and local refinement.

Implement foveal refinement using high-resolution pixel patches when necessary. Patch selection and coordinate restoration must derive from the current frame rather than fixed screen locations.

Maintain an explicit coordinate transform between:
- Screenshot pixel coordinates.
- Physical monitor pixel coordinates.
- Input API coordinates, where the operating system uses a different coordinate space.

Account for monitor origins, negative virtual-desktop coordinates, scaling differences and resolution changes. Detect invalid or stale geometry and fail closed rather than clicking a guessed location.

Primary localization target:
- Median absolute centroid error: `≤2.0` physical screen pixels on the declared high-resolution precision benchmark.
- 95th-percentile centroid error: `≤3.0` physical screen pixels on that same benchmark.

Report results by target size, resolution, monitor configuration and visual difficulty. Do not describe the system as universally sub-pixel accurate without appropriate measurements.

---

## R4 — Closed-Loop Sensorimotor Execution

Convert verified target positions into controlled cursor trajectories. A minimum-jerk trajectory may be used when it improves measured performance. Optional biological micro-tremor must remain configurable and must not interfere with the final docking or click position.

Separate the stages:

1. Perceive the current desktop frame.
2. Ground the requested target.
3. Evaluate presence, uncertainty and focus.
4. Plan the motor trajectory.
5. Revalidate the target immediately before execution.
6. Execute only an authorized low-level input action.
7. Capture fresh post-action RGB frames.
8. Evaluate the intended visual state transition.
9. Return `SUCCESS`, `FAILURE` or `UNKNOWN`.

Never treat a cursor movement, mouse-down event or lack of an exception as proof of task completion.

Use task-specific visual postconditions. For example, verifying that a button was clicked is not enough if the intended result was opening a particular dialog.

Use bounded retries and timeouts. If the interface changes unexpectedly, the target disappears, focus changes or the postcondition cannot be established, stop and report the actual status.

Live smoke tests must start in dry-run mode. Physical interaction must be explicitly enabled for a controlled test environment. Destructive or consequential actions require appropriate user authorization.

---

## R5 — System 2 ↔ AVCS Closed-Loop Task Execution

### Objective
Integrate AVCS as the visual sensorimotor execution layer of a larger AI agent architecture.

System 2 is responsible for natural-language reasoning, task planning, code generation, command selection and deciding the intended postconditions.

AVCS is responsible for perceiving the real desktop from raw RGB frames, grounding System 2's instructions in current visual evidence, executing authorized mouse and keyboard actions, verifying resulting state transitions and returning grounded textual feedback.

### Required Execution Cycle
`System 2 instruction → Pixel perception → Target grounding → Safe action authorization → Physical interaction → Fresh-frame verification → Structured textual feedback → System 2 decision`

### Terminal Readiness Protocol
A stopped stream of terminal output must not be interpreted as proof of a running server.

After an authorized `npm run dev` command, classify the visible terminal state as `STARTING`, `READY/LISTENING`, `ERROR`, `EXITED` or `UNKNOWN`.

Recognize relevant terminal patterns using the permitted locally trained pixel-based model. Then independently verify that the requested application is reachable by navigating to the specified local URL and inspecting its rendered state.

If readiness is ambiguous, report uncertainty instead of declaring success.

### Browser and Functional Testing Protocol
1. Ground the browser and navigate to the requested development URL.
2. Inspect the fresh frame for the expected application and visible error indicators.
3. Ground the requested input field and button using current pixels.
4. Verify focus using available visual evidence; do not depend exclusively on a specific caret-blink frequency.
5. Enter the instructed test data and activate the intended control.
6. Capture subsequent frames and compare the observed state against the expected postcondition.
7. Return `SUCCESS` only if the postcondition is supported by evidence. Otherwise return `FAILURE` or `UNKNOWN`.

### Motor and Localization Benchmarks
Keep predicted centroid accuracy separate from cursor trajectory accuracy and final physical docking accuracy.

Report median and 95th-percentile localization error, target acquisition success, focus errors, false actions and click success. Test arbitrary aspect ratios, high-DPI coordinate transformations, varied target sizes and multi-monitor layouts.

Treat micro-tremor as an experimental parameter, not a requirement that overrides reliable target acquisition.

### End-to-End Acceptance Test (Controlled Todo App Workflow)
1. System 2 provides the project setup command and expected postconditions.
2. AVCS opens the terminal through visual grounding and submits the authorized command.
3. AVCS observes the command output and verifies project initialization.
4. System 2 generates the application code and provides the required input actions.
5. AVCS starts the development server and opens the specified local URL.
6. AVCS verifies that the expected application renders and reports visible errors or uncertainty.
7. System 2 instructs AVCS to add `Buy milk`.
8. AVCS locates the input and button, enters the text and activates the button.
9. AVCS captures fresh frames and verifies that `Buy milk` appears in the list.
10. System 2 receives the verified result and decides whether another action or correction is required.

Evaluate each stage independently as well as the success rate of the complete task. Preserve failure evidence and never count an unverified outcome as a successful run.

---

## R6 — Privacy and System 2 Boundary

Maintain a strict local visual-data boundary.

No raw frames, image crops or learned image embeddings may be exported across the System 2 text interface.

System 2 may receive only the minimum structured textual result required for orchestration, such as the target label, presence decision, confidence, safe-action eligibility and verification status.

Do not send images, embeddings or frame-derived payloads to external APIs. Audit logs must avoid storing sensitive screenshot contents.

The System 1 interface must expose uncertainty explicitly rather than presenting uncertain inferences as established facts.

---

## Benchmark and Acceptance Criteria

### A. Grounding quality
- Held-out top-1 target-grounding accuracy: `≥90.0%`.
- Median centroid error: `≤2.0 px` on the specified precision suite.
- 95th-percentile centroid error: `≤3.0 px` on the same suite.
- Adversarial hard-negative rejection: `≥95.0%` across at least 1,000 predefined held-out negative queries.
- Blank/background false positives: exactly zero on the declared test suite, including solid colors, gradients and varied wallpaper backgrounds.

Report exact counts and test-set sizes for all false-positive claims. Do not interpret zero errors in a finite test suite as a guarantee of zero errors in every possible environment.

### B. Spatial-collapse and geometry tests
Evaluate targets distributed across a balanced screen grid, including corners, edges and center regions. Compare predicted centroid distributions with ground-truth distributions and report localization accuracy for every region.

Detect center-collapse, degenerate heatmaps and uniform-background predictions using explicitly defined metrics and thresholds. Do not use an undefined variance statistic as the sole evidence that collapse is absent.

Repeat relevant tests at different aspect ratios, resolutions, DPI settings, target sizes and monitor positions.

### C. Anti-cheating and integrity
- Zero OCR calls or imports.
- Zero accessibility or DOM semantic inspection.
- Zero hardcoded target coordinates.
- Zero external vision-service inference.
- Zero image, crop or embedding export across the System 2 text boundary.

Run source inspection, dependency inspection and tests of the actual execution paths. Publish the commands, scope and limitations of the audits. A passing text search alone does not establish that a prohibited path can never execute.

### D. Latency and throughput
Run on the declared CPU using one inference thread or one CPU core, with hardware and software configuration recorded.
- Neural forward-pass latency target: `≤15.0 ms`.
- Report P50, P95 and P99 latency, throughput and cold-start performance where measurable.
- Measure preprocessing, coordinate decoding, refinement, input delivery and visual verification separately.
- Report complete-pipeline FPS independently from neural forward-pass FPS.

Treat sustained 60+ FPS end-to-end execution as a measured performance goal, not an automatically satisfied consequence of meeting the forward-pass target.

### E. Live desktop verification
Demonstrate successful localization and visually verified interaction on a real `1920×1200` desktop in a controlled test environment.

Record the initial frame, target annotation, predicted location, confidence, coordinate transform, resulting frame and task-specific verification decision. Keep this evidence local unless the user explicitly authorizes otherwise.

---

## Required Deliverables

1. Source code and version-pinned dependency manifest.
2. Reproducible training and evaluation commands.
3. Model checkpoints and documented loading procedure.
4. Dataset manifests, split definitions and annotation provenance.
5. Benchmark scripts and machine-readable prediction results.
6. Per-metric PASS/FAIL report containing measured values and test counts.
7. Integrity audit detailing prohibited-dependency checks and their limitations.
8. Live smoke-test results and failure-case examples.
9. Known limitations, unresolved bugs and performance bottlenecks.
10. A packaging and operation guide with dry-run, live-test and emergency-stop procedures.

---

## Required Final Report Format

Return:
- Task identifier.
- Current observed application state.
- Actions attempted.
- Target grounding and confidence.
- Expected postcondition.
- Actual visual evidence.
- Final status: `SUCCESS`, `FAILURE` or `UNKNOWN`.
- Suggested recovery action.

> **Honest limitation:** Do not claim that the entire application works perfectly when only a limited user flow has been tested.

---

## Delegation Protocol

Begin with a repository audit. Inspect the existing architecture, tests, model checkpoints and dependency graph before changing the implementation.

Create a shared interface contract so that grounding, presence evaluation, coordinate refinement and action verification use compatible data structures.

Delegate independent work only after approval. Integrate changes through explicit contracts, review them for conflicting assumptions and run the complete test suite after integration.

Do not delete working components or rewrite the architecture without evidence that the change is necessary.

Do not invent test results, claim a benchmark passed without executing it, or label an untested capability as implemented.

## Final Definition of Done

AVCS is complete only when the implementation, held-out benchmark evidence, integrity audit, live desktop demonstration and reproducibility package are all available.

If any acceptance criterion fails, report the failure honestly, preserve the evidence and identify the next engineering step.

**Final rule: measured capability over architectural claims. A fully trained, honestly evaluated model is preferable to a sophisticated-looking implementation that has not demonstrated real visual grounding.**


## 2026-10-17T06:52:29Z

The user has explicitly confirmed and granted approval: "yes".

The Approval Gate is now officially UNLOCKED.
Orchestrator 4 is authorized to proceed with Milestone 2 implementation, neural grounding head training on CPU, unified checkpoint packaging (`runs/weights/human_eye_weights.pt`), and multi-agent auditing (Reviewers, Challengers, Forensic Auditor). Please proceed immediately.
