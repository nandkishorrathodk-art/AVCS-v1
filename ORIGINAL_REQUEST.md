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
