# AVCS Engineering Rules: Local System 1

These rules define the intended boundary between AVCS System 1 and a separate
System 2 language model. They are engineering constraints, not claims that the
current implementation already satisfies every goal.

## Target architecture

```text
User task ----------------------> System 2 (cloud LLM)
                                      |          ^
                 high-level text goal |          | text observation/result
                                      v          |
Local RGB screen -------------> System 1 (local eyes + hands)
                                  local grounding, action, verification
```

- **System 1 runs on the user's desktop.** It captures and processes local RGB
  frames, tracks visual changes, grounds targets, and performs authorized
  mouse/keyboard actions.
- **System 2 may run in the cloud.** It receives the user's original task and
  bounded, text-only observations/events from System 1, then returns a
  high-level goal or asks for clarification. System 1 reports the outcome so
  System 2 can replan.
- **Do not send screen images to System 2.** This includes screenshots, crops,
  video frames, image encodings, or visual embeddings. A future change to this
  boundary requires an explicit product/privacy decision and updated tests.
- Use `TextOnlySystem2Client` for the System 1 → System 2 boundary. It sends a
  bounded, allowlisted JSON observation and a text task, not a frame or
  embedding, and requires the calling application to confirm user consent
  before transmission. Basic secret-pattern masking is best-effort, not a
  comprehensive data-loss-prevention guarantee. The separate legacy
  `AVCSCloudClient` performs remote visual inference and is not part of this
  architecture.
- Text is not automatically safe: observations must not unnecessarily include
  secrets or private content. Terms and conditions do not prevent disclosure.
  Minimize and redact sensitive values locally, make sharing visible to the
  user, and test the serialized payload for accidental image/frame fields.

## Pixel-grounded perception

1. Treat the local captured pixels as the source for claims described as
   **pixel-derived**. Never label OS metadata as visual recognition.
2. Do not use Tesseract, EasyOCR, PaddleOCR, DOM inspection, or accessibility
   trees as hidden substitutes for pixel perception or visual target
   grounding. If a separately labeled metadata integration is considered, it
   must remain opt-in and must not be presented as evidence AVCS recognized
   something from pixels.
3. A pixel-only component may report only what its inputs and validated model
   support. For example, motion alone cannot establish that a scene is a game
   rather than a video; dense edges do not mean text was read.
4. Every observation exposed to System 2 should carry frame sequence/freshness,
   source, confidence or uncertainty where available, and a clear distinction
   between observation, inference, and verified outcome. If the model cannot
   identify an app/object, report unknown rather than inventing a name.
5. Keep the fast reflex loop separate from slower semantic description. Slow
   inference must not block frame capture, reflex updates, or motor safety.

## Grounding and hands

1. Never encode screen-specific click coordinates as a target or silently
   default to the screen centre. Resolve a target from current visual evidence
   and keep geometry inside System 1; coordinates are frame-relative and
   expire when the frame/state changes.
2. A learned grounding peak is a **candidate**, not proof that the requested
   object exists. Require validated weights and held-out evidence; abstain when
   the target is absent, ambiguous, stale, or below a threshold calibrated on
   independent data. A `GroundingPrediction` with
   `action_authorized=False` must never be converted into a motor action.
3. Foveal refinement and normalized coordinates are techniques, not accuracy
   guarantees. Do not claim “±2 px”, sub-pixel accuracy, or general target
   detection without a reproducible benchmark across resolutions, scaling,
   motion, and unseen layouts.
4. Use smooth trajectories where the input backend and task benefit from them.
   Minimum-jerk motion is a motion-planning choice; it does not prove
   human-equivalent behavior, evade anti-cheat/security systems, or guarantee
   that hover menus remain open. Do not implement stealth or anti-detection
   behavior.
5. Never infer task success from dispatching an input. Capture a fresh frame
   and check a task-relevant postcondition with an independent oracle where
   possible. Optical ripple/change is evidence that pixels changed, not proof
   the intended action succeeded.
6. Fail closed for typing and other consequential input when target/focus is
   not validated. Caret-blink detection is one signal, not a universal or
   infallible focus proof. Financial approvals, payments, sending messages,
   deletion, and other consequential actions require explicit user
   confirmation unless a separately reviewed policy authorizes them.

## Streaming and performance claims

- A stream, a 60 FPS configuration, a 60-packet-per-second bus, and a 60 FPS
  measured perception loop are different things. Report which one was measured.
- `AVCSController.start_streaming()` consumes fresh capture sequences when the
  source provides them. Its metrics report processed/dropped frames and
  measured processing rate, plus p50/p95 processing-tick latency and
  inter-tick interval; the configured `target_fps` is not an achieved capture
  or perception rate. Use `require_real_capture=True` for desktop acceptance
  so unavailable hardware cannot silently turn the run synthetic.
- Treat `<15 ms` as a local reflex target only when explicitly configured and
  benchmarked. It is not a cloud round-trip, semantic-caption, planning, or
  end-to-end action latency guarantee.
- Benchmarks must state hardware, OS, capture backend, screen resolution,
  workload, warm-up, sample count, p50/p95 latency, observed FPS, and dropped
  or stale-frame behavior. Measure capture, full reflex tick, slow semantic
  analysis, and action verification separately.
- Do not lower a performance gate simply to make a failing run pass. Diagnose
  and report the measured result.
- System 2 returns a text goal only. `consult_system2()` does not execute that
  goal; any local action must be separately reviewed, grounded, authorized,
  and verified. Do not parse free-form model text as an input command.

## Evidence and change requirements

- Keep synthetic, recorded, and live-desktop results separate. Synthetic tests
  validate mechanics, not real-app recognition.
- Evaluate app identity, text recognition (if ever implemented), grounding,
  abstention, and action verification on held-out real desktop cases with
  independent labels/oracles. Include negative examples where a target is not
  present.
- A checkpoint loading successfully is not evidence of accuracy. Record model
  version, dataset provenance, evaluation split, and measured metrics.
- Update focused tests whenever a rule or interface changes. Never describe an
  aspirational capability as implemented, measured, or production-ready without
  evidence from the relevant acceptance test.
- “Biological” describes inspiration only unless a specific biological
  property is defined and measured. Do not use it as a synonym for human-level
  perception or reliability.
