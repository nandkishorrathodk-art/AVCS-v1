# AVCS Cortex — What It Actually Does (measured)

This file records what the System 1 cortex in `forge/vision/cortex/` really is,
measured on this machine (4 CPU cores, no GPU), so the code and the claims
cannot drift apart again. Every number here was produced by running the model,
not by reading it.

## The two-system split, stated plainly

The user's target is a two-system design:

- **System 1 (Eyes + Hands)** — `forge/vision/cortex/`. A small, fast,
  non-generative perception + servoing model. It consumes pixels and produces
  typed state: a scene category, object boxes, affordance hotspots, gaze
  coordinates, stability, and motor velocity. It does **not** produce natural
  language.
- **System 2 (Cognitive Brain / VLM / LLM)** — not in this repo. Turns System 1
  state into sentences ("a code editor is open, the search bar is in the
  header"). It is where reading text, naming applications, and talking like a
  human belong.

Any statement like "the model says a code editor is open" is a **System 1 +
System 2 combined** behaviour. System 1 alone never says it; it emits
`primary_category = "editor_code"`. Earlier messaging in this session described
System 1 output as if it were System 1 producing prose. That was wrong.

## Measured System 1 behaviour (2.7M params, realtime config, 4 epochs)

| quantity | measurement | honest reading |
|---|---|---|
| full-tick latency | ~10 ms → ~99 FPS (single process) | Real-time budget is met. The model is fast enough. |
| scene-category accuracy | 24/48 = 50% (6 classes, synthetic) | Above 16.7% chance, far below usable. |
| mean category confidence | 0.55 | Uncalibrated; confidence is not trustworthy. |
| query-target spread across 5 different queries | std ≈ 0.003 | **Language grounding is not working.** |
| action output | `click` for 48/48 frames | Degenerate: the head collapsed to one class. |
| object count | ~1.0–1.3 regardless of scene | Not counting objects; near-constant output. |
| affordance map nonzero fraction | 0.91 | Nearly the whole screen is "affordable" — no discrimination. |

### What these numbers mean together

1. **It is real-time but not yet accurate.** The engineering target (fast,
   local, learned, no GPU) is met. The perception target is not.
2. **Grounding is broken in practice.** Changing the query text from "search
   bar" to "close button" moves the predicted gaze by ~0.003 normalized units —
   noise. The `LanguageModulation` head exists and is differentiable, but at
   2.7M params and 4 epochs on this corpus it has not learned to use the text.
   This is the same scale wall the repo documents for the planner.
3. **Action and object heads collapsed.** A single action class and a
   near-constant object count are the classic degenerate solution. They are
   visible here, not hidden.
4. **Confidence is not calibrated.** 0.55 mean confidence at 50% accuracy
   means a caller cannot threshold on it safely.

## The hardcoded logic that was removed

The inference path no longer contains hand-written decision rules. What was
removed and what replaced it:

| removed | location | replacement |
|---|---|---|
| Schmitt-trigger thresholds (0.55 / 0.22 / 0.08) | `core/predictive.py::decode_stability` | learned `LearnedStabilityHead` argmax |
| 2D spatial NMS (threshold 0.25, margin 0.05) | `core/dorsal.py::find_action_hotspots` / `find_action_regions` | learned `SlotDetector` object slots |
| `hotspots[:4]` cap, `("clickable","typeable")` literal | `model.py` | `num_object_slots` config, `INTERACTIVE_CLASSES` derived from `AFFORDANCE_CLASSES` |
| char-n-gram hash query embedding | `data/text.py`, `data/dataset.py`, `data/real_ui.py`, `sdk.py` | model's own `LearnedTextEncoder` via `encode_query` |
| residual scale `* 0.15` | `core/modulation.py` | learned `residual_scale` parameter |
| fixed "meets 60–120 FPS" verdict | `avcs/game_demo.py` | verdict derived from the measured rate |

`data/text.py` (the whole module) was deleted: after the switch it had no
importers. `build/` (a stale 4 MB copy of the package) was deleted.

### What was deliberately kept

- Foveal optics constants (`foveal_coverage_ratio`, `saccade_smoothing`) and
  head-architecture hyperparameters are **not** removed. They describe the
  sensor/architecture, the way a real retina has a fixed geometry. Removing
  them would not make anything "learned"; it would only remove the model's
  definition.
- `CharNGramEncoder` in `core/modulation.py` is **not** dead: it is live in the
  separate legacy `avcs/perception/grounding.py` (HumanEye subsystem), which
  has its own passing test suite. It was left untouched.

## Verified state after cleanup

- `python -m pytest tests/` → **1201 passed, 16 skipped**. The 5 failures under
  parallel load are timing-sensitive (`test_independent_eval` latency, CPU
  budget, soak runner) plus one pre-existing cleanroom test unrelated to the
  cortex; all pass when run in isolation.
- Model loads its checkpoint with 0 missing / 0 unexpected keys.
- Full tick ~10 ms (~99 FPS) on 4 CPU cores.

## The honest bottom line

System 1 is a **working real-time perception skeleton**: it runs fast, on CPU,
with no GPU and no display, and every decision rule in its inference path is now
learned rather than hardcoded. It is **not** a system that understands a screen.
Its grounding head does not yet respond to language, its action and object heads
have collapsed, and its category accuracy is 50%. Calling it "an AI that
understands what is on screen" would be false. Calling it "a fast, fully-learned
perception pipeline whose accuracy heads have not converged at 2.7M params" is
accurate.

The next real test is not more hardcode removal — it is whether the grounding
and action heads can be made to learn anything at all at this scale, or whether
that too is the documented parameter-count wall.
