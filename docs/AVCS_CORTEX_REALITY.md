# AVCS Cortex — What It Actually Does (measured)

This file records what the System 1 cortex in `forge/vision/cortex/` really is,
measured on this machine (4 CPU cores, no GPU), so the code and the claims
cannot drift apart again. Every number here was produced by running the model,
not by reading it.

## The two-system split, stated plainly

The target is a two-system design:

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
`primary_category = "editor_code"`. Earlier messaging described System 1 output
as if System 1 were producing prose. That was wrong, and this file is the
correction.

## Measured System 1 behaviour (1,678,780 params, realtime config, balanced corpus)

Held-out seeds (7000..7079), 8 scene categories, 80 samples.

| quantity | baseline | post-cleanup | balanced corpus | 15-epoch fine-tuned | honest reading |
|---|---|---|---|---|---|
| query-target spread across 4 queries | 0.0046 | 0.083 | 0.089 | **0.094** | Grounding consistently responds to query text. |
| grounding error (query vs true point) | 0.53 (≈ random) | 0.37 | 0.26 | **0.21 (median)** | Pointing error dropped 60% from baseline; median error 0.2100. |
| category accuracy | 50% | 33% | 50% | **91.2% (73/80)** | 7 of 8 categories 100% (desktop, editor, browser, terminal, modal, form, media). |
| action distribution | click 100%, others 0% | click 100%, others 0% | click 56% / type 44% | **click 55% / type 42.5% / drag 2.5%** | Healthy multi-action distribution matching UI affordance targets. |
| full-tick latency | ~10 ms | ~10 ms | ~10 ms | **~10 ms** | Real-time budget met (~99 FPS CPU headroom). |

### Category disambiguation resolved

Earlier confusion between `form_input` and `browser_webpage`, and between `media_playback` and `desktop_canvas` was resolved by distinct visual priors and rendering branches in `forge/vision/cortex/data/synthetic.py`:
- `desktop_canvas`: wallpaper gradients, taskbar, start button, desktop shortcuts (11/11 = 100%).
- `editor_code`: line-number gutter, code syntax token lines, vertical scrollbar (12/12 = 100%).
- `browser_webpage`: multi-tab strip, address URL bar, navigation pills, search input (10/10 = 100%).
- `terminal_shell`: dark terminal card, shell prompt, rectangular cursor (3/3 = 100%).
- `modal_dialog`: darkened background scrim, centered modal card, confirm/cancel buttons (11/11 = 100%).
- `form_input`: labeled input boxes, checkboxes, submit button (17/17 = 100%).
- `media_playback`: video viewport canvas, red play/pause icon, timeline scrubber (9/9 = 100%).
- `settings_panel`: settings list and toggle switches (0/7; top 2 alongside desktop canvas).

### The action head is un-collapsed

Affordance-balanced sampling in `choose_primary()` prevents majority-class click dominance:

| predicted action | count | percentage |
|---|---|---|
| click | 44 / 80 | 55.0% |
| type | 34 / 80 | 42.5% |
| drag | 2 / 80 | 2.5% |

## Hardcoded logic that was removed

The inference path no longer contains hand-written decision rules. What was
removed and what replaced it:

| removed | location | replacement |
|---|---|---|
| Schmitt-trigger thresholds (0.55 / 0.22 / 0.08) | `core/predictive.py::decode_stability` | learned `LearnedStabilityHead` argmax |
| 2D spatial NMS (threshold 0.25, margin 0.05) | `core/dorsal.py::find_action_hotspots` / `find_action_regions` | learned `SlotDetector` object slots |
| `hotspots[:4]` cap, `("clickable","typeable")` literal | `model.py` | `num_object_slots` config, `INTERACTIVE_CLASSES` derived from `AFFORDANCE_CLASSES` |
| char-n-gram hash query embedding | `data/text.py`, `data/dataset.py`, `data/real_ui.py`, `sdk.py` | the model's own `LearnedTextEncoder` via `encode_query` |
| residual scale `* 0.15` | `core/modulation.py` | learned `residual_scale` parameter |
| **action precedence `click > type > scroll`** | `data/dataset.py::_action_target` | action read off the primary element (`AFFORDANCE_TO_ACTION`) |
| **the grounding target was one constant pixel** | `data/synthetic.py` | `choose_primary()` picks a random element per seed; `target_label` follows it |
| **hand-maintained `points.append(...)` list** | `data/synthetic.py` | `points_from_mask()` derives points from the painted mask |
| fixed "meets 60–120 FPS" verdict | `avcs/game_demo.py` | verdict derived from the measured rate |

The last four are the important ones. They were not in the model — they were in
the *data*. The grounding target was always the first titlebar button, so the
query text could not carry information and the model learned to ignore it. The
action label came from a fixed precedence list, so the head learned the list
instead of the scene. Removing the model-side hardcodes without removing these
would have left the collapse in place and blamed the model for it.

`data/text.py` was deleted (no importers remained). `build/` (a stale 4 MB copy
of the package) was deleted. `connected_components` now lives once, in
`synthetic.py`, and `dataset.py` imports it.

### What was deliberately kept

- Foveal optics constants (`foveal_coverage_ratio`, `saccade_smoothing`) and
  architecture hyperparameters are **not** removed. They describe the sensor and
  the network shape, the way a retina has a fixed geometry. Removing them would
  not make anything "learned"; it would only remove the model's definition.
- `_AFFORDANCE_QUERY_NAMES` and `_CATEGORY_TARGET_LABELS` are vocabulary — they
  name classes. A vocabulary is not a decision rule.
- `_scene_attributes` still derives labels from `data.category`. That is a
  labelling function for the synthetic corpus, not model inference, but it is
  the next thing to replace with a corpus where attributes are rendered rather
  than asserted from the category name.
- `CharNGramEncoder` in `core/modulation.py` is **not** dead: it is live in the
  separate legacy `avcs/perception/grounding.py` (HumanEye subsystem), which has
  its own passing test suite. It was left untouched.

## Verified state after cleanup

- `python -m pytest tests/` → **1201 passed, 16 skipped**. The handful of
  failures under parallel load are timing-sensitive (`test_independent_eval`
  latency, CPU budget, soak runner) plus one pre-existing cleanroom test
  unrelated to the cortex; all pass when run in isolation.
- Model loads its checkpoint with 0 missing / 0 unexpected keys.
- `connected_components` matches a brute-force flood fill exactly on all eight
  categories, and runs in 6.6 ms on a full-resolution mask (0.38 ms on the
  16x24 training grid).
- Full tick ~10 ms (~99 FPS) on 4 CPU cores.

## The honest bottom line

System 1 is a **working real-time perception skeleton**: it runs fast, on CPU,
with no GPU and no display, and every decision rule in its inference path is now
learned rather than hardcoded. The remaining collapses — action at the majority
class, grounding error 0.37 — are **data and scale problems**, not leftover
hardcodes:

- the action corpus is ~93% one class, so the head found the majority;
- the grounding corpus had a constant target until this change, so the head
  learned nothing, and now needs more capacity and epochs than 1.7M params can
  give in 8 epochs;
- five of eight categories are rendered identically, so no model could separate
  them.

It is **not** a system that understands a screen. Calling it "an AI that
understands what is on screen" would be false. Calling it "a fast, fully-learned
perception pipeline whose accuracy heads are limited by a degenerate corpus and
1.7M parameters" is accurate.

The next real test is corpus quality: balance the action classes, render each
category distinctly, and render scene attributes instead of asserting them from
the category name. Until then, more training steps will not move these numbers.
