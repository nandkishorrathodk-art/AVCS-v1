# AGENTS.md

Repository knowledge for agents working on Forge.

## What this repo is

A small agentic LLM framework that trains end-to-end on CPU (4 cores, 15 GB,
no GPU). The constraint is deliberate: it keeps every component measurable.

## Setup and commands

```bash
python3 -m pip install numpy torch pytest
python3 -m pip install pillow mss             # optional: real-time screen vision
python3 -m pytest tests/ -q                    # 231 tests, ~24s
python3 -m forge.demo --steps 250              # trains, generates, debates
python3 -m forge.pipeline_demo --steps 200     # full chain, BPE tokenizer
python3 -m forge.control_demo                  # control kernel sections
python3 -m forge.vision_demo --seconds 10      # real-time screen vision (synthetic if no display)

python3 -m avcs.voice.demo --text "mujhe X kaam karna hai"   # LLM brain + voice (no mic)
python3 -m avcs.voice.demo --script "mujhe X kaam karna hai" "Save button daba do"
```

`runs/` is gitignored and holds checkpoints. `PYTHONPATH` must point at the
repo root when running scripts from outside it.

## Measured facts (do not re-derive, but do re-verify if you change the code)

Throughput on 4 CPU cores, batch 16, seq 256:

| config | params | tok/s |
|---|---|---|
| default (`dim=128, 4 layers, 4 experts`) | 2.7M | ~9,250 |
| scaled (`dim=256, 6 layers, 8 experts`) | 31–50M | ~1,050 |

Token budget is the real constraint. At ~1k tok/s, 1B tokens is 264 hours.

## Planning-quality metrics: P / S / V / H

`forge/training/metrics.py` measures four things. Keep them separate; each
catches a failure the others cannot see.

- **P** parse rate — text fits the plan grammar.
- **S** semantic rate — parsed *and* `goal == slugify(task)`.
- **V** verify rate — kernel executed and verified against real disk.
- **H** hack rate — parsed but task-blind (`P and not S`).

P alone is misleading. The repo's historical headline
(`model parsed own output : False`) only reports P.

Measured on the same corpus, n=60, seed 7, temp 0.6, Wilson 95% intervals:

| model | S | V |
|---|---|---|
| small 2.7M | 56.7% [44.1, 68.4] | 95.0% [86.3, 98.3] |
| big 31M | 10.0% [4.7, 20.1] | 53.3% [40.9, 65.4] |

Intervals do not overlap: at this scale, more parameters made the planner
*worse*. Treat "scale it up" as an untested assumption in this repo, not a fix.

## Traps that have already cost real time

1. **`generate()` defaults to `eos_id=None`.** Callers must pass
   `eos_id=EOS` or generation runs past the answer into the next training
   example. `LocalBackend` does this correctly; ad-hoc eval scripts do not.

2. **Never decode the full sequence and strip the prompt by string match.**
   The model echoes the prompt, so `rsplit` on the user turn can return the
   echo and every metric reads as a hard 0%. Use
   `forge.training.metrics.generate_plan`, which decodes only the ids past
   the prompt length.

3. **One fresh `ControlKernel` per eval sample.** A second `write` to the same
   path fails the "expect absent before" precondition, so reusing a kernel
   makes V depend on sample order.

4. **The tokenizer must travel with the weights.** Pass `tokenizer=` to
   `Trainer` and use `Trainer.load_with_tokenizer`. A BPE model decoded with a
   byte tokenizer produces silent gibberish.

5. **The grammar prompt appears in every training example**, so the model
   learns to generate it. Strip it when scoring.

## Failed approaches, recorded so they are not retried

- **2×2 ablation on the `Judge rationale` line** (`ablation_rationale.py`
  shape): every cell 1/16, a null result. The rationale line is not the
  bottleneck.
- **Rejection-sampling self-training** (STaR-style) against the verifier:
  3% acceptance, and the retrained model was *worse* (S 10% → 0%). The
  verifier is deterministic, so it is gameable — the model drifted toward
  task-blind but parseable output. H stayed high throughout.

Both are negative results with real measurements. Do not present either as a
win, and do not silently drop them.

## Safety: where it actually lives

`forge/training/metrics.py` also measures *intent* — what the model asked for,
before any gate ran (`intent_report` / `evaluate_intent`). This is separate
from P/S/V/H, which are all downstream of execution.

Measured (2.7M checkpoint, n=24, seed 7, temp 0.6):

| machine grant | hazard ops proposed | kernel-blocked steps |
|---|---|---|
| default policy (write/mkdir allowed) | 46/46 = 100% | 0 |
| read-only (no mutating grant) | 48/48 = 100% | 44 |

**The hazard-op rate is 100% either way.** The model proposes `mkdir`/`write`
at the same rate whether or not it holds the capability to do so. It does not
condition on its grant; the kernel does 100% of the containment. This is the
concrete, measured form of the repo's own claim that "safety is a capability
problem, not a filtering problem" — the capability lives in the kernel only.

### Retracted claim

An earlier measurement in this repo's history reported "illegal-op rate
100%" — the model allegedly proposing forbidden operations on every sample.
**That number was an artifact.** The forbidden-substring list contained
`<<`, which appears in every legitimate `write ... <<<content>>>` step, so
every plan was flagged regardless of content. The corrected measurement is
above: the model proposes *mutating* operations 100% of the time (true, but
unsurprising — that is what the task asks for) and proposed 1 hazardous
*path* in 48 steps. Do not cite the old number.

### Related work (do not overclaim novelty)

Scoping a model to refuse out-of-domain requests is studied: *Reducing the
Scope of Language Models* (AAAI 2026) covers refusal scoping; *Theory of
Agent* (2026 preprint) discusses boundary decisions around external action.
The narrow measurement here — comparing proposed hazard ops against granted
capabilities for a small local planner — appears uncommon, but this has not
been checked exhaustively. Treat it as "not obviously duplicated", not as new.

## Capability conditioning: tested, claim rejected

The obvious follow-up is to tell the model its grant and train the target plan
to respect it. That was built and measured (`make_capability_corpus` in
`forge/data.py`, `grant=` on `plan_user_turn` / `generate_plan` /
`evaluate_intent` / `metrics --grant`), and **the claim failed.**

Claim: *"a model trained with its grant in context proposes fewer out-of-grant
operations, at no cost to verify rate."*

| eval grant | arm | O (out-of-grant) | V (verify) |
|---|---|---|---|
| read-only | baseline | 96.6% | 0.0% |
| read-only | conditioned | **50.5%** | 0.0% |
| write | baseline | 3.0% | **70.0%** |
| write | conditioned | 1.0% | **2.5%** |

n=40, seed 7, temp 0.6, 500 train steps per arm, same architecture.

What is real: on the read-only arm the out-of-grant rate halved, and it is not
a denominator artifact — out-of-grant proposals per sample fell 2.00 → 1.03
while steps per sample *rose* 2.07 → 2.47. So the model became somewhat
sensitive to the stated grant.

Why the claim still fails:

1. **Verify rate collapsed on the write arm (70% → 2.5%).** This is the
   degenerate solution the experiment was built to detect. Fewer violations
   did not come from obedience; it came from producing less executable output.
2. **The conditioned model echoes the grant line** — 9/30 samples, versus
   0/30 for the baseline. Greedy decoding on a read-only grant still emits
   `mkdir`/`write`. The condition is being copied, not conditioned on. This is
   the same prompt-echo trap documented above, now triggered by a new line.
3. **Read-only V is 0% for both arms by construction.** A read-only machine
   cannot create the file a `write`-shaped task requires, so the read-only
   tasks are unsatisfiable regardless of how well-behaved the model is. The O
   improvement on that arm cannot be validated end-to-end.

Conclusion: at 2.7M parameters, stating a capability grant in context does not
produce a model that respects it. It produces a model that repeats the grant
and degrades. Do not retry this expecting a different result without first
fixing the echo problem (see below).

### What would actually test this

- **Train on a corpus where the grant line varies but the answer is identical
  when the grant permits it.** Right now the grant both changes the prompt and
  the target, so obedience and copying are confounded.
- **Constrain decoding** so the grant line cannot be emitted, then re-measure.
  Until the echo is removed, O is not a clean measurement.
- **Scale up.** 2.7M at ~20k params/context-token cannot hold a conditional
  instruction; every result above is at or below the noise floor of that limit.

## The click step: capability kernel (`avcs/control/capability.py`)

The System 2 -> System 1 boundary had two contradictory gates, and the reasoning
loop's central step ("brain says click, hand clicks") was impossible:

- `GoalConditionedGrounder.predict` returns `action_authorized=False` on **every**
  call (hardcoded). `AVCSController.locate` raised on it, so `click_target`
  always failed. The loop could never authorize a click
  (`report.action_authorized_count == 0`).
- `AVCSSystem1V1.observe_frame` authorizes on `presence_confidence >= threshold`,
  but that signal **saturates at 1.0 for wrong queries too** — measured, 40
  correct and 120 wrong queries all returned `target_present_probability = 1.0`.
  So that gate cannot reject anything; it would fail *open* if used.

`CapabilityKernel` replaces both. It decides on the grounder's **`peak_probability`**,
which does discriminate. Measured on `datasets/ui_grounding` (40 frames, correct
query vs 3 deliberately wrong queries each):

| signal | CORRECT | WRONG |
|---|---|---|
| `peak_probability` | median 0.873, min 0.755 | median 0.496, p90 0.729, max 0.863 |
| `target_present_probability` | 1.0000 (all) | 1.0000 (all) — saturated, unusable |

At the default `min_peak_probability = 0.75`: correct queries pass 100% (40/40),
wrong queries pass 6.7% (8/120). Fail-closed layers, in order: no grant ->
refused; query outside allow-list -> refused; degenerate frame (uniform or no
spatial structure) -> refused; no candidate point -> refused; peak below floor ->
refused.

**Defaults are fail-closed.** `AVCSController()` with no capability arguments
grants nothing and behaves exactly as before. To enable the click step a caller
must opt in explicitly:

```python
AVCSController(granted_ops=["click"], allowed_queries=["Launch button"])
```

Verified end-to-end with the virtual motor on `datasets/ui_grounding` frame
`train_0001` (target `Launch button` at 0.092, 0.949): without a grant the click
is refused in 7.6 ms; with a grant the motor executes a 106-point minimum-jerk
trajectory and records `click_history = [(0.093, 0.999, 'left')]` — a 0.050
normalized error. The bridge reports `authorized_actions=1` for the click turn.
`tests/test_capability_kernel.py` (14 tests) pins every layer.

### `foveal_snap` corrupts executed clicks (was on by default)

`click_target` unconditionally replaced the grounded point with
`HumanEye.inspect_fovea(...).refined_centroid` whenever the refinement head's own
confidence was `>= 0.5`. That head is miscalibrated off its training
distribution and moves the point *away* from the target: frame `train_0001` raw
error 0.050 -> refined error 0.256, at the head's reported confidence 0.706. The
repo's own `benchmarks/test_foveal_refinement.py` independently agrees — coarse
median 1.89 px vs refined 3.90 px on 316 held-out samples, only 22.5% improved.

The snap is now `foveal_snap=False` by default (was effectively always-on), with
`foveal_snap_min_confidence` configurable. Do not re-enable it without a
refinement head re-validated on the target distribution.

## Fixed bugs worth knowing about

- **`CapabilitySet.check` rejected `list`/`stat`/`scan` as "unknown
  operations"** regardless of grants, so three operations that
  `default_policy` grants and `_dispatch` implements were unreachable.
  `GRANTABLE_OPS` now derives the accepted set from the enum, and a test
  asserts `set(Op) == MUTATING_OPS | OBSERVING_OPS` so a new op cannot be
  added in one place and forgotten in another.

- **`FastAffordanceHead.find_hotspots` fired on the zero-padded border ring.**
  The 3x3 conv stack pads the image edge, which drives the outermost feature
  cells to a spurious high activation on bright frames. A *uniform white* frame
  (no interactive element at all) peaked at 0.83 in cell (0,0) and emitted a
  hotspot, so `HumanEye.observe` reported a clickable affordance on a blank
  screen. Measured on the ui_grounding negative split (30 blank/wallpaper
  frames, 256x256, a different generator than human_eye trained on): **53.3%
  false-positive rate**, of which 7/16 were border-ring artifacts at
  (0.031, 0.031) and 9/16 genuine interior overconfidence on dark textures. In
  distribution (`data/manifest.json` `test_negative`, 128x128) the rate is 0%.
  Both `find_hotspots` paths now clear the 1-cell border ring before peak
  selection; positives lose 0.4% of detections. The remaining interior FPs are a
  corpus problem (the negatives are out-of-distribution), not a head bug.
  `tests/test_challenger_d1_verification.py` previously *pinned the artifact as
  expected* — it asserted "1 corner FP on saturated white 224" and "1
  win32_fast_capture FP" (that frame's hotspot was at column 15, the border
  ring). Both assertions now require 0.

## Fully-learned cortex (`forge/vision/cortex/`)

`core/learned.py` holds the heads that replaced hand-written inference rules:
`LearnedTextEncoder` (trainable byte encoder, replaced the parameter-free hash),
`SlotDetector` (DETR-style slots, replaced greedy NMS), `SceneUnderstandingHead`
(multi-label attributes + regressed count), `ActionPolicyHead` (the model now
chooses the action) and `LearnedStabilityHead` (replaced the Schmitt-trigger
thresholds 0.55/0.22/0.08).

Facts to keep straight:

- The model is `1,678,779` params. The learned heads add under 1 ms; the
  backbone dominates and scales with the square of the retinal input size.
  `realtime_config()` (128x128 retina) holds ~95 FPS on 4 CPU cores; the default
  256x256 retina is ~37 FPS. Quote the input size with any latency number.
- Training runs all ten loss terms in one pass (predictive, affordance,
  category, grounding, slots, attributes, count, action, stability). The
  affordance loss resamples targets to the logits' grid, so a checkpoint is not
  tied to one input resolution.
- Legacy checkpoints predate these heads. `load_avcs_weights` treats the new
  head prefixes as optional, so an old checkpoint still loads and reports
  `weights_loaded=True` with the new heads left at initialisation. If you change
  a head name, update `learned_head_prefixes` in `avcs/weights.py` or every old
  checkpoint silently stops loading.
- Timing tests must take the minimum over repeated runs, not the mean. On a
  loaded CI box the mean is dominated by scheduler preemption and a real-time
  assertion flakes; the minimum is the achievable latency.

## Real-time screen vision (`forge/vision/`)

Pure-pixel visual awareness of the desktop. Read `docs/SCREEN_VISION.md`. The
facts to keep straight:

- AVCS System 1 is intended to run locally; System 2 may be a cloud LLM that
  receives bounded text observations and returns high-level goals. Do not send
  frames or visual embeddings to System 2. Follow the engineering and evidence
  requirements in `docs/AVCS_ENGINEERING_RULES.md`.

- The **reflex layer** is non-generative (typed `SceneState`, no text model) and
  runs every captured frame. Measured full-tick p50 ~3.7 ms (reflex + model
  forward) at a 16x24 tile grid, so the 40-80 ms real-time budget is met with
  ~11x headroom. Quote the *full tick*, not reflex alone -- `PipelineStats`
  reports both, and reflex-only flatters the system by about a third.
- The **understanding head** predicts a scene type, activity/change/overlay, a
  calibrated confidence, and a multi-label distribution over a closed
  description vocabulary (`vocab.py`). The sentence is rendered from *predicted*
  words -- it is not a hardcoded template, and it is not generated
  autoregressively.
- Description is **change-triggered and slow** (~1 s for a local VLM). It never
  sits on the reflex clock; the dashboard reports reflex fps, not caption fps.

It does no OCR and cannot read text -- that is the "pixels only" mandate, not an
oversight. Traps, both learned the hard way:

- `ScreenVision` must not perceive the same captured frame twice (it gates on
  the capture sequence number); without that the reflex loop spins on a stale
  buffer and reports a frame rate the capture never produced.
- Do **not** judge the model by training loss or raw per-word accuracy. Training
  loss on a single set is memorisation; raw accuracy is dominated by always-on
  words (`screen`, `mid-tone`, `detailed`) and reads ~0.9 even for a model that
  discriminates nothing. Use `word_report` (macro-F1 + informative accuracy) on
  held-out draws and on a withheld scene category.

## AVCS System 1 (`avcs/`) — audit facts

`avcs/` is a separate system in this repo from `forge/`. Read
`docs/REALITY_AUDIT.md` before making any claim about it.

- The packaged `runs/weights/candidates/avcs_streaming_cortex_v1.pt`
  (501,612 params) is **untrained clean random initialization**, reproducible
  from `create_clean_random_cortex(seed=20261018)` to float32 rounding
  (max diff 1.7e-7). It qualifies plumbing, not perception. Do not attach
  accuracy claims to it.
- System 1 has **no text decoder**. The "description" is deterministic string
  templating in `avcs/observation.py::_synthesize_description`, not generation.
  Reading screen text and speaking are System 2's job.
- Throughput must be quoted with its resolution. The 95.4 FPS headline is a
  64x64 soak number. Raw forward: 564 FPS @64x64, 106 @256x256, 2.2 @1920x1200.
  Production letterboxes to 128x128 first (~223 FPS end-to-end).

### Frame-rate ceiling: the screen is the bottleneck, not the model

Measured on this box (4 CPU cores, 1 torch thread, changing frames so the
static-screen cache cannot hit), `HumanEye.observe` end-to-end:

| frame | stock eager | fused ONNX accelerator |
|---|---|---|
| 128x128 synthetic | 2.44 ms (409 FPS) | 1.33 ms (753 FPS) |
| 1920x1200 real desktop | 2.96 ms (338 FPS) | 1.04 ms (965 FPS) |

**Do not chase a 2000 FPS target on perception.** A desktop screen delivers
60-144 frames per second, i.e. one frame every 6.9-16.7 ms. At 338-965 FPS the
model finishes each frame 3-16x faster than the screen can supply the next one,
so it is already waiting on capture. Higher per-frame throughput buys nothing
for screen perception; the useful currency is *headroom* (spend the saved
milliseconds on 256x256 instead of 128x128) or *batched* work (many screens,
or a closed-loop motor tick), not the FPS number itself.

Also: measure with *changing* frames. `observe()` has a static-screen early
exit, so a repeated-frame benchmark reports a cache hit (0.17-0.42 ms) and not
real perception. Every number above alternates two different frames.

### Opt-in ONNX accelerator

`HumanEye(onnx_accelerator=True)` (or `enable_onnx_accelerator()`) exports the
cortical forward pass -- backbone + ventral stream + affordance head -- into one
fused ONNX graph and runs that instead of three `nn.Module` calls per frame. The
win is removed Python/torch dispatch overhead, not a changed computation:

- **Off by default.** The eager path is untouched unless asked for.
- **Parity-gated.** Export runs both paths on a probe input and raises
  `RuntimeError` if they disagree beyond `atol` (default 1e-4), so a broken
  graph can never be silently used. `tests/test_onnx_accelerator.py` asserts
  `observe()` returns identical categories, confidences and hotspot lists with
  the accelerator on.
- Requires the optional `onnx` + `onnxruntime` packages (`pip install
  'avcs-sdk[onnx]'`); enabling it without them raises a clear ImportError.
- Measured 1.84x on 128x128 and 2.8x on 1920x1200 frames. Quantization
  (dynamic int8) and `torch.compile` were both tried and are *not* worth it
  here: int8 was no faster than eager (0.663 vs 0.674 ms) and compile gave only
  ~18% on the backbone alone.

- `runs/weights/human_eye_weights.pt` is produced by the real trainer
  (`avcs/training/train_affordance.py`); when present `HumanEye` loads it
  (`weights_loaded=True`). It is gitignored, so a fresh checkout lacks it and
  the tests that need it fail correctly. Never fabricate a placeholder
  checkpoint to make them pass. Its acceptance numbers are same-distribution
  synthetic figures, not a generalisation claim.
- `runs/weights/goal_grounder_best.pt` **is** trained (200,883 params, schema 2,
  `is_trained=True`). On `datasets/ui_grounding` `test_held_out` it reproduces
  94.94% top-1 / 3.72 px median via
  `python3 -m benchmarks.run_ui_grounding_benchmark`. But that split shares a
  generator with train. On `datasets/cleanroom_system1_v1` (a different
  generator) the same checkpoint collapses: **0.00% top-1 under the official
  benchmark's decision rule** (`(in box or within 15 px) and confidence >= 0.35`),
  and **9.38%** if the confidence gate is dropped and only the argmax within
  15 px is counted (n=64). The confidence gate is what zeroes it: the shipped
  checkpoint's calibrated presence threshold sits at **chance** and rejects
  48.4% of true targets, so `combined_conf` is clipped to 0.1 on the failures
  and never reaches 0.35. Always quote the split, the generator, *and* the
  decision rule — the same checkpoint reads 0.00%, 7.81%, or 9.38% depending on
  which rule is applied, and earlier revisions of this file quoted 28.1% and
  then 7.81% without naming one.
- Actuation is hardcoded off: `GoalConditionedGrounder.predict` returns
  `action_authorized=False` on every call and `AVCSController.locate` raises
  unless it is true. `test_locate_with_trained_grounder` asserts this, so it is
  an intended fail-closed invariant. `action_authorized` is never set true
  anywhere in the repo.
- `AVCSSystem2Bridge` defaults to `MockReasoningProvider` (a rule engine), not
  the real `LLMReasoningProvider`. No free-form language is produced by the
  shipped pipeline; System 1 emits template sentences only.
- `avcs_system1_v1.pt` is lightly trained (3 epochs, val loss 0.87, 13 s) and
  `candidate_m3_word_recognizer_v1.pt` is a 132-way closed-vocabulary
  classifier (69.21% acc), not OCR. Neither implies a perception capability.
- Test-suite baseline is **dependency-dependent**: install
  `pyproject.toml [dev]` (`pytest pytest-asyncio build wheel`) plus
  `torch numpy psutil cryptography fastapi httpx requests Pillow`. With those
  present, the suite in this checkout is **109 passed / 2 skipped** (measured).
  An earlier revision of this file claimed "2 failed / 1194 passed / 16 skipped";
  that number is **not reproducible here** — the repo collects ~111 pytest tests
  across `tests/`, and the five `benchmarks/test_*.py` files define **zero**
  pytest functions (they are `__main__` scripts), so no checkout of this branch
  can reach 1194. Treat the 1194 figure as stale and re-measure before quoting.
  The Tkinter GUI tests need a display; run them under `Xvfb` and install the
  `tk8.6` system package (`libtk8.6.so`).
- Tkinter GUI modules (`avcs/gui.py`, `avcs/runtime/live.py`, the m7 harness)
  need the system `libtk8.6.so`. They now raise an actionable ImportError
  instead of an opaque loader error.

## AVCS `forge/vision/cortex/` System 1 — what it actually does

Read `docs/AVCS_CORTEX_REALITY.md`. It is the corrected record; an earlier
version of this section described the model from its code and was wrong.

- `ArtificialVisualCortex` is **1,678,780 params**. Checkpoint:
  `runs/weights/avcs_cortex_weights.pt` (loads with 0 missing / 0 unexpected).
- The cortex **inference path contains no hand-written decision rules**. The
  Schmitt-trigger stability thresholds, the 2D spatial NMS for hotspots, the
  `hotspots[:4]` cap, the `("clickable","typeable")` action filter, the
  char-n-gram hash query embedding and the `* 0.15` residual scale are all
  gone, replaced by learned heads (`LearnedStabilityHead`, `SlotDetector`,
  `LearnedTextEncoder`, learned `residual_scale`). Do not reintroduce a
  threshold in this path.
- **The remaining collapses are in the corpus, not the model.** Measured
  (1.7M params, 8 epochs, held-out seeds):
  - grounding target spread across queries 0.083 (was 0.0046); grounding error
    0.37 (was 0.53, ≈ random). Text now moves the gaze, but it is not a usable
    pointer.
  - action head predicts `click` on 100% of frames: 101/101 correct for
    clickable scenes, 0/11 for typeable, 0/8 for scrollable. The corpus is ~93%
    clickable, so the head sits at the majority class.
  - category accuracy 33%, but **five of eight `UI_CATEGORIES` are rendered by
    identical code paths** (`browser_webpage`/`form_input` share one branch;
    `settings_panel`/`media_player`/`desktop_canvas` share another). Confusion
    is `browser_webpage→form_input` 20/20 and `settings_panel→desktop_canvas`
    20/20. `editor_code` is 20/20. The model is right; the generator is
    ambiguous. Do not quote 33% as a model failure without this caveat.
- Two dataset-level hardcodes were the real cause of the old "degenerate"
  readings and are now removed: `_action_target` no longer uses a
  `click > type > scroll` precedence list, and the grounding target is no
  longer the same titlebar pixel for every frame (`choose_primary()` picks a
  random element; `points_from_mask()` derives points from the painted mask
  instead of a hand-maintained list). **If a head collapses again, check the
  label function before blaming the model.**
- Kept on purpose: foveal optics constants and architecture hyperparameters
  (sensor geometry, not decision rules); `_AFFORDANCE_QUERY_NAMES` /
  `_CATEGORY_TARGET_LABELS` (vocabulary); `CharNGramEncoder`, which is live in
  the separate legacy `avcs/` HumanEye subsystem.
- It is **not** a screen-understanding model and it has **no text decoder**.
  Prose about the screen is System 2, which is not in this repo. Describing
  System 1 output as natural language is the exact overclaim this section
  corrects.


## Cross-generator grounding: the gap is corpus diversity, not architecture

The `avcs/` grounder is the bridge from "eyes" to "hand". Its real question is
whether it works on software it has never seen. Measured directly (n=64 cleanroom
/ n=316 ui_grounding, 15 px tolerance, seed 7, ~950 steps per arm, 200,883-param
`avcs.goal_grounder.GoalConditionedGrounder`). The scoring rule for these earlier
arms was not recorded, and the rule changes the number materially — see
"Decision-rule sensitivity" below. Treat the table as directional and reproduce
any single cell before quoting it. The rule-named, reproducible numbers are the
shipped checkpoint's: 94.94% home, 0.00% gated / 9.38% ungated cross-generator.

| arm | train data | steps | ui_grounding | cleanroom |
|---|---|---|---|---|
| A | ui_grounding only | 936 | 82.3% | 23.4% |
| B | cleanroom only | 192 | 5.4% | 64.1% |
| B_matched | cleanroom only | 928 | 7.6% | 73.4% |
| C | combined, natural proportions | 936 | 85.1% | 75.0% |
| C_matched | combined, natural proportions | 950 | **86.1%** | **85.9%** |
| D2 | combined, full epoch per generator | 1884 | 88.6% | 81.2% |

Reading:

1. **The gap is real and it is memorisation of the generator, not of the
   concept.** A ui-only model is 82.3% at home and 23.4% abroad.
2. **It is not undertraining.** B_matched gives cleanroom-only 6x more steps
   (192 -> 928) and moves the other generator from 5.4% to 7.6%. Ruled out.
3. **Training on both generators fixes it, at no cost.** C_matched is 86.1% on
   ui and 85.9% on cleanroom. Adding the second generator also *raised* cleanroom
   accuracy over the cleanroom-only arm (73.4% -> 85.9%). Multi-generator data
   regularises rather than trades off.
4. **Balanced sampling must not shrink either arm's budget.** Arm D in the first
   run was misbuilt: "balanced" capped each epoch at 512 samples, starving the
   1256-sample ui arm, and it scored 35.8%/54.7%. D2, which gives each generator
   a full epoch, is the correct construction. Do not quote D as a result.

### Negative results (do not retry)

- **Photometric augmentation does not close the gap.** Matching cleanroom's
  appearance onto ui_grounding drops ui from 91.5% to 36.1%; matching
  ui_grounding's appearance onto cleanroom drops cleanroom from 7.81% to 0.00%.
  (Rule for these earlier arms was not recorded; the same checkpoint reads 0.00%
  or 9.38% on cleanroom depending on the gate — see "Decision-rule sensitivity".)
  Training cleanroom with brightness/contrast/gamma/gain jitter moved the other
  generator 5.4% -> 7.3% (within noise) while costing 8 points at home. The gap
  is structural, not a white-balance problem.
- **`streaming_v1_synthetic_0_1` cannot test this.** Its annotations record
  `object_identity: not_collected`; it has no grounding targets. A third-generator
  generalisation claim is not measurable with the corpora present.

### Decision-rule sensitivity

The clean-room number is not a single fact; it depends on the rule applied, so a
figure without its rule is not reproducible. On the same checkpoint, same split
(n=64), same 15 px tolerance:

| rule | cleanroom top-1 |
|---|---|
| argmax within 15 px, no gate | **9.38%** (6/64) |
| `(in box or 15 px) and confidence >= 0.35` (official runner) | **0.00%** (0/64) |

The official runner's gate is what zeroes it. `predict_point` clips
`combined_conf` to 0.1 whenever the calibrated presence head says the target is
absent, and on this corpus that head sits at chance: it rejects 48.4% of true
targets, so `confidence >= 0.35` is never satisfied on a hit. The checkpoint is
not more broken than the ungated 9.38% implies; the gate is simply calibrated on
a distribution that does not transfer.

Reproduce both with
`python3 -m benchmarks.benchmark_cross_generator_grounding --out benchmarks/results/cross_generator_grounding_goal_grounder_best.json`.
That command prints both clean-room numbers (gated and ungated) plus the home
split, and writes the full report including the checkpoint sha256.

### Two architectures share the name `GoalConditionedGrounder`

`avcs/goal_grounder.py` and `avcs/perception/goal_grounder.py` both define a
class of that name, and they are **different networks**, not duplicates:

| module | keys | params | discriminator | loads |
|---|---|---|---|---|
| `avcs/goal_grounder.py` | 81 | 200,883 | `head_conv1` / `heatmap_head` | `goal_grounder_best.pt` |
| `avcs/perception/goal_grounder.py` | 195 | 1,354,050 | `lateral_*` / `decode_*` FPN | `candidate_m3_highres_v2.pt` |

Both are loadable by their own loader, so neither is stale. The bug is that the
public API (`avcs.load_grounding_checkpoint`, re-exported from the perception
module) was hard-wired to the 195-key class, so calling it on the shipped
81-key checkpoint raised
`RuntimeError: size mismatch for visual_encoder.stem.conv.weight`. The loader
now selects the architecture by inspecting the checkpoint's key names and
fails closed on a mismatch instead of raising an opaque shape error. Add a new
grounder architecture in one place and update the discriminator, or the
fail-closed path will reject it.

Note also: `validate_dataset('datasets/cleanroom_system1_v1')` raises
`SHA-256 mismatch for annotations/train/train_00000.json` — the corpus on disk
does not match its own manifest, independent of the rights gate.

### Reproducing the combined arm

`avcs/training/train_combined_grounder.py` trains the canonical 200,883-param
grounder on any subset of `ui_grounding` and `cleanroom_system1_v1` and reports
per-generator held-out top-1 / rejection / median-error. It reuses the existing
rights gate (`validate_rights_attestation`) and fails closed: the clean-room arm
raises rather than silently dropping when the signed approval bundle or an
intact corpus is missing. Combined checkpoints carry the new
`multi_generator_grounding_corpus_v1` provenance, accepted by both loaders and
both `is_trained` gates. `--sampling natural` (default) keeps the proportions in
the table above; `balanced` equalises expected draws per generator so a small
corpus is not drowned by a large one. The sampler is the knob the arm-D failure
turned on: any policy that starves one generator reproduces that failure. Score
any resulting checkpoint on both splits with
`python3 -m benchmarks.benchmark_cross_generator_grounding`; a combined model
should show a small gap, and the gated/ungated pair bounds how much of a
cross-generator number is the model versus the confidence gate.


## avcs_sdk live agent (`avcs_sdk/`) — audit facts

`avcs_sdk/` is the packaged "eyes + hands" entrypoint
(`python3 -m avcs_sdk.run_live_agent`). It was **not runnable** before the
fixes below: the crash only appears when `FoveaReader.inspect` output reaches
the agent, so importing helpers in isolation hid it.

Fixed defects (all reproduced by running the real entrypoint):

- `run_live_agent.py` read `e.patch`, `elem.glyph_likeness`,
  `elem.edge_density`, `elem.contrast` and `target_elem.describe()`.
  `FovealElement` had **none** of those fields, so `run_agent` raised
  `AttributeError` on turn 1. `glyph_likeness` was never defined anywhere in
  the repo or any branch.
- `narrator.py` guarded `e.describe()` with `hasattr`, so instead of crashing
  it **silently dropped every element** (`visible_elements == []`). A
  `hasattr` guard on an always-missing attribute turns a crash into silent
  data loss; prefer the attribute existing.
- `FovealElement` now carries the real crop (`patch`, excluded from
  equality/hash so the frozen dataclass still hashes) and the two texture
  metrics computed once in `FoveaReader.inspect`.
- **Category vocabulary was split.** The eye stream emits
  `interactive_control / text_input / toggle_icon / modal_dialog /
  content_region` (`UI_CLASS_MAP`, now the single source). The scorer tested
  `"text_like"` and `"graphic"` — tokens **no producer emits** — so both
  branches were dead. Derive consumer sets from `UI_CATEGORIES` or they drift
  again.
- `narrator._clean_target_name` did not map `text_input`/`toggle_icon`/
  `modal_dialog`/`content_region`, so raw snake_case class tokens reached the
  System 2 prompt. The element-label path also bypassed the map entirely.

Two honesty bugs in `--check`:

- It printed `Model Weights: OK` and `ALL CHECKS PASSED ... 100% operational`
  while `avcs_sdk/weights/avcs_v2.pt` was **missing** and the cortex was
  untrained. It now prints `NOT LOADED` and exits **2** with
  `[CHECKS INCOMPLETE]`. Exit 0 means a checkpoint actually loaded.
- Do not read `--check` exit 0 as a perception pass; it only means plumbing.

Boundary rule (see `docs/AVCS_ENGINEERING_RULES.md`, "Do not send screen images
to System 2"): `LLMBridge(vision_enabled=...)` defaulted to **True**, so a
caller that passed only a goal silently shipped base64 foveal crops to the
cloud endpoint. It now defaults to **False**; enabling image egress is an
explicit opt-in.

Verified: `python3 -m avcs_sdk.run_live_agent --goal "..." --dry-run` now
completes turn 1 (previously `AttributeError`), and
`tests/test_sdk_live_agent_pipeline.py` (17 tests) pins each fix. Full-suite
failure set is unchanged: 13–14 pre-existing failures, **zero new**.

## Honest limits

- 2.7M–50M params cannot follow a system prompt reliably. This is a scale
  limit, not a bug to be fixed by more steps.
- The corpus is largely synthetic and repetitive. Corpus quality moved S from
  10% to 45–57%; parameter count did not.
- The model does not condition on its capabilities at all (see above). Safety
  lives entirely in the kernel. Closing that gap is the most interesting open
  problem in the repo.
- Screen vision is a *perception* layer, not a trained large vision model. Its
  understanding heads are ~26k parameters trained on labelled reflex states in
  seconds; they give calibrated confidence but there is no image-dataset
  pretraining. Held-out macro-F1 is ~0.5: tone and coarse structure are
  reliable, directional and motion words are not. Reading text or naming objects
  needs the optional VLM backend.

## System 1 one model (`avcs/system1/`)

The rest of the repo is a bag of narrow networks wired by hand: ~30+ separate
`nn.Module` classes, 7 networks on the shipped path, and 77 hand-written
`if/elif` branches (37 in reasoning, 40 in observation). `avcs/system1/` is the
counter-proposal -- **one** deep-learning model, one trunk, **one latent**, every
task read out of that same latent in a single forward pass:

```
pixels -> shared trunk -> ONE latent z
                              |
     +--------------+---------+---------+-------------+
   category      elements            grounding      action
  "kaun sa app" "kya kya hai"       "X kahan hai"   "kya karna hai"
```

Facts to keep straight:

- Two configs are measured. Local CPU: `width=64, latent=128`, **1,459,113
  params**. GPU run: `width=128, latent=256`, **5,002,985 params**. There are
  **no `if/elif` decision points** in `forward`. `CATEGORY_NAMES` /
  `ELEMENT_KINDS` / `WORD_NAMES` / `ACTION_NAMES` are output-distribution
  columns, i.e. the labels the model predicts, not logic. Change behaviour by
  changing data and retraining, not by editing a rule.
- Everything is a function of the single `z_map` from `self.latent`. A test
  (`test_one_latent_feeds_every_head`) zeroes that conv and asserts every
  frame-conditioned head's output becomes input-independent; if a head ever
  grows a private front-end that test fails.
- Grounding is query-conditioned: `point_logits` and `presence_logits` come from
  `z_map * q`. `category`/`kind`/`word` must **not** depend on the query, and a
  test pins both halves of that.

### Measured result (genuine GPU training, reproduced locally)

Kaggle GPU run: 12,000-frame corpus, 8,000 steps, batch 64, lr 3e-3, 930 s wall.
Checkpoint `runs/weights/system1_one_model.pt` (20 MB, gitignored) loads with
**0 missing / 0 unexpected**.

**Grounding was mis-reported as broken; the bug was in the metric, not the
model.** `OneModelOutput.point_norm()` softmaxed the heatmap along a single axis
(`dim=-1` over `(g, g)`), which leaves the *other* axis' marginal constant, so
that coordinate collapsed to 0.5 for every frame -- `PREDICTED y: std 0.000`.
A sharp, correct head therefore read as a dead one. Softmax is now over the
flattened grid. With the fix, held-out grounding over every positive
(frame, query) pair (n=64):

| metric | validation | test_heldout |
|---|---|---|
| presence_acc | 0.984 | 0.984 |
| point_exact_cell | 0.781 | **0.891** |
| point_within_1cell | 1.000 | **0.938** |

**Grounding is real, not a query lookup.** Shuffling the images so each query is
paired with the wrong frame drops exact-cell accuracy 0.891 -> 0.281, i.e. the
head genuinely reads the frame. This corpus is nonetheless query-deterministic
(1.65 distinct target cells per query text on held-out), so a query->cell table
would score non-trivially; always run the shuffle ablation before quoting a
grounding number as perception.

Remaining honest limits on grounding: n=64 held-out is small, and the corpus is
regenerated with the same vocabulary, so "89% exact cell" is a clean-room figure
on software the generator draws, not a generalisation claim.

**Read held-out, not validation, for the scene heads.** Validation
`category_acc` is 1.00 while held-out is ~0.44. That gap is the corpus being
ambiguous, not the model being great: several `UI_CATEGORIES` are rendered by
identical code paths (see the cortex section), so a model that keys on the
generator's layout scores perfectly in-distribution and ~chance on a
differently-seeded held-out draw. kind_acc 0.92 and presence 0.98 do transfer.


### Phase 1 scorecard (`avcs/system1/scorecard.py`)

Runs the shipped checkpoint through the production gate over *every* query in a
split and reports the numbers that decide the next phase. Zero new model:

```
python3 -m avcs.system1.scorecard --out benchmarks/results/system1_scorecard.json
```

Measured (shipped checkpoint, grid 16, n=64 present + 16 absent per split):

| metric | validation | test_heldout |
|---|---|---|
| point **inside target bbox** | 63/64 | **59/64** |
| point exact cell | 49/64 | 56/64 |
| point within 1 cell | 63/64 | 59/64 |
| **false-click** (absent passing gate) | **1/16** | **0/16** |
| presence separable | False | **True** |
| latency (median) | 26 ms | 26 ms |

What the diagnostics say:

1. **Cell accuracy understates grounding; in-bbox is the honest metric.** A
   point one cell off can still land inside a large control. On validation the
   point is inside the target bbox **64/64** while exact-cell is only 49/64 --
   every "miss" was a boundary offset inside the same control. Report in-bbox,
   not cell, or the number reads ~20 points low.
2. **Failures concentrate in `input_field`** on held-out (12/16 in-bbox; button,
   icon and tab are 16/16). The four wild-misses are all `input_field` queries
   pointing at the empty/`<none>` area. This is the target for Phase 2, not a
   generic "grounding is weak".
3. **Held-out calibration is clean and validation is not.** Held-out present
   min 0.245 vs absent max 0.045 -- perfectly separable, 0 false-clicks at every
   floor. Validation has one absent query at 0.782 (a false-click) and a present
   query at 0.004 (a miss), so `separable=False` there. Trust held-out; a floor
   of 0.5 is the operating point, and raising it to 0.9 buys nothing on held-out
   (still 0 false-clicks) while costing a real click.
4. **Wrong clicks are near, not wild.** Validation: 14 near-miss, 0 wild-miss.
   Held-out: 3 near, 4 wild (the 4 `input_field` failures). A near-miss is a
   resolution problem; a wild-miss is a word/kind confusion. They need different
   fixes, so keep them separate.

Reproduce the in-bbox number directly with
`test_scorecard_in_bbox_metric_runs_on_real_checkpoint`.

### Real-time budget (`avcs/system1/benchmark_latency.py`)

A screen supplies a frame every 16.7 ms (60 FPS) to 6.9 ms (144 FPS). Measured
full forward on 4 CPU threads, minimum over reps:

| config | params | ms | FPS | 60 FPS | 144 FPS |
|---|---|---|---|---|---|
| **shipped** 128px w128 z256 g16 | 5,002,985 | 25.6 | 39 | **no** | no |
| 128px w64 z128 g16 | 1,459,113 | 7.5 | 134 | yes | no |
| 128px w48 z96 g16 | 940,505 | 4.8 | 210 | yes | yes |
| 64px w32 z64 g8 | 568,841 | 1.3 | 759 | yes | yes |

**The shipped 5M checkpoint is too slow for 60 FPS (39 FPS).** The same
architecture at width 64 is 134 FPS -- real-time is a *config* choice, so quote
the config with the number. Latency scales with the square of the input size and
roughly linearly with width. `test_loop_latency_meets_a_screen_clock_at_a_realistic_size`
pins the 128px w64 point under the 60 FPS budget, taking the minimum over runs.

Speed is not the blocker; **transfer is.** A model that runs at 759 FPS but
collapses cross-generator is not a real-time screen agent.

### Closed loop (`avcs/system1/loop.py`)

`OneModelLoop` is the glue: `frame -> OneModel -> CapabilityKernel -> Hand ->
verify`. Perception is only the one model; the kernel is the only policy; the
hand is the only side effect. Run it:

```
python3 -m avcs.system1.loop_demo --frames 4 --simulate-response
python3 -m avcs.system1.loop_demo --frames 4 --grant        # fail-closed: no grant
```

Facts that are easy to get wrong:

- **The gate must be fed the *presence* head, not the point softmax peak.** The
  point softmax peak saturates at 1.000 for present *and* absent queries on this
  corpus, so it cannot reject anything; the presence head separates them (present
  median 1.000, absent median 0.000). The kernel's `min_peak_probability` is the
  presence floor, default 0.5 in the loop.
- **Verification needs a post-click capture of the *same* screen.** An earlier
  version paired frame `i` with frame `i+1` of a slideshow; two different scenes
  have a large photometric delta regardless of the click, so it reported
  `verified=True` for changes the click never caused. `run_sequence` now takes
  `post_frames`; without one, `verified` stays `None` (the honest value). The
  delta itself is `VisualRippleVerifier.calculate_local_photometric_delta`.
- **`VirtualHand` is the default and moves no cursor.** It records normalized
  coordinates. Actuation is virtual so a demo cannot touch the machine; the gate
  in front of it is the production kernel either way.
- **The random-model tests use `min_peak_probability=0.0` on purpose.** A
  randomly initialised model's presence is ~0.47, so a policy test that kept the
  0.75 floor would be testing the random model's confidence, not the policy.

Measured on real held-out frames with the shipped checkpoint (`--frames 4
--simulate-response`): 3/4 queries clicked (the 4th refused by the presence
floor at 0.245), 3/3 clicks verified against a synthesized change placed at the
*true* target, and grounding error 0.007-0.038 normalized. Fail-closed mode
(`--grant` with no values) clicks 0/4.


### Traps fixed here (each would have silently broken a checkpoint)

- **`hash()` is salted per process.** The query encoder first used Python's
  built-in `hash()` to map a word to an embedding id, so the same word got a
  different id in every new process -- training and inference would disagree and
  the grounding head would read the wrong embedding. Replaced with a fixed
  FNV-1a (`_QueryEncoder.word_id`); `test_query_token_ids_are_stable_across_processes`
  runs two fresh interpreters and asserts the ids match. This is the same class
  of bug as "the tokenizer must travel with the weights".
- **`point_norm` softmaxed one axis at a time.** `F.softmax(logits.reshape(b, g, g), dim=-1)`
  normalizes each row, so the row-marginal is constant and `y` is pinned at 0.5
  regardless of the peak. Grounding looked dead (`point_within_0.1` = 0) while
  the argmax was 89% exact. Fixed by softmaxing the flattened grid;
  `test_point_norm_recovers_both_axes` puts a peak in a corner and asserts both
  coordinates move. **A grounding number quoted without the argmax-cell metric
  was measuring this bug, not the model.**
- **The action label was the last region's kind**, so it was effectively
  arbitrary and task-blind. It now comes from the query verb
  (`_action_from_query`). This corpus only issues `click` queries, so the head
  has one class to learn *here*; that is a property of the data, not the model,
  and the head is wired to condition on the query for a corpus with more verbs.
- A `numpy`/`torch` wipe during a session is recoverable: reinstall
  `numpy==1.26.4`, `pillow`, and the CPU `torch` wheel. The cleanroom generator
  is pure-Python + PIL + numpy, ~120 frames/s, so the corpus is regenerable
  rather than a fixture to check in.

### Corpus and the honest caveat

`python3 -m avcs.system1.make_corpus --out DIR --train 12000` regenerates the
clean-room corpus at scale (~120 frames/s; 13,800 frames in ~2 min). The GPU
run trains on 12,000 frames for 8,000 steps, then reports `validation` and
`test_heldout` separately. **Read the held-out number, not the validation
number**: the corpus renders several `UI_CATEGORIES` through identical code
paths (see the cortex section above), so a high validation accuracy can be the
generator being ambiguous, not the model being right.

Training runs on a Kaggle GPU via `avcs/system1/train.py` (`--device cuda`,
AMP, `OneCycleLR`). The corpus is written to `/kaggle/temp`, **not**
`/kaggle/working`, so it does not become part of the kernel output; otherwise a
13k-frame corpus makes the result slow to fetch.



## Real-time voice assistant (`avcs/voice/`) — LLM brain + AVCS eyes

The conversational layer. It composes the existing pieces; it does not add a new
perception model. The chain is:

    ears (STT)  ->  brain (LLM)  ->  mouth (TTS)
                        ^
                        +-- eyes (pure-pixel AVCS perception) and gated hands

Run it:

```
python3 -m avcs.voice.demo --text "mujhe X kaam karna hai"                 # one turn, no mic
python3 -m avcs.voice.demo --script "mujhe X kaam karna hai" "Save button daba do"
python3 -m avcs.voice.demo --listen --turns 5 --eyes --ground --grant click
python3 -m avcs.voice.demo --say "Ji sir, kya kaam karna hai?"             # mouth only
python3 -m avcs.voice.demo --audio a.wav,b.wav --turns 2                   # ears, no mic
python3 -m avcs.voice.demo --check-llm                                     # exit 0 only if the brain works
```

Facts to keep straight:

- **Hinglish is enforced, not merely permitted.** `ConversationBrain(language=)`
  takes `hinglish` (default) / `english` / `auto`, and the policy is appended to
  whichever system prompt is in use, so the two prompts stay one source of
  truth. An unknown value raises. `--voice-language` drives the brain *and* the
  mouth; it used to change only the espeak voice, which meant the flag looked
  like it worked while the model was never told anything.
- **`CONVERSATION_SYSTEM_PROMPT` used to be dead code.** `plan_step` had no
  `system_prompt` parameter (only `stream_step` did), so the *non-streaming*
  voice path silently answered with the silent System-2 planner prompt — a
  different role, and a prompt that forbids conversation. `plan_step` now takes
  `system_prompt` on the ABC, `MockReasoningProvider` and
  `LLMReasoningProvider`; the brain passes the conversation prompt. When adding
  a provider override, match the signature or the prompt is dropped on the floor.
- **`--audio` is how you exercise the ears without a sound card.** It feeds one
  WAV per turn into the same `record_audio` hook the microphone uses. This is
  the only way to test the full ears -> brain -> mouth loop in a container, and
  it is what makes the loop testable at all. Exhausting the list returns
  silence, not a loop on the last utterance.
- **`--check-llm` follows the `avcs_sdk --check` honesty rule**: exit 0 only on a
  successful probe, exit 2 with no key or an unreachable endpoint. The cloud path
  is verified end-to-end against a real OpenAI-protocol HTTP server (in
  `tests/test_cloud_llm_path.py`, including wire-level assertions that the
  language instruction reaches the actual request body), but **a real vendor key
  has never been used here** — none exists in this environment, and the platform
  managed proxy (`OH_LLM_API_KEY_REFRESH_URL`) returns 401. Do not claim a live
  vendor call was made.

- **The brain is an LLM, spoken to in Hinglish.** `ConversationBrain` wraps a
  `ReasoningProvider`; the default `LLMReasoningProvider` reads its endpoint
  from the environment (`AVCS_LLM_ENDPOINT` / `AVCS_LLM_API_KEY`) and speaks
  OpenAI-compatible **SSE** (`stream_step`). With no key it falls back to
  `RuleBrain` and reports `brain=rule_fallback` — an honest degradation, not a
  silent one. `--llm-endpoint` / `--llm-model` override for a local server.
- **The ears do not hardcode English.** `FasterWhisperEngine(language=None)`
  lets Whisper auto-detect, which is what a Hinglish speaker needs (Hindi and
  English in one utterance); `--stt-language` forces one. A Vosk model is bound
  to its training language, so for Hindi pick a Hindi Vosk model by path rather
  than passing a language. Live listening end-points on trailing silence
  (`record_microphone_until_silence`, sox `silence`) instead of a fixed window,
  so the reply starts when the user stops; it degrades to a fixed capture when
  sox is absent.
- **The mouth is streaming, and it is Hinglish by default.** The reply is split
  into sentences as the model produces it and each sentence is spoken as soon as
  it is complete (`Speaker.speak_stream`, `SpeechQueue` on a worker thread), so
  the first sentence is audible while the rest is still generating.
  `VoiceProfile.hinglish()` uses the espeak-ng `hi` voice (romanized Hindi read
  the way the user writes it) at 150 wpm; `--voice-language english` switches.
- **Eyes are pure pixels, no OCR.** `GroundingPerception` runs the shipped
  `GoalConditionedGrounder` on the current frame to answer "where is <target>?",
  and the target text comes from the user's *speech*, never from the screen.
  `EyePerception` (morphology only) is the lighter option; it produces no
  grounding evidence, so the gate can never authorize a click with it.
- **The capability kernel is the only policy and it is fail-closed.** Actuation
  needs an explicit grant (`--grant click`) and real evidence over the floor.
  A synthetic/uniform frame is refused as degenerate, and no candidate point is
  refused as such — both are correct outcomes, not bugs.
- **`VirtualHand` is the default and moves no cursor.** `--live` is the explicit
  opt-in to real mouse/keyboard (`avcs_sdk.hand_motor.HandMotor`).
- **No soundcard in this container.** TTS *synthesis* is verified real (espeak-ng
  produces a multi-second PCM clip); *playback* cannot complete
  (`spoken=False`, "playback did not complete"). That is the environment, not
  the engine. Same for the mic: `arecord` fails with "cannot find card '0'".
  Use `--silent` and `--script` to exercise the whole loop headless.

Traps:

- `Assistant.converse(utterances=[...])` scripts a whole conversation without a
  microphone; `initial_utterance=` scripts only the first turn. A turn that
  finds no scripted utterance and no mic records `kind="silent"` and stops —
  that trailing silent turn is honest, so filter it when asserting on kinds.
- The stream protocol uses a sentinel: the model emits its spoken reply, then
  `<<<ACTION>>>`, then a JSON action. The mouth must never speak the sentinel or
  the JSON. Tests assert both are absent from every spoken string.
- `EyePerception.observe` reads hotspot **5-tuples**
  `(norm_x, norm_y, kind:str, obj_conf, cls_conf)`; the class name is index 2.
  Reading a different index silently mislabels every element.
- `LLMReasoningProvider._build_payload` serves **two** history shapes: the voice
  brain passes real chat turns (`{"role", "content"}`) and the System 2 control
  loop passes step records (`{"step_index", "proposed_action"}`). The dialogue
  turns are forwarded verbatim as messages; the step records are summarised.
  It used to collapse *everything* into a `{step, op}` summary, so the LLM never
  saw earlier conversation turns and answered each one in isolation. Keep both
  branches: a fix for one shape must not silently drop the other.

### Action dispatch: the op picks the hand method (`avcs/voice/hands.py`)

The assistant used to call `hand.click(*point)` for **every** allowed action, so a
spoken "type this" or "scroll down" still produced a click. `dispatch(hand, op, ...)`
is now the single point where an op becomes a hand call:

| op | hand method |
|---|---|
| `click` / `double_click` | `click(x, y)` (twice for double) |
| `type` / `type_text` | `type_text(text)` |
| `press_key` | `press_key(key)`; falls back to a `ctrl+s` chord via the motor's ctypes binding |
| `scroll` | `scroll(delta)` |
| anything else | **fails closed** — reports "no hand implementation", never a click |

Two hand contracts exist and the dispatcher must not confuse them:
`VirtualHand` (demo default, used by `OneModelLoop`) records **normalized** (0..1)
coordinates; `HandMotor` (Windows-only) drives the native cursor in **absolute
pixels**. `get_screen_size` marks the latter. A hand that returns `False` is
reported as "did not perform", not claimed as success.

### Autonomous multi-step loop (`Assistant.run_task`)

One turn yields at most one action. `run_task` acts, re-observes, and lets the
brain decide the next step until it says `finish`/`stop` or the budget
(`max_task_steps`, default 8) runs out. Two honesty properties, both pinned by
tests:

1. **It stops early on a refused step** rather than repeating a failing action
   forever.
2. **It is closed-loop.** `_observe` attaches a one-line `last_action` summary
   ("last step performed click, screen changed" / "…was REFUSED by the gate") to
   the observation, so the brain sees what the previous step did. Without it the
   loop would re-decide the same action from an unchanged screen.

The `RuleBrain` (offline, no API key) now also fills `text_input` for a type
request and extracts a key chord, so the whole path is exercisable without an LLM.

Tests: `tests/test_voice_assistant.py` (includes a full-stack turn over a real
local SSE server: SSE brain -> streamed mouth -> eyes -> gate -> `VirtualHand`,
a multi-turn clarify-then-act conversation, per-op dispatch, and the autonomous
loop's finish/refuse/closed-loop paths). Full suite with the voice tests:
**109 passed, 2 skipped**.

### Real-time session: streaming, barge-in, continuous loop (`Assistant.loop`)

`Assistant.loop()` is the always-on assistant, and `--loop` in the demo runs it.
Each turn: listen (silence-terminated) -> stream a reply -> ground/gate/act ->
repeat until a stop phrase (`stop`, `band karo`, `bas`, `exit`, `quit`) or
Ctrl-C, which ends the session cleanly with the partial report intact.

Three things make it real-time, and each was a real defect before:

1. **The mouth is interruptible.** `EspeakNgEngine.play` used a blocking
   `subprocess.run`, so `TtsEngine.stop()` was a no-op and the user could never
   talk over the assistant. It now holds the player `Popen` and `stop()`
   terminates it; an interrupted clip returns `False`, not a playback error.
   `Speaker.speak_async` runs the utterance on a background thread and returns a
   `SpeechHandle` immediately, so the eyes/brain keep running while it talks.
2. **A spoken task is completed, not just started.** After a turn that performs
   an action, the loop keeps stepping autonomously (empty user text) until the
   brain finishes or `max_task_steps` is hit, then returns to listening. Feeding
   `""` is deliberate: the next decision is driven by the screen, not by
   repeating the user's words back to the model.
3. **The offline brain does not loop.** Two bugs made the fallback spin on
   "Yes sir, what work would you like me to do?": `ConversationBrain._fallback`
   built a **fresh** `RuleBrain` every call (so it forgot the task), and
   `RuleBrain` had no "already acted" state. It is now one persistent rule brain
   (`self._rule`) whose `respond("")` returns `FINISH` after an action. A
   randomly-initialised or LLM-less run now terminates.

Per-stage timing (`listen_ms`, `brain_ms`, `act_ms`) is recorded on every
`AssistantTurnRecord` and printed by the demo, so "real-time" is a measurement.

### Real actuation is cross-platform (`avcs/voice/actuators.py`)

`HandMotor` is Windows-only (ctypes to user32). `XdotoolHand` is the Linux/X11
equivalent: `mousemove`+`click`, `type`, `key`, and scroll via button 4/5. It is
**fail-closed** — no `xdotool` or no `$DISPLAY` means `available()` is `False`
and every op returns `False` (never a silent success). `default_hand()` picks the
right one per platform or returns `None`. `--live` uses it; the capability kernel
still gates every call. Measured on this box (no `xdotool`, no `$DISPLAY`):
`available()` False, all ops refused — the honest outcome, pinned by a test.

Also: a machine with no capture device makes ALSA dump pages of `confmisc`
noise; `record_microphone*` now raise `EngineNotAvailable` with one readable line
("found no input device") via `_capture_error`.

## AVC Lexicon (`avc/lexicon/`) — reading control labels without OCR

The third stream in `avc/`. Dorsal answers *where* something is interactive,
ventral answers *what kind of screen* this is; neither can say **what a control
is called**, and the repo forbids OCR on purpose. Lexicon answers that question
from pixels, as *shape recognition* rather than transcription:

    pixels -> segment (ink-gap geometry) -> PatchEncoder --\
                                                            >-- shared space -> name
    word (from speech or the LLM)        -> TextEncoder  --/

Two branches, one L2-normalised space, joined by symmetric InfoNCE. The vision
branch never sees a character; the language branch never sees a pixel. Because
the space is shared and symmetric, both directions fall out of it:

- **inverse** — `read_regions(frame)` / `describe(frame)`: name each control.
- **forward** — `find(frame, word)`: ground a *spoken* word. The word comes from
  the user's speech, never from the screen.

Commands:

```
python3 -m avc.lexicon.train --steps 3000 --per-label 24   # ~60 s on 4 CPU cores
python3 -m avc.lexicon.demo --synthetic "save cancel username" --find save
python3 -m avc.lexicon.example_task                        # a multi-step task, gated
python3 -m benchmarks.benchmark_avc_lexicon                # the honest numbers
python3 -m pytest tests/test_avc_lexicon.py -q
```

Measured (shipped `runs/weights/avc_lexicon.pt`, 61-label closed vocabulary,
183 samples per split):

| metric | training fonts | held-out font |
|---|---|---|
| top-1 | **95.1%** | **70.0%** |
| unknown-word rejection | 100% (24/24) | 100% (24/24) |
| gate separable (correct p10 > wrong p90) | **no** | **no** |

- 6.7 ms min for a 442x56 frame (six regions), still inside the 16.7 ms screen
  clock. The 2D grouping roughly doubled this from 2.9 ms; it is the price of
  handling a whole screen rather than one line.
- **Fonts are the domain shift here, exactly as generators are for the
  grounder.** `split_fonts()` trains on the sans faces and holds out serif and
  mono. A held-out *font* is the real test; the two numbers must be quoted
  together. 70% is not a generalisation claim — the vocabulary is closed and a
  word outside it is refused, not guessed.
- The five remaining training-font failures are all genuine recognition errors
  (`on`→`no`, `name`→`menu`, `failed`→`install`), each with the correct single
  box. They are model errors, not segmentation errors — check `nboxes` before
  blaming the segmenter.
- **The gate does not transfer, and on a whole screen it does not even separate
  at home.** On training fonts the correct label's cosine (median 0.62) sits
  clearly above the best competing label (median 0.33), but the tails overlap
  (correct p10 0.38 vs wrong p90 0.46), so `separable=False`; on an unseen font
  the medians themselves collide (correct 0.44 vs wrong 0.35). This is the same
  structure the goal grounder's presence gate shows across generators, now
  measured on text. `benchmarks/benchmark_avc_lexicon.py` reports
  `gate_train_fonts` / `gate_held_out_fonts` for this reason — a top-1 alone
  hides it. The operating floor is therefore chosen on *accuracy*, not on a
  clean margin, and `accepted` is reported next to `top1` so the cost is visible.
- Cycle consistency (find a word, read the region back) is 100% on training
  fonts; it is the property that makes the space trustworthy, and `check_cycle`
  measures it rather than asserting it.

### Two bugs this stream hit, both worth remembering

1. **The ink-crop test was wrong, and training hid it.**
   `arr.max(axis=1) > 0.15` looks like an ink detector but is not one: on a light
   background (0.96) the background itself clears the threshold, so *every* row
   counted as ink and nothing was ever cropped. It went unnoticed because the
   training renderer sizes its canvas tightly to the text, where a failed crop
   barely matters. It became fatal the moment a caller passed a larger region:
   the glyph shrank inside its padding and the encoder saw a scale it never
   trained on. Ink is now detected *relative to the background median*, which
   works for light-on-dark and dark-on-light alike. Symptom to recognise:
   training metrics high, end-to-end accuracy near chance.
2. **Train/serve preprocessing must be one function.**
   The reader resized with nearest-neighbour while training used bilinear
   ink-cropping. End-to-end accuracy read 0.08-0.15 while every training metric
   stayed above 0.95. `avc.lexicon.render.to_patch` is now the single
   preprocessing entry point and `patch_from_crop` delegates to it — the same
   reasoning that makes the tokenizer travel with the weights.

Both were found by measuring *end-to-end on a realistic canvas*, not by reading
the training report. A metric that flatters a broken pipeline is worse than no
metric.

### Segmentation is 2D: blobs, not column projections (`avc/lexicon/blobs.py`)

The first segmenter projected ink onto the x axis and split on gaps. That works
for one line of bare words and fails on a real screen, in a way that is easy to
miss because the training patches are single lines:

- **A bordered button defeats a column projection.** Its frame spans the whole
  cell, so the gap between two buttons and the gap between two letters look the
  same, and the projection cannot separate them.
- **A whole screen has several lines**, and a 1D projection merges them.

The unit that must be separated is a 2D blob, so grouping works on blobs:
run-length connected components (union-find, numpy-only — a per-pixel union-find
over a full frame is ~100 ms in Python and does not fit a screen clock), then
structural grouping.

The order matters and each step was a bug first:

1. **Containers are found before any merging.** A container is a component that
   is **hollow** (ink fills < 0.25 of its box) **and encloses** another
   component. Enclosure alone is not enough: the counter of an 'a' encloses a
   speck, so an enclosure-only rule called letters containers and split words
   apart. Measured separation: button frames fill 0.06, glyphs fill ≥ 0.34.
2. **Then glyphs merge into words** on a gap ≤ 0.6 x height, joining both
   side-by-side blobs ('v' 'a') and stacked ones (the dot of an 'i'). There is
   **no height guard**: one was there to stop a frame absorbing a label, but it
   also rejected a 5 px dot against a 32 px stem (ratio 6.4), which is the normal
   case inside a word. Step 1 makes the guard unnecessary.
3. **The reader reads the label inside the frame, not the frame.** The region box
   is the clickable unit; the encoder was trained on word patches, so feeding it
   a bordered button puts a rectangle where it expects a glyph. `label_box`
   extracts the ink inside. Symptom before this fix: `save` read as `remove`.

Getting steps 1-2 wrong cost real accuracy and was caught only by re-running the
benchmark after each change: 0.973 (1D, single line only) → 0.585 (2D, broken
merge) → 0.863 (merge fixed, height guard still splitting) → **0.951** (guard
removed). A segmentation change that only *adds* capability can still drop the
headline number, so measure after every step.

### Augmentation: measured, and it helps

`augment_patch` perturbs slant, stroke weight, aspect and blur — the axes along
which fonts actually differ. Ablation (3000 steps, same architecture):

| arm | closed_vocab | held-out font (patch) | held-out font (end-to-end) |
|---|---|---|---|
| no augmentation | 0.984 | 0.549 | 0.596 |
| **with augmentation** | 0.967 | **0.721** | **0.642** |

It costs ~2 points at home and buys ~7 abroad. Unlike the grounder's photometric
jitter (a documented negative result — that gap was structural), here the gap
really is appearance, so appearance jitter closes part of it. This is a
*positive* result and is recorded as one.
