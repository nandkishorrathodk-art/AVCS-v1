# AGENTS.md

Repository knowledge for agents working on Forge.

## What this repo is

A small agentic LLM framework that trains end-to-end on CPU (4 cores, 15 GB,
no GPU). The constraint is deliberate: it keeps every component measurable.

## Setup and commands

```bash
python3 -m pip install numpy torch pytest
python3 -m pip install pillow mss             # optional: real-time screen vision
python3 -m pytest tests/ -q                    # 327 tests, ~40s
python3 -m forge.demo --steps 250              # trains, generates, debates
python3 -m forge.pipeline_demo --steps 200     # full chain, BPE tokenizer
python3 -m forge.control_demo                  # control kernel sections
python3 -m forge.vision_demo --seconds 10      # real-time screen vision (synthetic if no display)

python3 -m avcs.voice.demo --text "mujhe X kaam karna hai"   # LLM brain + voice (no mic)
python3 -m avcs.voice.demo --script "mujhe X kaam karna hai" "Save button daba do"
python3 -m avcs.voice.demo --get-ears             # download a Vosk model (once)
python3 -m avcs.voice.demo --check-llm            # exit 0 only if the brain works
python3 -m avcs.voice.demo --listen --eyes --ground --grant click   # live session
python3 -m avcs_sdk.run_live_agent --goal "save the file" --dry-run # packaged agent
```

The brain resolves its key from the user's environment, then from the
platform-managed key, so it is live without any setup in a managed runtime. With
no key anywhere it degrades to the offline `RuleBrain` and says so
(`brain=rule_fallback`).

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
loop's finish/refuse/closed-loop paths). Full suite: **209 passed, 2 skipped**
(`tests/test_live_assistant.py` adds 18 for the live-brain wiring below).

### Live LLM brain: the key is resolved, not pasted (`avcs/reasoning/credentials.py`)

The brain reads its endpoint/key/model from, in order: explicit arguments, the
user's environment (`AVCS_LLM_API_KEY` and the vendor aliases), then the
**platform-managed key** (`SESSION_API_KEY` + `OH_LLM_API_KEY_REFRESH_URL`,
fetched once and cached in-process). That last source is what makes
`ConversationBrain()` live out of the box inside a managed runtime instead of
silently degrading to `RuleBrain`. Nothing logs the key; `mask()` is the only
rendering used in messages. An explicitly empty `api_key=""` means **force
offline** — the platform key is not substituted — which is how the keyless tests
stay keyless.

Four bugs this path exposed, all fixed:

- **`verify_live_connection` crashed on reasoning models.** `deepseek-v4-*` (and
  any R1-style model) can return `content: null` with the answer in
  `reasoning_content`; indexing `["content"]` and calling `.strip()` raised
  `'NoneType' object has no attribute 'strip'`, so a healthy endpoint reported
  `FAILED`. `extract_message_text` falls through `content` / `reasoning_content`
  / `reasoning` / `text`. The probe's `max_tokens` also went 32 → 512: a tight
  budget is spent on hidden reasoning and leaves `content` empty.
- **The non-streaming conversation path used the wrong schema.** It called
  `plan_step`, which parses the System 2 *planner* contract (`next_intent`). The
  conversation prompt emits `speech`/`kind`/`action`, so the model's action was
  silently dropped and every turn became a bare `observe_target`. The brain now
  calls `raw_step` and parses the conversation contract itself. (The streaming
  path was always correct; only `respond()` — used by the GUI and non-stream
  callers — was wrong.)
- **The stream parser only read `action["op"]`.** Models paraphrase the keys
  even when the prompt names them (`type` for `op`, `target` for
  `target_description`), and a valid action was downgraded to a reply. Both
  spellings are accepted now.
- **The task was sticky.** `if user_text and not self.task` meant the first
  utterance stuck for the whole session, so "ab close karo" after "save karo"
  still clicked save. A new spoken instruction now replaces the task; an
  autonomous step (empty text) keeps it.

`avcs_sdk.llm_bridge.LLMBridge` uses the same resolver and extractor, so the
packaged `python3 -m avcs_sdk.run_live_agent` is live too (`planner=llm`).

### A natural-language target reaches the closed vocabulary (`LexiconPerception`)

An LLM names targets the way a person does — "the Save button", "Edit menu" —
which is rarely the bare vocabulary word (`save`, `edit_menu`). `_match` tries
candidate forms in order (strip leading filler, strip generic suffixes
`button`/`field`/`menu`/…, try the head word, try `_`↔space both ways) and only
then matches against a read label, still fail-closed on an unknown word. Before
this the demo said "Ji, main save kar deta hoon" and the gate denied with
`grounder produced no candidate point`, because the model said "save button" and
the reader's label is "save".

### The mouth can be the browser (`BrowserEngine`)

The server is a container with no sound card; the browser has one. `BrowserEngine`
still synthesizes with espeak-ng (same voice) but parks the newest WAV for the
page to fetch (`GET /api/speech.wav`, `X-Speech-Seq`) and play through the Web
Audio API, instead of handing it to `aplay`. `play()` returns `True` because the
clip *was* delivered to the client's mouth. `--mouth browser|local|silent` picks
it; `browser` is the default.

### The ears find their own model (`avcs/voice/stt.py`)

`default_stt_engine()` now calls `find_vosk_model()`, which looks in
`AVCS_VOSK_MODEL`, then `models/`, `~/.cache/vosk`, `/opt/vosk`, preferring a
Hindi/Indian-English model. `python3 -m avcs.voice.demo --get-ears` downloads a
small model into `models/` (urllib + zipfile, no new dependency). So ears are
zero-config after one command, and `--audio a.wav,b.wav` exercises the whole
ears -> brain -> mouth loop headless.

Measured live on this box (managed proxy, `deepseek-v4.1-flash`):

| check | result |
|---|---|
| `--check-llm` | exit 0, ~284 ms, `{"status": "verified"}` |
| `respond("mujhe save karna hai")` | `act` / `click` / "Save button", `brain=llm` |
| `respond("tum kaun ho")` | Hinglish reply, no action |
| GUI `/api/turn` save→close→undo | each grounds at its own point (57,68)/(152,66)/(250,66) |
| `/api/turn.stream` | `speech` event then `turn` event; speech WAV 7.3 s @22.05 kHz |
| `/api/hear` on an espeak WAV | vosk transcript returned |
| `--audio` loop | ears → live LLM → mouth, then `stop` ends the session |


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

### Carrying a task through: the loop had three bugs, all measured live

An assistant that answers but never *finishes* anything is a chat bot. Three
separate defects made every task end after one glance; each is fixed and pinned.

1. **Keyboard ops were gated on a grounded point.** `CapabilityKernel.authorize`
   required a candidate point for *every* mutating op, but `type` / `type_text` /
   `press_key` / `scroll` have no coordinate — they act on the focused widget or
   globally. The natural "type the path, press Enter" route was therefore refused
   with "grounder produced no candidate point" for an op that never had one.
   `POINT_OPS` (click / double_click only) now names the ops that must be
   grounded; the rest need only their grant.

2. **The autonomous loop broke on `not performed`, which is what an observing
   step reports.** `observe_target` actuates nothing, so `performed=False` — that
   is success, not failure. Breaking there ended the task the moment the brain
   looked at the screen. The loop now breaks only on a *refused mutating* step
   (continuing would repeat a failing action) or a pure conversational turn (no
   action to carry forward).

3. **The brain was never told what it could do.** The observation carried the
   screen but not the capabilities, so the model could not tell a held click
   grant from none and stalled on `wait`. `to_brain_dict(capabilities)` now
   includes `can_click` / `can_type` / `can_press_key` / `can_scroll` /
   `can_see_targets`, derived from **hand presence AND grant** — a grant with no
   hand still refuses every click, so reporting `can_click=True` there would be a
   lie the model would act on.

**The GUI and the voice demo share one loop.** `avcs/gui/server.py::task`
delegates to `Assistant.run_task`; a second copy of the loop in the GUI is how
the two drift apart. The GUI's own copy had already lost the
`if not record.action: break` guard, so a greeting streamed eight identical
replies before this was caught (measured live, then fixed).

**A repeated dead action now stops the loop.** When the screen never changes
(e.g. a click with no visible effect), nothing in the model's input says "give
up", so it repeats to the step budget. Two identical no-effect steps is the
limit; the third stops the loop with `kind="stop"` and a `detail` naming the
action and the repeat count, so "gave up" is distinguishable from "finished".
The stop record deliberately carries **no** `action`: a record with an action
means an action was attempted, and a caller can count attempts without
subtracting the stop.

Measured live (LLM brain, `--live --grant click type press_key scroll
--eyes lexicon`):

```
you   : save button daba do
avcs  : Ji sir, save button daba raha hoon.        -> click, performed, no change
avcs  : Save pe click kiya par kuch change nahi, ab Ctrl+S try karta hoon.
        -> stopped: click repeated 2 times with no change
```

```
you   : file manager kholo
avcs  : Theek hai sir, keyboard shortcut se try kar raha hoon.   -> press_key x2
avcs  : Sir, shortcut se file manager nahi khul raha, screen par sirf
        close, ok, save jaise buttons dikh rahe hain. Aap bataiye ...?  -> ask
```

Both are honest: it tried, reported that nothing changed, and asked instead of
repeating or claiming success. A greeting is one turn, not eight.

**Test trap fixed here:** `test_loop_continues_autonomously_after_a_spoken_task`
defined a `_ScriptedBrain` and then never passed it to `Assistant`, so the test
ran the *default* brain — which resolves a real `LLMReasoningProvider`. It passed
only because a keyless provider falls back to `RuleBrain`, and it took 8 s
because of the connection attempt. It now uses `RuleBrain` directly and is
deterministic (1.5 s). Do not let a test's scripted double go unwired.

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

## The GUI model (`avcs/gui/`) — the click step now closes

`python3 -m avcs.gui.run --source desktop --port 12000` serves one page: the
model's own view of the screen, a conversation, and a gate log.

```
python3 -m avcs.gui.run --source desktop   # synthetic desktop with real controls
python3 -m avcs.gui.run --source screen    # the real screen
python3 -m avcs.gui.run --live             # real actuation (needs xdotool / Windows)
python3 -m avcs.gui.run --source screen --eyes lexicon --servo --live --proactive
                                           # the whole thing: pixel eyes, human
                                           # motion, real actuation, watches and
                                           # speaks first
```

### The bug this fixes: the click step was unreachable

Before this, the assistant could name an action but never carry one out. The
voice demo showed it plainly:

```
you   : mujhe save karna hai
avcs  : Okay sir, click on save.
action: click target='save'
gate  : DENY (grounder produced no candidate point)
```

Two causes, both structural:

1. `GoalConditionedGrounder.predict` returns `action_authorized=False` on every
   call, and `GroundingPerception` needs that trained grounder, whose
   confidence gate collapses off its training generator (0.00% gated
   cross-generator, documented above). So the kernel never received a candidate
   point and denied for the honest reason.
2. Nothing connected the lexicon reader to the assistant. `grep` confirmed
   `avc/lexicon` was referenced only by its own benchmark and tests.

`avcs/voice/lexicon_perception.py` closes both. `LexiconPerception` implements
the `Perception` protocol (`observe`, `frame`) **and** `grounding_evidence`,
which `Assistant._act` already looked for. It reads the frame once
(`LexiconReader.read_regions`), then answers "where is <word>?" with the centre
of the box that read as that word, at the reader's own confidence.

Measured on the synthetic desktop (900x520, controls `save/close/undo/redo`),
real kernel, `granted_ops=["click"]`:

| utterance | target | gate | point |
|---|---|---|---|
| mujhe save karna hai | save | **ALLOW** | [57, 68] |
| ab close karo | close | **ALLOW** | [152, 66] |
| undo karo | undo | **ALLOW** | [250, 66] |
| upload karo (absent) | upload | DENY | — |
| print karo (out of vocab) | print | DENY | — |

The two refusals are different and both correct. `upload` is a real vocabulary
word that is not on screen; `print` is **not in the closed 61-word vocabulary at
all**, so there is no prototype to match and it is refused rather than snapped to
the nearest label. Put `upload` on screen and the same utterance is allowed.

### A Hinglish filler bug that made actionable requests look unactionable

`RuleBrain._STOPWORDS` had no temporal or address words, so "ab close karo"
("close it now") extracted the target **`ab close`** — a phrase no control is
called. The gate then denied a request that was perfectly clear. Added the
fillers the user actually speaks (`ab`, `abhi`, `phir`, `jaldi`, `zara`, `sir`,
`ji`, `plz`, …). After the fix `_extract_target("ab close karo") == "close"`.
This is the same class of bug as the prompt-echo trap: a token that is not the
target becoming part of it.

### Honesty rules the GUI keeps

- **No OCR**, unchanged: the boxes come from glyph-shape matching against a
  closed vocabulary. Nothing transcribes characters.
- **The boxes are the model's own regions**, from the same `read_regions` call
  the gate uses — not a hand-drawn overlay. A box on the page is evidence the
  reader saw that control. A label below the confidence floor is drawn yellow.
- **A refusal is shown as a refusal**, with the kernel's own reason string.
- **Virtual by default.** The GUI attaches a `VirtualHand`, which records the
  coordinate a granted click resolved to and moves no cursor. Real actuation
  needs `--live` and a hand that actually exists.
- **Provenance per turn**: `brain_source` (`llm` or `rule_fallback`),
  `gate_allowed`, `gate_reason`, and per-stage milliseconds, so "real time" is a
  measurement. On this box the page reports ~34 ms/frame for the full
  grab+read+paint path at 900x520.
- **The streamed reply is shown once.** The SSE turn emits the reply as `speech`
  deltas (shown immediately), then a `step` record whose `said` is that same
  sentence. The `step` branch used to leave the streaming bubble in place, so the
  assistant appeared to repeat itself on every turn — a rendering bug, not a
  second utterance. The branch now removes the streaming bubble first;
  `test_streaming_page_clears_the_streamed_bubble_when_a_step_lands` pins it.
- **The pane says "live screen" only when the pixels came off a screen.** This
  was a real overclaim. `EyeStream.grab_screen` silently returns a flat 128-grey
  buffer when `mss` is missing or the capture device fails, and that buffer is
  indistinguishable from a real grey screen. The page title said "live screen"
  either way. `EyeStream.capture_is_live` / `LexiconPerception.capture_is_live`
  now expose the distinction, `/api/capabilities` reports `capture: live |
  synthetic`, and the pane title switches to "synthetic view" when it is not
  live. This is the same class of bug as the `avcs_sdk --check` false "OK": a
  capability reported as present when the machinery behind it is absent.
  Measured on this box before the `mss` install: `capture=synthetic`, frame
  `min=max=128`, and `--source screen` described the same blank buffer as
  `--source desktop` did. After installing `mss`: `capture=live`, frame
  `(800, 1280, 3)` with 251 distinct values.

### Real GUI applications: what the reader actually does on them

Measured directly on this box (Xvfb `:99`, real `ScreenCapture`, shipped
`avc_lexicon.pt`):

- **The reader works on clean in-distribution text.** A PIL canvas in DejaVuSans
  (the training face) reads `save` / `cancel` / `close` at 0.46-0.64 confidence.
  That is the model doing its job.
- **It does not transfer to a real browser window.** A chromium page with
  `<button>Save</button>` etc. is read as `country`, `email`, `install` — wrong
  labels at 0.31-0.40 confidence. This is the documented **font/scale domain
  shift** (see the AVC Lexicon section: 70% on a held-out *font*, and the gate
  does not transfer), now observed end-to-end rather than in a benchmark.
- **Full-screen segmentation is the second failure.** On the whole desktop the
  reader returns one box covering the entire frame (`off`, `no`) because a real
  window's frame is not a bordered control in the sense the segmenter expects.
  It behaves correctly on a tightly cropped control region and degenerates on a
  full screen.

Consequence, stated plainly: on a real application the click path resolves
`DENY (grounder produced no candidate point)` — the honest outcome, not a bug in
the gate. The synthetic desktop demo is where the loop closes because the
generator and the reader share a vocabulary and a renderer by construction. Do
not read a working synthetic demo as evidence about unseen software.

### The synthetic desktop is not a generalisation claim

`avcs/gui/desktop.py` draws controls using words from
`avc.lexicon.vocab.LABELS`, so the corpus and the reader share a vocabulary **by
construction**. That is the honest scope: it shows the reader working on
software this generator draws. It is not evidence about unseen applications, and
the cross-generator caveats for the grounder apply in spirit here too.

`tests/test_gui_model.py` (26 tests) runs the real reader, kernel, assistant and
FastAPI app — no mocks, because the thing under test is whether the pieces agree
with each other, and a mock would assert that they do.

### The kernel floor is a per-perception choice, not a global constant

Wiring the adapter in revealed a second, subtler reason the click step looked
broken. With the candidate point finally reaching the kernel, the demo still
denied:

```
gate: DENY (grounding peak 0.691 below capability floor 0.75 (target likely absent))
```

The point was real and the read was correct; the **floor was wrong**. 0.75 was
calibrated for the goal grounder's `peak_probability`, which sits above 0.75 on a
hit and below it on a wrong query (documented above). The lexicon's confidence is
a different scale — a cosine margin against 61 prototypes — and a control it read
correctly lands at 0.47-0.70. Reusing 0.75 refuses *every* real control, which
would have looked exactly like "grounding is broken" rather than "the threshold
belongs to the other model".

`--lexicon-eyes` therefore defaults the floor to 0.45 (`--peak-floor` overrides;
the goal-grounder path keeps 0.75). `test_lexicon_confidence_scale_differs_from_grounder`
pins the fact that a correct lexicon read is below 0.75, so the day confidences
rise above it the default is revisited rather than silently kept.

With that, the demo closes the loop end to end:

```
you   : save karo
avcs  : Okay sir, click on save.   (brain=rule_fallback)
action: click target='save'
gate  : ALLOW (capability 'click' granted and grounding peak 0.691 >= 0.45)
hand  : virtual click at (57, 68)
verify: no change (delta=0.0)
```

`verify: no change` is the honest reading: the synthetic desktop does not change
when a virtual hand clicks it, so the delta is 0.0 and it says so instead of
claiming success. Real actuation (`--live`) and a real screen are what produce a
non-zero delta.

### The hand now moves like a hand (`avcs/voice/actuators.py`, `--servo`)

`XdotoolHand` teleported: `mousemove x y`. A person does not. `ServoHand` wraps
any hand and replaces the jump with the neuromuscular trajectory that was already
built and measured in `avcs/control/servoing.py` -- 5th-order minimum-jerk,
Fitts' Law duration, 8-12 Hz micro-tremor -- so the pointer travels instead of
appearing. This is a *wire*, not a new model: the trajectory code existed and
nothing on the voice/GUI path used it.

- **Opt-in.** `--servo` (GUI) or `ServoHand(hand)`. The default is still the
  teleport, because a cursor that takes ~0.3-0.9 s to arrive is a deliberate
  choice, not a silent one.
- **`--servo` without `--live` never actuates.** In the non-live branch the servo
  wraps a `VirtualCursorDriver`, not a real hand. The trajectory is planned and
  painted in memory, so the motion is observable and testable with no display,
  and no machine is touched. A test pins this; an earlier version of this change
  had `default_servo_hand()` in that branch, which would have moved the real
  cursor on a plain `--servo` run.
- **`dispatch` needs no special case.** `ServoHand` exposes `get_screen_size`
  and takes absolute pixels, exactly like `XdotoolHand`/`HandMotor`, so
  `avcs.voice.hands.dispatch` drives it through the existing path.
- **`XdotoolHand` gained `move_to_pixels` and `get_cursor_pos`.** It had no bare
  move -- `mousemove` lived only inside `click()` -- so a servo trajectory would
  have computed a path and never painted it. `move_to_pixels` is that move;
  `get_cursor_pos` is the readback the servo needs to know where it starts.
- **`available()` is cached.** It was re-probed on every call, and the servo calls
  it once per waypoint, so a single click spawned ~2x the processes it needed.
  Measured on the live server: `act_ms` 5.6 s -> 2.0 s after the cache, at 60 Hz
  tick (a 60 Hz screen cannot consume 120 Hz anyway).

Measured on this box (`--source desktop --eyes lexicon --servo --live`, real
`xdotool` on Xvfb `:99`): the trajectory is 10-80 waypoints with a speed profile
that peaks in the middle, a single click at the end, and Fitts ID 0.0 (near) to
3.0 (far). The pointer genuinely moves: `POST /api/turn {"text":"save karo"}`
takes the cursor from 640,400 to 81,105 and the record says
`performed=true, actuation=live`.

### Xvfb resets the pointer unless it is started with `-noreset`

An earlier revision of this file claimed "Xvfb is known not to honour
`XWarpPointer` for readback". **That is wrong**, and it hid a real bug. Xvfb
honours the warp fine; the pointer simply resets to the screen centre every time
the last client disconnects, so `xdotool mousemove` (which exits, disconnecting)
appears to do nothing when a *later* process reads the position. Measured:

| server | `xdotool mousemove 900 600` then a separate read |
|---|---|
| `Xvfb :99 -screen 0 1280x800x24` | stays at 640,400 (reset on disconnect) |
| `Xvfb :99 -screen 0 1280x800x24 -noreset` | moves to 900,600 |

This matters beyond benchmarks: `XdotoolHand._probe` used to check only
`getdisplaygeometry`'s exit code, so on a reset-prone server it reported the hand
available while **every click silently no-opped** -- a false success, the one
failure a fail-closed hand must never produce. The probe now moves the pointer in
one process, reads it back in a second, and requires the read-back to match
(restoring the original position either way). Two consequences:

- Run Xvfb with `-noreset` for real actuation, or the probe correctly reports the
  hand unavailable and every op is refused.
- A chained single-invocation probe (`mousemove 1 1 getmouselocation`) does
  **not** work: it reads the position before the client disconnects, so it
  reports success even on a broken server. The read must be a second connection.


### Hesitation: it looks before it clicks

`Assistant.think_pause_s` (default 0.45 s) pauses before a *mutating pointer*
action, and `record.hesitated_s` reports the real duration. This is the
difference between a macro and a person: a hand that moves in the same frame the
target is named does not look like it checked anything. It applies only to
`POINT_OPS`; a keyboard or file op gains nothing from a pause, so it does not get
one. `think_pause_s=0` disables it for a fast scripted demo.

The spoken half has an ordering rule that a test pins: the aside ("ruko, pehle
dekh leta hoon") is spoken *before* the reply that announces the action. Spoken
the other way round the user hears "click kar raha hoon" and then "ruko, dekh
leta hoon" -- a contradiction, and the macro behaviour the pause exists to avoid.
This is why the thinking line lives in `_ask_brain` (which owns speech ordering),
not in `_deliberate` (which owns the pause). On the *streaming* path there is no
separate aside: the thought is parsed from the action JSON, which only arrives
after the reply was already spoken, so an aside there would land after the
sentence it should precede. The model's own streamed words carry it instead, and
`_deliberate` still takes the pause, so the visible thinking is the same on both
paths. The conversation prompt asks for a short in-sentence aside for the same
reason.

### Self-correction: a miss is re-grounded, not accepted

A performed pointer action that changed nothing (`verified is False`) is retried
up to `max_corrections` times (default 2). The retry re-grounds from a fresh
frame, so a target that moved is followed. Three honesty properties, all pinned:

1. **A no-op retry never reaches the hand.** The new point is grounded *before*
   actuating; if it equals the previous point there is nothing to correct, and
   `detail` says so. Clicking the identical pixel again is the "stuck" behaviour
   the loop exists to avoid, and an early version did exactly that (a same-point
   miss clicked 3x; the tests caught it).
2. **A drag is compared on its source, not its destination.** `dispatch` reports
   the drag *destination*, so comparing against `record.clicked` made every drag
   look like a moved target and retry forever.
3. **A refusal is not retried.** A gate denial is a different outcome with a
   different reason; only a *performed* miss is a candidate.

`record.corrections` / `record.corrected` carry the result, and the page shows a
`CORRECTED` / `RETRIED` badge, so a correction is never mistaken for a first-try
success.

### A window Close box is a control, not a panic corner

The servo's failsafe treats top-left and bottom-right as emergency breakout
zones, and *excludes* top-right (a Close box) and bottom-left (a Start button).
Without that exclusion every click on a window's close button would trip the
failsafe and be refused -- the same reasoning the repo's own hardware failsafe
documents. The panic gesture is the top-left/bottom-right diagonal; both are
pinned by tests.

### What the assistant still cannot do (unchanged, and not hidden)

Run `python3 -m avcs.voice.demo ... --lexicon-eyes --desktop-eyes` and the header
prints the truth:

- `brain_has_key: False` — no LLM key is configured, so `brain=rule_fallback`.
  The brain is a keyword matcher, not the conversational LLM. Setting
  `AVCS_LLM_API_KEY` (and optionally `AVCS_LLM_ENDPOINT` / `AVCS_LLM_MODEL`) is
  what turns it into the real brain.
- `mouth: null`, `ears: null` — **no TTS engine and no STT backend are installed
  on this box** (`espeak-ng` absent; `vosk`/`whisper`/`faster_whisper` all
  missing, as are `sounddevice`/`pyaudio`/`mss`). So the run is text-in,
  text-out with `--silent`. The `mouth_available`/`ears_available` flags report
  whether the *code path* exists, not whether a binary is installed — do not read
  `mouth_available: True` as "it can speak here".



## The assistant GUI (`avcs/gui/`, `avcs/voice/scene_perception.py`) — a brain, not a clicker

The GUI is an **AI assistant** first: it looks at a screen, talks about it in
Hinglish, and only actuates if a caller opts in. The control-grounding path is
still there, but it is opt-in, not the product.

```
python3 -m avcs.gui.run --port 12000                 # looks + talks (no capability)
python3 -m avcs.gui.run --source screen              # the real screen
python3 -m avcs.gui.run --eyes lexicon --grant click # opt in to the click path
```

`ScenePerception` (`avcs/voice/scene_perception.py`) is the default eyes: pure
pixel perception via `forge/vision`, returning a `ScreenObservation`. It
**never names a control label** -- it stays at scene level ("a mid-tone, detailed
screen, structure reads as panel"). That is the whole point: a scene describer
is not a button detector, and a page that framed every reply as a gated click
would be the clicker the user rejected. `DemoSceneSource`
(`avcs/gui/demo_source.py`) scripts scene *archetypes*
(`desktop/terminal/media/dialog/scroll`), not controls.

Default capabilities with scene eyes are empty: `can_act: False`, `actuation:
virtual`. `--eyes lexicon` is the only thing that turns on a `click` grant,
because only the lexicon reader produces a grounding point.

### Two bugs fixed on this path

1. **`CapabilityKernel.authorize` ran the grounding checks on observing ops.**
   `is_granted` already says observing ops need no capability, but `authorize`
   then demanded a candidate point anyway, so a plain "look at the screen" step
   (`observe_target`) came back **denied** -- a perception step rendered as a
   denied click. Observing ops now return `allowed=True` before any grounding
   check. A caller can still tell a real actuation apart from a glance: the GUI
   marks each transcript entry with `mutating: bool` and the page shows a
   `LOOK` badge instead of `ALLOW`/`DENY` for a non-mutating action.
   `test_authorize_allows_an_observing_op_with_no_evidence` pins it.

2. **A `204` response carried a body.** The mouth-polling endpoint
   (`/api/speech.wav`) returned `JSONResponse({...}, status_code=204)` when there
   was nothing new. A 204 must have no body, so uvicorn raised
   `RuntimeError: Response content longer than Content-Length` on every idle
   poll -- and the browser polls four times a second, so the log filled with
   tracebacks. It now returns a bare `Response(status_code=204)`.
   `test_speech_endpoint_returns_204_with_no_body_when_idle` pins it.

### The live brain is not deterministic; tests must inject one

`ConversationBrain()` resolves a real LLM when a key is present (here the
platform-managed key resolves), so any test that asserts on an assistant's
*decision* becomes a network round-trip. `GuiApp(..., brain=...)` and the
`_assistant(brain=...)` helper exist for this: a test that pins gating or the
autonomous loop passes an offline brain (`ConversationBrain(provider=...)`
whose provider raises, so it falls back to `RuleBrain`). Verified live
end-to-end: the page renders a Hinglish reply with `brain llm` and no click, and
`/api/speech.wav` returns a real WAV (espeak-ng) for the mouth to play.

### Real mouse and keyboard actuation (`avcs/voice/actuators.py`, `hands.py`)

The hand is real, cross-platform, and **fail-closed**, and it is off unless asked
for. `avcs/voice/hands.py::dispatch` is the single point where a gated op becomes
a hand call: `click`/`double_click` -> `click(x, y)`, `type`/`type_text` ->
`type_text(text)`, `press_key` -> `press_key(key)`, `scroll` -> `scroll(delta)`,
anything else refuses. `XdotoolHand` (Linux/X11) drives the native cursor in
**absolute pixels**, matching `HandMotor` (Windows `user32`); `default_hand()`
picks the right one per platform and returns `None` when neither is usable.

The gate still owns the decision: nothing actuates until
`CapabilityKernel.authorize` returns `allowed=True` with a grounded coordinate.
`--live` is required *and* a grant (`--grant click type press_key scroll`); with
neither, `Assistant` holds a `VirtualHand` that records the coordinate and moves
no cursor.

Measured on this box (Xvfb `:99`, 1280x800, real Tk window at +80+80), a full
`POST /api/turn` with `--live --grant click --eyes lexicon`:

```
cursor before : 500,300
action        : click | gate: True (capability 'click' granted and grounding peak 0.691 >= 0.45)
actuation     : live | performed: True | clicked: [81, 105]
cursor after  : 81,105          # the real pointer moved
window recv   : CLICK 1,25      # (81,105) - (80,80) -- the real window got it
```

Typing and keys were verified the same way: `dispatch(hand, 'type_text', ...)`
delivered all 15 characters plus `Return` to the focused window, in order.

**`--live` with no hand must not read as "live".** `GuiApp.capabilities()` now
derives `actuation` from whether a hand is actually attached:
`live` (hand present) / `live-unavailable` (`--live` requested, none usable) /
`virtual` (default). It also reports `hand` by name. `run.py` prints a warning
for the middle case. A page that showed "live" while every op was refused would
be the same dishonesty as a checkpoint claiming accuracy it does not have.
`test_live_without_a_hand_reports_unavailable_not_live` and
`test_live_with_a_hand_reports_live_and_names_it` pin both halves.

Note on Xvfb: start it with `-noreset`. The pointer position resets to the screen
centre whenever the last client disconnects, so a short-lived `xdotool mousemove`
followed by a separate `xdotool getmouselocation` reads the centre, not the move.
See "Xvfb resets the pointer unless it is started with `-noreset`" above for the
measurement and for why the availability probe now reads the pointer back.

### The panic-corner rule belongs on the destination, not the start

Two bugs in the neuromuscular servo made a real click impossible in ordinary use:

- **A corner *start* locked the motor out permanently.** `servo_to` checked the
  *start* position against the panic corners (top-left / bottom-right) and
  `_move_cursor` checked *every waypoint*. So if the user had parked the pointer
  in a corner, the first click tripped the failsafe before a single move; because
  the next attempt read the same start position, `reset_failsafe()` did not help
  either -- a permanent lockout with no click ever performed. Measured before the
  fix: cursor at 1279,799 -> `click` returns False, `tripped=True`, and a second
  click after `reset()` also False.
- **The override detector never ran.** `ServoHand.override_threshold` was stored
  and `CornerBreakoutFailSafe.check_override` existed, but nothing in `servo_to`
  called it, so a user grabbing the mouse mid-trajectory was never noticed and the
  servo kept dragging the pointer out from under them -- the opposite of the
  "user always wins the pointer" behaviour the docstring promised.

The corner rule now applies to the **destination** only: a target that resolves
into a panic corner is refused (the user keeps that corner), an interior target is
clicked from any start, and an intermediate waypoint crossing a corner is fine.
`servo_to` also calls `check_override` per waypoint against a driver that can
report its position, so a real takeover latches and the click is refused. Measured
after the fix: corner start 1279,799 -> click lands at 400,300 (`performed=True`);
corner *target* still refused; a driver reporting a fixed far-off position gives
`overrides>=1`, no click, `tripped=True`.

## Persistent memory, trajectory log, and the wider op set (`avcs/voice/`)

The assistant now carries durable memory, logs every turn, and can do more than
click. All of it is opt-in and all of it is fail-closed.

### Memory (`avcs/voice/memory.py`)

A JSON store of **facts** (user-scoped, keyed, LRU-evicted at `max_facts`) and
**episodes** (what happened, in order). `Assistant` attaches it to the brain
(`brain.memory`) so the model reads recalled facts in its observation and can
write new ones back (`memory` list in its JSON). A fact survives a server
restart; an episode is a log entry, not a belief.

- `--memory PATH` on `avcs.voice.demo`, or `--memory` on `avcs.gui.run`.
  Without it a run is amnesiac, which is the honest default: a file on disk is a
  side effect.
- Measured across a restart: "mera naam Rahul hai yaad rakho" stored
  `User ka naam Rahul hai`; a fresh process answered "Aapka naam Rahul hai" and
  the header reported `memory: True`. The GUI badge reports `on`/`off` from the
  real capability, never a cosmetic "on".

### Trajectory (`avcs/voice/trajectory.py`)

Append-only JSONL, one row per turn, with the three booleans kept separate
(`gate_allowed`, `performed`, `verified`) so a reviewer can tell "allowed but the
hand failed" from "refused". It is **write-only on purpose**: no `train`/
`learn`/`finetune` method exists, and a test asserts that. Rejection-sampling
self-training was tried in this repo and made the model worse; a log that quietly
feeds the model would repeat that mistake.

### The wider op set

`right_click`, `drag`, `launch_app`, `read_file` and `list_dir` are now real
across the contract (`ActionOp`), the kernel, the hands (`VirtualHand`,
`XdotoolHand`, `HandMotor`) and the dispatcher.

- **`right_click` must never degrade into a primary click.** The dispatcher's
  branch was falling through to `click`, so a right-click that could not be
  performed silently became a left click -- a destructive wrong action. It now
  returns a `HandAction` that reports "did not perform", and a test pins it.
- **`launch_app` / `read_file` / `list_dir` are allow-listed.** The kernel takes
  `allowed_apps` and `path_roots`; with neither, the ops are off. `read_file`
  refuses a path outside the roots *and* refuses traversal even when the string
  has a valid prefix (`/root_evil` vs `/root`), resolved via `os.path.realpath`.
  Measured live: `read_file /tmp/avcs_gui_test/note.txt` ->
  `gate_reason: "capability 'read_file' granted (no grounded point needed)"`,
  `performed: True`, `detail: "hello avcs\n"` -- the real file contents.
- **`drag` needs both ends grounded.** A half-authorized drag (source grounded,
  no destination) is refused, and a low-confidence destination is refused.

### Two prompt/parser bugs the wider ops exposed

1. **The LLM never emitted an operand, because the schema did not have one.**
   `CONVERSATION_SYSTEM_PROMPT`'s JSON example listed only
   `target_description` / `text_input` / `key_name`, so `read_file` came back
   with no path and the gate refused it for the honest reason "no path given".
   The parser already accepted `parameters` (and `path`/`app`/`destination` at
   the top level), but a schema the model is never shown is a schema it cannot
   fill. The example now includes `parameters` and states which op requires
   which operand, so an op emitted without one is a model error, not a
   prompt gap.
2. **`RuleBrain` could not reach the new ops at all**, so the offline path was
   untestable for them. It now extracts a path/app, asks for the operand when it
   is missing ("padho" -> "Ji sir, kaun sa path?"), and handles `right_click`
   and `drag`.
   - Operand ops are matched **before** target extraction: `read_file` needs a
     path, not an on-screen target, so the target-not-found branch would wrongly
     refuse an actionable request.
   - The action's own word is stripped from the target: "save button par right
     click karo" names the control **`save button`**, but the object-verb split
     kept "right" and the gate looked for a control called "save button right",
     which is on no screen. Only `right`/`secondary` are stripped -- the verb
     itself must stay, or the split loses the token that marks where the target
     ends. `test_rule_brain_does_not_put_the_action_word_in_the_target` pins it.
     This is the same class as the `ab close` filler bug above.

### The GUI hand in virtual mode (a claim that was false, now true)

AGENTS.md previously stated "the GUI attaches a `VirtualHand` by default". It
did **not**: the non-live branch left `self.hand = None`, so `can_click` was
False and *every* granted click was refused with "no hand attached" -- the same
symptom as a broken grounder, for an unrelated reason. The GUI now attaches
`VirtualHand(width, height)` in virtual mode (size from `_frame_size()`,
defaulting to 900x520), and `capabilities()` reports `actuation: virtual` with
`hand: virtual-hand`. Virtual mode maps the normalized point into real pixels via
the frame size, so the recorded coordinate is comparable to a live one.
`test_gui_virtual_mode_attaches_a_hand_so_a_grant_is_reachable` pins it.

Measured live through the GUI on the synthetic desktop
(`--source demo --eyes lexicon --grant click right_click drag read_file`):

| utterance | op | gate | performed |
|---|---|---|---|
| mujhe save karna hai | click | ALLOW | True |
| save button par right click karo | right_click | ALLOW | True |
| save button ko close button par drag karo | drag | ALLOW | True |
| note.txt file padho (no path) | read_file | DENY (no path given) | False |
| read_file /tmp/avcs_gui_test/note.txt | read_file | ALLOW | True |
| list_dir /tmp/avcs_gui_test | list_dir | ALLOW (LLM did not emit it) | False |

The last row is honest and worth keeping: the model *described* listing the
directory but emitted no action, so nothing ran. A narration without an emitted
action is a false statement about the world, and the page shows no `ALLOW` for
it. The system prompt already forbids it; the trajectory log is where that
failure becomes measurable.

Full suite after these changes: **271 passed, 2 skipped**.

## Proactive awareness: the assistant speaks first (`avcs/gui/awareness.py`)

Everything above describes an assistant that *answers*. That is a command line
with a voice: nothing is said unless the user types or talks first. The gap
between "answers well" and "an assistant in the room" is that a person notices
things and mentions them, unprompted.

`AwarenessLoop` is that half. It is a daemon thread that watches the cheap pixel
reflex the page already runs, and lets the brain speak when something is worth
interrupting for. It is **opt-in** (`--proactive`) because an assistant that
talks unprompted must be asked for.

### When it is allowed to speak

The reflex already computes the signals, so no model runs per frame. A *reason*
is raised only when a signal crosses a threshold, in priority order:

| reason | signal | default |
|---|---|---|
| `overlay` | something small and high-contrast appeared over the scene | `change_fraction >= 0.004` |
| `scene_change` | the whole screen changed at once | `change_fraction >= 0.30` |
| `appear` | something new and prominent appeared | `novelty >= 0.45 and change >= 0.55` |
| `stuck` | local motion repeating in one spot, no scene change, for a while | `peak_motion >= 0.12` for `45 s` |

Everything else is ignored on purpose. **Narrating a moving mouse is noise, and a
noisy assistant is worse than a silent one.** Two more gates follow: a
**cooldown** (never speak twice within 20 s) and a **dedup window** (the same
reason is not repeated for 120 s, even past the cooldown).

Measured live on the scripted demo, one minute of watching with the real LLM
brain:

```
remark_count: 2
  scene_change  "Lagta hai aapne poora naya screen ya app khol liya hai."
  overlay       "Ek chhota sa dialog ya overlay screen par aa gaya hai, ek nazar daal lena."
skipped: {cooldown: 26, repeat:scene_change: 20, silent:scene_change: 1}
```

Two remarks in ~60 s of change, 46 ticks suppressed by cooldown/dedup, and one
tick where **the model chose silence** (`silent:scene_change`). The restraint is
the feature; the skipped counters are how it is measured rather than claimed.

### What it cannot do

**It has no hand.** There is no path from this module to an actuation, and a
test asserts the source imports no `VirtualHand`/`HandMotor`/`XdotoolHand` and
never mentions the kernel. A watcher that could also click is the clicker this
product explicitly is not. The brain is even told `can_click: False` in the
proactive observation, so it cannot announce an action it will never take.

It also stays quiet while a user turn is in flight (`GuiApp._turn_in_flight`),
so a remark can never talk over the conversation it exists to support.

### A separate prompt, because the role is different

`PROACTIVE_SYSTEM_PROMPT` is not the conversation prompt. Here the model is a
colleague looking over your shoulder and **silence is the correct answer most of
the time**. The prompt lists when to stay silent (a mouse move, a mid-task user,
a repeat, uncertainty) and asks for an empty `speech` when in doubt. Offline, the
rule fallback speaks only for an overlay or a whole-screen change -- the two
things a keyword matcher can be sure of.

`ConversationBrain.comment()` returns ``None`` for "nothing worth saying", so a
caller cannot accidentally speak an empty string. It never raises: an
unreachable model degrades to the conservative rule remark.

### Hands-free listening: no button

The mic was push-to-talk (hold to speak). A person does not press a button to be
heard, so the page now records short windows on a loop and submits any that
contain speech, dropping silent windows rather than printing "no speech" every
few seconds. Two guards matter:

- It pauses while the assistant's own mouth is playing (`mouthPlaying`), so the
  TTS output is not transcribed back as the user's words -- the feedback loop
  that would make the assistant answer itself forever.
- It is off until the first click **anywhere** on the page. That click is the
  browser's mandatory microphone-permission gesture, not a "start listening"
  button; after it, nothing needs pressing.

The header reports `listening: off/on` from the real state, and a blocked mic
shows `mic blocked` rather than a silent failure.

### Honest limits

The signals are coarse pixel statistics, not understanding. `overlay` is a
bright small region, not "an error dialog about your file" -- the brain turns the
observation into words, and with the offline rule brain it will be a generic
sentence. Thresholds are defaults tuned on the synthetic demo on this box, not
calibrated across real applications; `AwarenessConfig` exists so a caller
retunes rather than trusting the constants. The whole thing was verified on a
headless container: remarks were produced and rendered, but the browser-side
microphone path could not be exercised here (no sound card), so hands-free
listening is code-verified, not end-to-end audio-verified.

Tests: `tests/test_proactive_awareness.py` (20 tests) drives the real loop, the
real brain and the real FastAPI app. Full suite: **327 passed, 2 skipped** (with Xvfb `:99 -noreset` running; the skip count rises to 5 when no display is present)
(after the human-motion, hesitation and self-correction work).
