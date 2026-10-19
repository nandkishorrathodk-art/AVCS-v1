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

## Fixed bugs worth knowing about

- **`CapabilitySet.check` rejected `list`/`stat`/`scan` as "unknown
  operations"** regardless of grants, so three operations that
  `default_policy` grants and `_dispatch` implements were unreachable.
  `GRANTABLE_OPS` now derives the accepted set from the enum, and a test
  asserts `set(Op) == MUTATING_OPS | OBSERVING_OPS` so a new op cannot be
  added in one place and forgotten in another.

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
- `runs/weights/human_eye_weights.pt` is produced by the real trainer
  (`avcs/training/train_affordance.py`); when present `HumanEye` loads it
  (`weights_loaded=True`). It is gitignored, so a fresh checkout lacks it and
  the tests that need it fail correctly. Never fabricate a placeholder
  checkpoint to make them pass. Its acceptance numbers are same-distribution
  synthetic figures, not a generalisation claim.
- `runs/weights/goal_grounder_best.pt` **is** trained (1.66M params, schema 2,
  `is_trained=True`). On `datasets/ui_grounding` `test_held_out` it reproduces
  94.94% top-1 / 3.72 px median via
  `python3 -m benchmarks.run_ui_grounding_benchmark`. But that split shares a
  generator with train. On `datasets/cleanroom_system1_v1` (a different
  generator) the same checkpoint drops to 28.1% top-1 and its calibrated
  presence threshold is at **chance** (34.4% of true targets rejected, 65.6% of
  true absences accepted). Always quote the split *and* the generator.
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
  present, the full suite is **2 failed / 1194 passed / 16 skipped**. The two
  failures are pre-existing and environment-dependent: the cleanroom-rights
  test (the shipped attestation binds manifest hash `abde4338…` but the current
  manifest hashes to `06c6050d…`, so the fail-closed gate reports a binding
  mismatch first) and the R3 latency test (passes in isolation, fails under
  full-suite CPU contention). Always quote the installed deps with the number.
  The Tkinter GUI tests need a display; run them under `Xvfb` and install the
  `tk8.6` system package (`libtk8.6.so`).
- Tkinter GUI modules (`avcs/gui.py`, `avcs/runtime/live.py`, the m7 harness)
  need the system `libtk8.6.so`. They now raise an actionable ImportError
  instead of an opaque loader error.

## AVCS `forge/vision/cortex/` System 1 — completed & measured

A second System 1 implementation, trained to completion in this repo. Read
`docs/REALITY_AUDIT.md` section 11.

- `ArtificialVisualCortex` is **1,139,032 params**. Checkpoint:
  `runs/weights/avcs_cortex_weights.pt` (loads with 0 missing / 0 unexpected).
- Four pipeline defects were fixed to make it train at all: `frame_next` was a
  pixel-identical copy (world model learned identity); the `CharNGramEncoder`
  query path was dead (grounding trained on noise); `ScreenEye.look_at` used a
  *different, untrained* encoder than training (train/inference mismatch); the
  trainer saved to a path discovery does not look at.
- Real text now enters via `forge/vision/cortex/data/text.py::ngram_query_embedding`.
  Both the trainer and `ScreenEye.look_at` **must** use this same encoder —
  using `ByteQueryEncoder` at inference silently ungrounds the model.
- Held-out (n=40, procedural, same generator): grounding median error
  **0.0188 norm (~12 px @640)**. Same-distribution only; not a generalisation
  result.
- It is an **active-vision** model: fovea fixation is state carried between
  calls, so consecutive `look_at` calls on one frame legitimately differ. Reset
  fixation for bit-identical output.
- It still has **no text decoder** and does not speak. Speech is System 2.

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
