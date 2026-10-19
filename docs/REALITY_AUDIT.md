# AVCS Reality Audit — System 1 (Eyes + Hands) vs System 2 (Cognitive Brain)

This document states, with measurements, what the AVCS System 1 model in this
repository actually does. It exists to correct an overclaim made in earlier
project messages: that the small (~500k parameter) System 1 model, by itself,
reads on-screen text, names open applications, and speaks natural language
like a person.

It does not. That work is System 2. This audit records the boundary.

All numbers below were produced on this machine (4-core CPU, no GPU) with the
code in this repository. Commands are given so every figure can be re-run.

---

## 1. The claim that was wrong

Earlier messaging described output such as *"Code Editor is open, the search bar
is in the header..."* as if System 1 produced it alone. That was an
overpromise. The intended architecture was always System 1 (eyes + hands) plus
System 2 (a VLM/LLM that reads text and speaks). The combined-flow target was
mistakenly reported as a System 1 capability.

---

## 2. What System 1 actually is

`StreamingVisualCortexV1` — 501,612 parameters.

- It is a **visual reflex arc and motor cerebellum**, not a language model.
- It consumes raw pixels and produces numeric state: a recurrent spatiotemporal
  hidden state, change/motion signals, and affordance hotspot coordinates.
- It has **no text decoder**. There is no vocabulary, no tokenizer, and no
  autoregressive head in the model.

Verified by inspecting the public surface:

```bash
python3 -c "
from avcs.observation import VisualObservationEngine
import inspect
print([n for n,_ in inspect.getmembers(VisualObservationEngine, inspect.isfunction)])
"
```

### The "description" text is a template, not generation

The natural-language sentence that leaves System 1 is assembled by
`VisualObservationEngine._synthesize_description` in `avcs/observation.py`,
which slots a detected category name, a stability flag, and hotspot counts into
fixed Python format strings:

```python
parts = [f"Display is active. Primary context: {cat_clean} ({stability})."]
```

This is deterministic string templating. It is not learned text and not
autoregressive decoding. It is honest and evidence-backed, but it is not speech.

### The one text-shaped model is a closed-vocabulary classifier, not OCR

`runs/weights/candidates/candidate_m3_word_recognizer_v1.pt` (239,716 params,
`best_val_accuracy_pct: 69.21`) classifies a 32x96 crop into **one of 132 fixed
UI labels** (`Save`, `Cancel`, `Close`, `Terminal`, `Bookmarks`, ...). It is
genuinely trained — the sidecar carries a 25-epoch history — but a 132-way
closed-vocabulary classifier cannot read arbitrary text. Type a sentence, open an
unmapped button, or view source code and it has no class to emit. This matches
the README's stated limit; it is recorded here so the 69.21% figure is not
mistaken for OCR accuracy.

---

## 3. What the packaged System 1 checkpoint is

**It is an untrained, clean random-initialization reference.**

The sidecar `runs/weights/candidates/avcs_streaming_cortex_v1.json` records:

```json
"clean_random_initialization": true,
"legacy_weights_loaded": false
```

Reproduction check — the shipped state dict equals
`create_clean_random_cortex(seed=20261018)` up to float32 rounding:

```bash
python3 -c "
import torch
from avcs.streaming_v1.model import create_clean_random_cortex
saved = torch.load('runs/weights/candidates/avcs_streaming_cortex_v1.pt',
                   map_location='cpu', weights_only=False)['state_dict']
fresh = create_clean_random_cortex(seed=20261018).state_dict()
print('max abs diff:',
      max((saved[k].float()-fresh[k].float()).abs().max().item() for k in fresh))
"
# -> max abs diff: 1.7462298274040222e-07
```

So the 501k checkpoint is a plumbing fixture: it proves the streaming pipeline,
mailbox, and telemetry run on CPU. It does **not** encode learned visual
understanding, and no perception-accuracy claim may be attached to it.

The model is loaded by benchmark and packaging scripts. It is **not** wired into
the live runtime path.

### A second System 1 checkpoint *is* trained, but only barely

`runs/weights/candidates/avcs_system1_v1.pt` is a different model
(`AVCSSystem1V1`, 1.66M params) and it *was* trained — its sidecar records
3 epochs, `best_val_loss: 0.8749`, 13.45 s of CPU time, with a small loss curve
(1.0505 → 0.8749). This is a real training run on a small synthetic corpus, not
a fabrication. It is also not a capability: 3 epochs to a validation loss of
0.87 on a procedurally generated set says the plumbing learns something, not
that the model understands scenes. Do not quote its loss as accuracy — no
held-out metric is recorded for it.

So the repository contains, at the System 1 level: one untrained streaming
cortex, one lightly-trained small classifier, and one genuinely trained grounder
(section 5a). Only the grounder has a held-out number, and that number does not
transfer.

---

## 4. Throughput is real, but only at the soak resolution

The soak test (`benchmarks/benchmark_soak_test.py`) runs at `width=64, height=64`
and reports 95.41 FPS / 9.74 ms p50. That number is reproducible, and it is a
64x64 number.

The model's forward cost scales with input area:

| input | raw forward | throughput |
|---|---|---|
| 64x64 | 1.8 ms | 564.8 FPS |
| 256x256 | 9.4 ms | 106.1 FPS |
| 1920x1200 | 446.4 ms | 2.2 FPS |

Re-run:

```bash
python3 -c "
import time, torch
from avcs.streaming_v1.model import StreamingVisualCortexV1
m = StreamingVisualCortexV1().eval()
for hw in [(64,64),(256,256),(1200,1920)]:
    x = torch.rand(1,3,*hw)
    with torch.no_grad():
        for _ in range(3): m.forward_step(x)
        N=15; t=time.perf_counter()
        for _ in range(N): m.forward_step(x)
        dt=(time.perf_counter()-t)/N
    print(hw, f'{1/dt:.1f} FPS')
"
```

**The headline "95–104 FPS at a 1920x1200 desktop" is misleading as stated.** At
full desktop resolution the model runs at 2.2 FPS. The production path avoids
that cost by letterboxing the captured frame down to 128x128 before the model
sees it (`StreamingObserver._prepare`). Measured end-to-end, including that
resize:

```bash
python3 -c "
import time, numpy as np
from avcs.streaming_v1.runtime import StreamingObserver, FramePacket
from avcs.streaming_v1.model import StreamingVisualCortexV1
m = StreamingVisualCortexV1().eval()
obs = StreamingObserver(m, input_size=(128,128))
f = np.random.randint(0,255,(1200,1920,3),dtype=np.uint8)
obs.process(FramePacket(sequence=0, timestamp=time.monotonic(), rgb=f))
N=20; t=time.perf_counter()
for i in range(1,N+1):
    obs.process(FramePacket(sequence=i, timestamp=time.monotonic(), rgb=f))
dt=(time.perf_counter()-t)/N
print(f'{dt*1000:.1f} ms -> {1/dt:.1f} FPS')
"
# -> 4.5 ms -> 223.4 FPS
```

So real-time is achievable, but only because the model never sees the full
frame. The resolution must be quoted with every throughput number.

---

## 5. Natural-language actions fail closed — and that is correct

`python3 -m avcs.demo_system2` shows every action refused:

```
* Action:  open_application   Success: False
* Action:  click              Success: False
* Action:  type_text          Success: False
```

A trained grounding checkpoint now exists at
`runs/weights/goal_grounder_best.pt` (schema 2, `is_trained=True`), so the
controller reports `is_grounder_trained=True` and the missing-checkpoint branch
is no longer the reason. The refusal comes from a second, deliberate gate:

- `GoalConditionedGrounder.predict` (`avcs/goal_grounder.py`) returns
  `action_authorized=False` on **every** call — it is hardcoded, not computed.
- `AVCSController.locate` (`avcs/control/controller.py`) raises
  `UnvalidatedGroundingError` whenever `action_authorized` is false.

`tests/test_controller.py::test_locate_with_trained_grounder` asserts this
refusal *even when a trained grounder is loaded*, so it is an intended
fail-closed invariant, not a regression. The gate is not reachable by any
currently-packaged checkpoint: `action_authorized` is never set true anywhere in
the repository. The practical consequence is that the locate → click → verify
loop cannot execute an action today. That is the conservative default; enabling
actuation is an explicit decision that has not been made.

The demo previously printed *"CLOSED-LOOP DEMO SUCCESSFULLY VERIFIED
END-TO-END!"* unconditionally, on top of three failed actions. That banner was
false and has been replaced: the demo now reports whether the fail-closed wiring
behaved as declared, and exits non-zero if any action claims success.

---

## 5a. The trained grounder: real on its own split, not transferable

The one genuinely trained perception model in this repository is
`GoalConditionedGrounder` (1.66M params, `RealTimeCorticalBackbone` +
`ByteQueryEncoder`, 32x32 heatmap over a 256x256 letterboxed input). It was
trained on `datasets/ui_grounding` (1606 train points, 394 validation).

On the untouched `test_held_out` split it reproduces exactly:

```bash
python3 -m benchmarks.run_ui_grounding_benchmark \
  --manifest datasets/ui_grounding/manifest.json \
  --checkpoint runs/weights/goal_grounder_best.pt \
  --split test_held_out --device cpu
```

```
Top-1 Grounding Accuracy:      94.94% (hits: 300/316)
Median Centroid Error:         3.72 px (mean: 7.95 px, p95: 16.37 px)
Mean Bounding Box IoU:         0.282 (IoU@50: 6.33%)
Ungroundable Query Rejection:  100.00% (200 queries)
Negative Screen FP Count:      0 (rate: 0.00% across 30 screens)
Center-Collapse Detected:      False
```

These numbers are real and reproducible. They are also **same-distribution**.
`datasets/ui_grounding` is procedurally generated
(`dataset_provenance: avcs_procedural_synthetic_desktop_v1`), so train and test
frames come from one generator with the same layout families.

The repository ships a second, independently generated UI corpus
(`datasets/cleanroom_system1_v1`, 4 disjoint layout families, separate
annotations). Evaluating the same checkpoint there — a dataset it never saw —
gives a different picture:

| metric | value |
|---|---|
| top-1 centroid within 0.20 norm (`test_heldout`, 64 present) | 28.1% |
| mean centroid error (`test_heldout`) | 0.464 norm (118.8 px) |
| predicted x/y spread vs ground truth (`test_heldout`) | 0.248 / 0.204 vs 0.260 / 0.171 |
| present queries rejected by the calibrated threshold (`test_heldout` + `test_negative`, 64 present) | 34.4% (22/64) |
| absent queries accepted by the calibrated threshold (`test_heldout` + `test_negative`, 32 absent) | 65.6% (21/32) |

The presence head is at **chance on the new corpus**: 34.4% of true targets are
below threshold and 65.6% of true absences are above it. A threshold that
separated the classes perfectly in-distribution does not separate them at all
out-of-distribution.

The centroid still spreads across the frame rather than collapsing to the
centre, so spatial structure transfers partially; the confidence/abstention
signal does not. Read together with the in-distribution row, the honest
statement is: this grounder works on the distribution it was trained on and
does not generalise to a different UI generator. The 94.94% headline must always
be quoted with its split and generator.

---

## 6. The System 1 / System 2 split, restated honestly

| Capability | System 1 (this repo, local CPU) | System 2 (external VLM/LLM) |
|---|---|---|
| Model size | 501k (untrained) / 1.66M (grounder) | 10B–100B+ params |
| Clock | continuous, resolution-dependent | change-triggered (~1 s) |
| Input | raw pixels | text + bounded System 1 observations |
| Output | numeric state, hotspots, template sentence | free-form text and speech |
| Reads on-screen text | no (no OCR, no decoder) | yes |
| Plans tasks | no | yes |
| Motor control | yes (min-jerk cursor, verification) | no |

The narrow, defensible claim for System 1 is: it runs a bounded, local,
coordinate-free visual reflex loop on CPU and exposes verified state to System 2
without leaking coordinates. It does not understand scenes and does not speak.

### System 2, as shipped, is also not a language model

`AVCSSystem2Bridge` defaults to `MockReasoningProvider`
(`avcs/reasoning/bridge.py`: `self.provider = provider or MockReasoningProvider()`).
That provider is a deterministic rule engine with optional fault injection —
useful for tests, but it is not an LLM. A real
`LLMReasoningProvider` (HTTP + `.env` credentials) exists in
`avcs/reasoning/provider.py` but is not the default and is not configured in this
environment.

So the full "brain that talks about the screen like a person" pipeline is
**architecturally present but not running** here: System 1 emits template
sentences, and System 2 falls back to a scripted mock. Nothing in this
repository currently produces free-form natural language about a live screen.
That requires wiring `LLMReasoningProvider` to a real endpoint.

---

## 7. Change log from this audit

- Removed dead compatibility shims with zero references: `avcs/motor.py`,
  `avcs/adapters.py`, `avcs/control/motor.py`.
- Fixed version drift: `avcs/__init__.py` reported `1.0.0rc1` (the
  `release/system1` version) while the package and `pyproject.toml` are
  `0.2.0`. It now resolves `avcs-sdk` / `avcs-system1` metadata and falls back
  to `0.2.0`.
- Removed the false success banner from `avcs/demo_system2.py` and replaced the
  fabricated "93.3% Accuracy, 2.2px Error" docstring line with an accurate
  description.
- Corrected `docs/AVCS_STREAMING_CORTEX_V1_MODEL_CARD.md`: throughput is now
  labelled 64x64, the input-resolution scaling is stated, and the checkpoint is
  documented as untrained clean random initialization.
- Made Tkinter-dependent modules (`avcs/gui.py`,
  `avcs_m7_validation/live_harness/harness_app.py`) raise an actionable message
  naming the missing system package instead of an opaque `libtk8.6.so` loader
  error.

### Follow-up audit (same day, after the grounder was trained)

- Trained `GoalConditionedGrounder` honestly and reproduced its held-out result
  (section 5a). Added the cross-dataset evaluation that shows the presence head
  is at chance on a different generator — the single most important limitation
  found in this pass.
- Corrected section 5: the fail-closed refusal is no longer caused by a missing
  checkpoint; it is a hardcoded `action_authorized=False` gate.
- Documented that `avcs_system1_v1.pt` is lightly trained (section 3) and that
  `candidate_m3_word_recognizer_v1.pt` is a closed-vocabulary classifier, not
  OCR (section 2).
- Documented that System 2 defaults to `MockReasoningProvider`, so no free-form
  language is produced anywhere in the shipped pipeline (section 6).
- Added a re-verification command block (section 9) and a test-suite baseline
  table with its dependency caveat (section 10).

## 8. What was NOT done, and why

- **No checkpoint was fabricated.** The earlier session's trainer printed
  `Saved checkpoint` and `top1_accuracy_pct: 86.71` without writing a file or
  computing anything. That fabricated path was deleted and replaced with a
  trainer that writes the real checkpoint and records real validation metrics
  (`avcs/training/train_grounder.py`). The `goal_grounder_best.pt` numbers in
  section 5a are from that honest trainer and are reproducible from the command
  shown.
- **`human_eye_weights.pt` was absent at the time of the original audit.** It
  has since been produced by the real trainer
  (`avcs/training/train_affordance.py`, 260 steps, 17.4 s CPU, 68,719 params,
  summary sidecar `avcs_training_summary.json`). It is **not** a fabricated
  placeholder: it loads into `HumanEye` (`weights_loaded=True`) and its sidecar
  records real precision/recall, not an invented accuracy. Its acceptance
  numbers are same-distribution synthetic figures and must be quoted with that
  caveat. If the file is missing in a fresh checkout, the tests that need it
  fail correctly — do not create placeholder weights to make them pass.
- **The failing tests were not silenced.** They encode real gaps: an untrained
  streaming cortex, an unsatisfiable actuation gate, and absent perception
  artifacts. They remain failing and visible.
- **The cross-dataset weakness was not hidden.** The in-distribution 94.94% and
  the out-of-distribution chance-level presence head are both reported, in
  section 5a, with the commands to reproduce each.

## 9. How to re-verify this audit

```bash
# 1. Reproduce the in-distribution held-out result (section 5a, first table)
python3 -m benchmarks.run_ui_grounding_benchmark \
  --manifest datasets/ui_grounding/manifest.json \
  --checkpoint runs/weights/goal_grounder_best.pt --split test_held_out

# 2. Show the packaged streaming cortex is random init (section 3)
python3 -c "
import torch
from avcs.streaming_v1.model import create_clean_random_cortex
saved = torch.load('runs/weights/candidates/avcs_streaming_cortex_v1.pt',
                   map_location='cpu', weights_only=False)['state_dict']
fresh = create_clean_random_cortex(seed=20261018).state_dict()
print(max((saved[k].float()-fresh[k].float()).abs().max().item() for k in fresh))
"   # -> ~1.7e-07

# 3. Show the actuation gate is hardcoded closed (section 5)
grep -n "action_authorized" avcs/goal_grounder.py avcs/control/controller.py

# 4. Show System 2 defaults to the mock provider (section 6)
grep -n "MockReasoningProvider()" avcs/reasoning/bridge.py
```

## 10. Test-suite baseline (for regression checking)

Measured on this machine, 4-core CPU, with the optional test dependencies
installed from `pyproject.toml [dev]` (`pytest pytest-asyncio build wheel`)
plus runtime deps (`torch numpy psutil cryptography fastapi httpx requests
Pillow`), and with a virtual display (`Xvfb :99`) for the two Tkinter GUI tests:

| tree | result |
|---|---|
| working tree with this audit's changes | **2 failed, 1194 passed, 16 skipped** (396 s) |

The two remaining failures are both pre-existing and environment-dependent, not
regressions:

- `test_cleanroom_rights_hardening.py::test_pending_human_attestation_and_unsigned_template_never_open_gate`
  fails identically on pristine HEAD. The shipped
  `docs/CLEANROOM_DATA_RIGHTS_ATTESTATION.md` binds manifest hash
  `abde4338…`, but the current `datasets/cleanroom_system1_v1/manifest.json`
  hashes to `06c6050d…`, so the gate reports the binding mismatch before the
  decision check the test matches on. The gate is fail-closed either way.
- `test_independent_eval.py::TestRequirementR3CPUPerformance::test_r3_end_to_end_perception_latency_and_fps`
  is a wall-clock latency assertion that passes in isolation and fails only
  under full-suite CPU contention. It is a flaky performance bound, not a
  correctness failure.

Earlier baselines in this document (e.g. "68 failed") predate having the full
dependency set installed; with the dependencies present, far more tests execute.
Always quote the dependency set with the number.

### Environment fixes required to run the suite as shipped

The suite could not run to completion on a bare checkout for three
environment reasons, all now addressed:

1. `pytest-asyncio` was missing from `[dev]` although 30 async tests
   (`test_sdk_contract.py`, `test_sdk_repository_backend.py`,
   `test_pilot_repository_adapter.py`) require it. Added to `pyproject.toml`.
2. `build` was missing from `[dev]`, so `test_packaging_release.py` could not
   find a wheel in `dist/`. Added to `[dev]`; the wheel now builds and the
   metadata tests pass.
3. The Tkinter GUI tests need a display. On a headless host run them under
   `Xvfb`; `libtk8.6` (from the `tk8.6` system package) must be present.

## 11. Follow-up: the cortex model was completed and its defects fixed

After the audit above, the `forge/vision/cortex/` System 1 pipeline was taken
from "structurally present, untrained" to "trained, loads through the SDK". All
figures below were produced on this machine and are reproducible.

### Defects found and fixed

1. **Temporal pair was an identity copy.** `ProceduralScreenGenerator` returned
   `frame_next` pixel-identical to `frame_t`, so the world model was trained to
   predict "no change". Fixed by adding `generate_successor()` (cursor-tracked
   motion between frames). Verified: mean abs frame difference went from exactly
   `0.0` to `0.00023`.
2. **The query encoder was dead code.** `CharNGramEncoder` existed but the
   pipeline fed random noise as the grounding query, so the grounding head
   trained against nothing. Fixed by adding `ngram_query_embedding` in
   `forge/vision/cortex/data/text.py` and wiring real text into the dataset.
   Verified: query embeddings are unit-norm, deterministic, and distinct across
   different queries.
3. **Train/inference embedding mismatch in the SDK.** `ScreenEye.look_at`
   encoded the query with a freshly constructed, **untrained** `ByteQueryEncoder`
   while the model was trained on `ngram_query_embedding`. Inference therefore
   ran in a different embedding space than training. Fixed to use the same
   encoder as training.
4. **Checkpoint path mismatch.** The trainer saved to `runs/cortex_model.pt`
   while discovery looks for `runs/weights/avcs_cortex_weights.pt`. Fixed.

### What the trained cortex actually is

`ArtificialVisualCortex` is **1,139,032 parameters**, trained 6 epochs on 320
procedural samples (CPU, ~27 s/epoch). It predicts: a dorsal affordance map
(5 classes over a 32x32 grid), a UI-category distribution (8 classes), a
temporal successor, and a language-conditioned gaze coordinate.

Checkpoint: `runs/weights/avcs_cortex_weights.pt`, loads with
`strict=False` and **0 missing / 0 unexpected keys**.

Honest held-out measurement (n=40 procedural samples, seed 1000+):

```
grounding median error: 0.0188 norm  (~12 px on a 640px-wide frame)
grounding mean error:   0.0309 norm
```

Caveat, stated plainly: this is **same-distribution** performance on a
procedural generator, exactly the limitation documented for the grounder in
section 5a. It demonstrates that the pipeline learns and grounds; it is not a
generalisation result and must not be quoted as one.

Affordance hotspots are produced (typeable confidence ~0.95 on an editor
frame). Determinism note: the cortex is an **active-vision** model — its fovea
fixation is state carried between calls, so two consecutive `look_at` calls on
the same frame can return slightly different coordinates by design. Given an
identical fixation, the output is bit-identical.

### Where this leaves the System 1 / System 2 split

Unchanged in substance from section 6. `forge/vision/cortex/` is a second
System 1 implementation alongside `avcs/streaming_v1/`. It now has trained
weights and grounds language to coordinates, but it still has **no text
decoder** and does not speak. The natural-language "brain" remains System 2,
and System 2 still defaults to `MockReasoningProvider` — wiring a real
`LLMReasoningProvider` is required before anything in this repository produces
free-form speech about a live screen.
