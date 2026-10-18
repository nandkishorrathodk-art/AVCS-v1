# AVCS Streaming Visual Cortex V1 — Research Implementation

## Status

This is a separate experimental branch of the AVCS visual model. It does not
reuse or modify the System 1 UI-grounding baseline or clean-lineage candidate.
The implementation provides a multi-scale spatial encoder, a recurrent
ConvGRU state, a learned next-feature predictor, a pixel-change head, and a
normalized displacement head. These are engineering mechanisms inspired by
temporal visual processing; they are not a simulation or validated account of
the biological visual cortex.

The checked-in corpus produced by the included generator is deliberately
procedural and its manifest marks training rights as pending. A valid signed
approval bound to that exact manifest and a separately pinned trust file is
required before persistent training. No such approval for the new manifest is
assumed from approval of the earlier, unrelated 112-frame corpus.

## Existing model and data audit

`AVCSSystem1V1` is a spatial pyramid with UI region proposals, a query
grounder, and word-recognition heads. Its task and annotations are UI
grounding-oriented. The earlier `TemporalVisualCortex` processes adjacent
frames with a pairwise encoder and does not carry a recurrent state between
runtime calls. The 112-frame clean-room corpus consists of individual generated
UI-like frames and is not a timestamped recording set; it is not repurposed as
temporal video evidence.

The preserved checkpoints are:

| Checkpoint | SHA-256 | Action |
|---|---|---|
| `runs/weights/candidates/avcs_system1_v1.pt` | `c30eb0686da96eb6f67981021bcb646b60b3ed6b050927ee97085e59ee6125a4` | Preserved |
| `runs/weights/clean_lineage_candidates/avcs_system1_clean_v1.pt` | `8e612ff9316a4985c6d75877807bc1c8cfd1caa20b096725e96495b6432b9530` | Preserved |

The current machine is Windows 11 with 10 logical CPUs and CPU-only PyTorch
2.14.1. OpenCV, PyAV, MSS, imageio, and FFmpeg are not installed in the isolated
environment. The live Windows capture path uses the repository's GDI adapter;
the model itself has no capture-library dependency. Recorded replay currently
uses the documented timestamped NPZ sequence format rather than arbitrary MP4.

## Architecture and learned outputs

`avcs/streaming_v1/model.py` implements `StreamingVisualCortexV1`:

1. Three convolutional stages form half-, quarter-, and eighth-resolution
   features. Lateral projections are fused at the coarsest grid.
2. A ConvGRU updates a spatial hidden state for every incoming frame. A
   learned positive decay rate attenuates state according to elapsed time.
3. A latent projection and predictor produce a next-observation feature
   prediction.
4. A change head emits a dense change probability map.
5. A motion head emits normalized sampling-grid displacement. It is not
   calibrated optical flow or pixels-per-second velocity.

There is no language generation, OCR, button detection, coordinate grounding,
or motor-control path in this model.

`avcs/streaming_v1/runtime.py` maintains the stateful inference loop and a
one-frame latest-value buffer. Capture and inference run in separate threads.
When capture outruns inference, pending frames are replaced rather than queued.
Duplicate pixels and old/reordered packets are counted and skipped. A long
timestamp gap resets recurrent state before updating on the new frame.

`avcs/streaming_v1/sources.py` provides paced NPZ sequence replay and opt-in
Windows GDI screen capture. Screen capture requires explicit
`confirm_screen_capture=True`; captured pixels are processed in memory and are
not recorded by the benchmark.

## Sequence format and splits

Each recording is a compressed NPZ with:

- `frames`: `uint8[T,H,W,3]` RGB pixels.
- `timestamps`: finite, strictly increasing `float64[T]` seconds.
- Optional temporal labels, each declared `available`, `not_collected`, or
  `ambiguous` in the manifest. The initial procedural fixture has per-pixel
  change masks and normalized moving-object displacement labels. Object
  identity and explicit occlusion-event annotations are not collected.

The JSON manifest stores recording ID, source ID, source-level split,
dimensions, timestamps by reference, provenance, annotation availability, and
SHA-256 for each sequence. The validator rejects path escape, data/hash/schema
mismatch, invalid times, and source IDs shared between train, validation, and
test.

The generated fixture currently has 20 independent recordings and 480 frames
(12 train, 4 validation, 4 test; 24 frames per recording). Its manifest SHA-256
is `0a8fccbceb115307edff1863628680c725756bd613ff4fa63e7122f614fb97d0`.
The generator creates only deterministic colored shapes on a textured
background, with a temporary occluding bar. These data validate the code path
and controlled motion/change objectives only. They are not evidence of screen,
video, object, or real-world generalization. Its rights status remains
`PENDING_OWNER_ATTESTATION`.

## Reproduce the data pipeline

From the repository root, in an environment with the project dependencies:

```powershell
python -m avcs.streaming_v1.build_dataset --output datasets\streaming_v1_synthetic_0_1
```

The generator refuses to overwrite a non-empty output directory. Re-running
the same seed into a fresh directory yields the same recording bytes and
manifest. The default has 12 train, 4 validation, and 4 test source recordings
of 24 frames each at 64x64 and 10 Hz. The manifest SHA-256 is printed and must
be treated as the exact corpus identity.

## Rights gate and training command

Create an owner-reviewed Ed25519 approval JSON for the exact generated
`manifest_sha256`, with all required permission decisions and
`internal_training: GRANTED`. Verify it against the project's independently
pinned trust configuration. The signature used for the older clean-room UI
manifest does not authorize this dataset. No signing key is created by AVCS.

PowerShell example after the separate approval and trust files have been
reviewed and provisioned:

```powershell
$env:AVCS_RIGHTS_TRUST_CONFIG_SHA256 = "<out-of-band SHA-256 of trusted key configuration>"
python -m avcs.streaming_v1.train `
  --dataset datasets\streaming_v1_synthetic_0_1 `
  --approval docs\STREAMING_V1_RIGHTS_APPROVAL.json `
  --trust-config docs\STREAMING_V1_RIGHTS_TRUST.json `
  --output runs\weights\streaming_v1\avcs_streaming_v1_candidate.pt `
  --epochs 12 --batch-size 2 --seed 731 --learning-rate 0.001 --patience 3
```

Training always starts from a seeded random initialization and records an empty
pretrained-checkpoint list. Each completed epoch writes an atomic recovery
checkpoint. If interrupted, resume only the same run with `--resume`; its
manifest hash, seed, model configuration, batch size, learning rate, and
patience must match. Final output paths are never overwritten.

Objectives are next-latent Smooth-L1 prediction, pixel-change BCE, masked
synthetic displacement L1, and a small latent-variance regularizer. Validation
recordings select the best epoch; the test recordings are excluded from
training and model selection.

## Held-out evaluation and runtime benchmark

```powershell
python -m avcs.streaming_v1.evaluate `
  --checkpoint runs\weights\streaming_v1\avcs_streaming_v1_candidate.pt `
  --dataset datasets\streaming_v1_synthetic_0_1 `
  --split test `
  --output runs\weights\streaming_v1\test_evaluation.json
```

The evaluator reports source counts, pixel-change precision/recall/F1,
displacement endpoint error in normalized grid units, future-feature prediction
error, latent cosine similarity on unchanged transitions, model-only latency,
and source-level bootstrap intervals. It also compares against random
initialization with the same seed and an inference ablation that resets
recurrent state each frame. Results from the generated corpus remain synthetic
research measurements, not generalization claims.

## Regression checks

Run the focused model/runtime, legacy pairwise-temporal, and signed-rights
verification tests with:

```powershell
python -m pytest tests\test_streaming_v1.py tests\test_temporal_cortex.py tests\test_cleanroom_rights_hardening.py -q
```

The V1.1 implementation session ran this command with the repository virtual
environment and observed `51 passed`. The suite exercises deterministic
initialization, recurrent state, irregular timestamps, frame ownership,
latest-frame replacement, capture recovery, interruption-gap reset, replay,
manifest integrity, sequence ingestion, unavailable-label masking,
training-rights fail-closed behavior, and cryptographic approval verification.
It does not measure learned temporal quality or live-screen accuracy.

## Completed clean-lineage synthetic experiment (2026-10-18)

The owner-signed approval for the exact manifest was verified with the
independently pinned trust configuration. It grants internal training only;
commercial inference, public weight redistribution, and source dataset
redistribution are denied. The candidate was trained on CPU from random
initialization with seed 731, batch size 2, AdamW learning rate 0.001, and a
12-epoch maximum (validation-selected best epoch: 12). No pretrained weights
or source checkpoints were loaded. Environment: Python 3.14.7,
PyTorch 2.14.1+cpu, NumPy 2.5.4, Pillow 13.0.0. Training took 64.07 seconds;
validation total loss fell from 0.52453 at epoch 1 to 0.10623 at epoch 12.

Artifacts under `runs/weights/streaming_v1/`:

- `avcs_streaming_v1_candidate.pt` — SHA-256
  `befe03f21fa5c977881ed267b445c8573f2cd491c437e948fb6ce11980a0f1f6`.
- `avcs_streaming_v1_candidate.json` — checkpoint integrity sidecar.
- `avcs_streaming_v1_candidate.training.json` — configuration, lineage, and
  per-epoch training/validation metrics.
- `test_evaluation.json` — source-held-out evaluation and ablations.
- `replay_runtime_benchmark.json` — bounded 10-Hz replay measurements.

The untouched synthetic test split contains only 4 recordings / 96 frames.
Trained change-map pixel F1 was 0.00624 (source-bootstrap 95% interval
0–0.01399), versus 0.00042 at matched random initialization and 0.01786 when
recurrent state was reset every frame. Change recall was only 0.00315. Mean
normalized-grid motion endpoint error was 0.05329 (95% interval
0.04427–0.05949), versus 0.08824 at random initialization and 0.07552 in the
no-recurrence ablation. Future-feature MSE was 0.07956 (95% interval
0.07893–0.08018), versus 1.02856 at random initialization and 0.15855 without
recurrence. Stable-frame latent similarity was unavailable (`null`): the
fixture has no qualifying unchanged pairs for that metric. The weak change
recall and the change-F1 ablation show that these results are not uniformly
better than the controls.

One held-out synthetic recording (`test-0016`, 24 frames) was replayed at its
recorded 10-Hz cadence through the continuous observer: observed capture/update
rate 9.39 FPS, model-only P50/P95 6.85/9.86 ms, capture-to-update P50/P95
7.93/10.69 ms, zero replaced/stale/duplicate/out-of-order frames, and an
estimated 52% of one CPU core over this short run. This verifies a bounded
recorded replay under this fixture only; it is not a live desktop benchmark or
evidence of real-world generalization. Live capture remains unrun.

Reproduce the training and test evaluation using the verified trust pin:

```powershell
$env:AVCS_RIGHTS_TRUST_CONFIG_SHA256 = "7d5b97c892dafae82cf654e9b1f52020152a0feffb8a1e51f1708533b7b3447d"
python -m avcs.streaming_v1.train --dataset datasets\streaming_v1_synthetic_0_1 --approval docs\STREAMING_V1_RIGHTS_APPROVAL.json --trust-config docs\CLEANROOM_RIGHTS_TRUST.json --output runs\weights\streaming_v1\reproduction\avcs_streaming_v1_candidate.pt --epochs 12 --batch-size 2 --seed 731 --learning-rate 0.001 --patience 3
python -m avcs.streaming_v1.evaluate --checkpoint runs\weights\streaming_v1\reproduction\avcs_streaming_v1_candidate.pt --dataset datasets\streaming_v1_synthetic_0_1 --split test --output runs\weights\streaming_v1\reproduction\test_evaluation.json
```

The separate `reproduction` path avoids overwriting the delivered run. Training
refuses to overwrite any existing output; choose another unused path for
additional runs.

Live capture is deliberately not started by training or evaluation. After
loading an internally approved, trained checkpoint, the operator can run:

```powershell
python -m avcs.streaming_v1.live `
  --checkpoint runs\weights\streaming_v1\avcs_streaming_v1_candidate.pt `
  --confirm-screen-capture --duration-seconds 60 --target-hz 10
```

Close private windows before confirming. The report separates configured and
observed capture rate, model update rate, model-only P50/P95, capture-to-update
P50/P95, latest-frame replacements, stale/duplicate/out-of-order frames,
recovery count, process CPU time, and process working set. A single local
session is an operational benchmark, not independent evaluation.

## Minimum next data collection

The procedural fixture is intentionally insufficient for the stated goal.
Before making a real-screen claim, collect owner-authorized recordings from
multiple independent sources and environments, with source-level partitions.
As a first engineering evaluation (not a guarantee of scientific sufficiency),
target at least 30 independently recorded sessions: 20 training, 5 validation,
and 5 untouched test sessions, each 2–5 minutes at 5–10 FPS, spanning stable
scenes, gradual movement, abrupt scene changes, object entry/exit, occlusion,
frame-rate changes, and brief capture interruptions. Keep raw pixels local,
record capture parameters and permission provenance, and annotate a reviewed
subset of change regions, visible-object persistence, and event times. Choose
sample size again after measuring between-session variance and desired
confidence intervals.

## Limitations and release status

- The signed approval authorizes internal training for this exact synthetic
  manifest only. It does not authorize commercial inference or redistribution
  of weights or source data.
- No independent, permissioned, human-annotated real recording dataset is
  present. One later passive live session measured runtime only; it was not an
  accuracy evaluation and no screen content was retained.
- The candidate is a research artifact, not a validated real-time or
  general-purpose visual perception model. Human-like perception, occlusion
  memory, and temporal generalization remain unproven until evaluation on
  untouched, permissioned real recordings and a separately authorized live
  benchmark.
- No package publication, checkpoint distribution, commit, or deployment was
  performed.

  ## V1.1 diagnosis and controlled ablations (2026-10-18)

  ### Diagnosis

  - Generated change masks match exact adjacent-frame RGB inequality on the
    fixture. The transitions are ordered and the ConvGRU state is threaded
    through `forward_sequence`; a gradient probe found nonzero gradients in the
    encoder, recurrent unit, and task heads. The initial low score was not caused
    by a reversed sequence or disconnected recurrent state.
  - Changed pixels are sparse: 1.69% of train, 1.39% of validation, and 2.02%
    of test transition pixels. Original per-pixel BCE had no positive-class
    weighting. At a fixed 0.5 threshold the first model was nearly all-negative,
    making change recall and F1 very low.
  - A representative training batch produced nonzero gradient norms in the
    encoder (sum 0.049), ConvGRU (0.069), future predictor (0.452), change head
    (0.020), and motion head (0.880). This rules out a disconnected temporal
    path in that batch, but shows the change head's signal was much smaller than
    the motion-head signal; it is not a claim of general gradient health.
  - The old fixed threshold was not enough to characterize the model. V1.1
    validation-selected a threshold on a fixed grid, then evaluated once on the
    held-out test set. The test split did not choose thresholds.
  - The old change head had no explicit absolute feature-difference input. The
    new `convgru_delta` mode adds that input. `spatial` is a trained no-state
    ablation; `convgru` is the same recurrent architecture without the explicit
    delta path.
  - The data remains extremely simple: one moving colored square, static
    textured backgrounds, and a short rectangular occluder. Change labels are
    exact pixel differences, not human-reviewed semantic events. There are no
    qualifying stable transitions in the held-out split, so stable-latent
    consistency remains unmeasured.

  ### Controlled results

  All three V1.1 variants used manifest
  `0a8fccbceb115307edff1863628680c725756bd613ff4fa63e7122f614fb97d0`, the same
  12/4/4 recording split, seed 731, CPU, 12-epoch maximum, batch size 2,
  learning rate 0.001, patience 3, and positive change-pixel weight 8.
  Validation early stopping completed spatial-only in 9 epochs (best epoch 6),
  ConvGRU in 10 (best epoch 7), and ConvGRU+delta in 11 (best epoch 8).
  Thresholds were selected from the same ten-value grid on validation and then
  frozen for test. Test has 4 recordings / 96 frames. The 95% intervals use a
  recording-cluster bootstrap that resamples recordings and recomputes micro F1
  from the summed pixel counts; with only four sources, uncertainty estimates
  are highly limited.

  | Variant | Validation threshold | Test change F1 (95% recording bootstrap interval) | Precision | Recall | Motion EPE | Future-feature MSE | Test F1 without recurrent state |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Spatial-only | 0.50 | 0.380 (0.350–0.445) | 0.259 | 0.712 | 0.08525 | 0.23651 | 0.380 |
  | ConvGRU | 0.65 | 0.373 (0.338–0.455) | 0.260 | 0.660 | 0.05312 | 0.12092 | 0.350 |
  | ConvGRU + explicit feature delta | 0.50 | 0.406 (0.365–0.447) | 0.265 | 0.859 | 0.05914 | 0.10725 | 0.330 |
  | Historical V1, unweighted loss | 0.05 | 0.340 (0.300–0.387) | 0.208 | 0.919 | 0.05397 | 0.07956 | 0.311 |

  The V1.1 weighted-loss variants materially improve the selected-threshold
  change scores over the original fixed-0.5 run, but precision remains low and
  the cluster-bootstrap intervals overlap. ConvGRU+delta improves F1 relative to its
  own no-state ablation by about 0.076; this is promising only for this synthetic
  fixture, not evidence of a general temporal capability. ConvGRU and the
  spatial-only control are close on F1. Motion/prediction are also synthetic
  objectives. These results justify retaining the delta model as an experimental
  candidate, not promoting or deploying it.

  New variant checkpoints, hashes, and complete reports are under
  `runs/weights/streaming_v1_1_comparison/`. The consolidated metrics and
  threshold sweep are in `controlled_comparison.json`; each variant has its
  training report and test evaluation.

  Reproduce an individual variant by adding the selected mode and positive
  weight to the signed training command, and use its validation-selected
  threshold for test evaluation. For example:

  ```powershell
  python -m avcs.streaming_v1.train --dataset datasets\streaming_v1_synthetic_0_1 --approval docs\STREAMING_V1_RIGHTS_APPROVAL.json --trust-config docs\CLEANROOM_RIGHTS_TRUST.json --output runs\weights\streaming_v1_1_comparison\reproduction\candidate.pt --epochs 12 --batch-size 2 --seed 731 --learning-rate 0.001 --patience 3 --change-positive-weight 8 --temporal-mode convgru_delta
  python -m avcs.streaming_v1.evaluate --checkpoint runs\weights\streaming_v1_1_comparison\reproduction\candidate.pt --dataset datasets\streaming_v1_synthetic_0_1 --split validation --change-threshold 0.50
  python -m avcs.streaming_v1.evaluate --checkpoint runs\weights\streaming_v1_1_comparison\reproduction\candidate.pt --dataset datasets\streaming_v1_synthetic_0_1 --split test --change-threshold 0.50
  ```

  ### Data and live-observation blockers

  No new dataset was generated or ingested: the existing signed approval is
  bound only to the 20-recording synthetic manifest. A richer dataset requires
  permissioned recordings, a source-disjoint manifest, per-recording hashes,
  timestamp/annotation validation, provenance and an owner-signed approval for
  the resulting exact manifest hash before training. At minimum, additional
  recordings must include genuinely stable periods, gradual motion, entry and
  exit, multiple moving elements, partial and full occlusion, abrupt scene
  changes, and variable-rate/drop/interruption events. The project guide's
  30-session target (20 train / 5 validation / 5 untouched test, 2–5 minutes
  each) remains a starting data-collection plan, not a claim of statistical
  sufficiency.

  The new `python -m avcs.streaming_v1.ingest --index <index.json> --output
  <new-dataset-dir>` pipeline accepts prepared, permission-documented NPZ
  recordings with RGB `frames` and strictly increasing original `timestamps`.
  The index supplies stable source IDs, source-level splits, rights-evidence
  references, permitted uses, and an explicit `available` / `not_collected` /
  `ambiguous` status for each supported annotation. Ingestion copies and hashes
  the sequence arrays, prevents path escape/overwrite and source leakage, and
  leaves manifest-level training rights pending for the separate signer. It
  does not decode arbitrary video containers; video decoding and its
  reproducible timestamp/color conversion need to be performed and recorded
  before NPZ ingestion. The current trainer requires change and motion labels
  for its train/validation splits. The evaluator scores change only on
  explicitly labeled recordings and reports motion only where its labels are
  available. Unlabeled recordings remain in sequence evaluation but do not
  become negative examples.

  After separate explicit authorization and confirmation that private windows
  were closed, the V1.1 delta candidate ran a 30-second passive local Windows
  desktop session on 2026-10-18. It captured and processed
  277 frames: 9.23 capture/update FPS at configured 10 Hz, zero replaced, stale,
  duplicate, or out-of-order frames, and no capture/inference error. End-to-end
  capture-to-update P50/P95 was 30.85/33.66 ms; model-only P50/P95 was
  8.44/10.09 ms. Process working set was 270,249,984 bytes. Process CPU time
  corresponded to 594% of one logical CPU (about 59% of this machine's ten
  logical CPUs). These are runtime metrics, not proof that the model correctly
  understood the live screen; no screen content was persisted. The aggregate
  record is `runs/weights/streaming_v1_1_comparison/passive_live_20261018.json`.

## V1.2 real-sequence evidence and ingestion contract (2026-10-18)

### What V1.1 has demonstrated

- Across four procedurally generated held-out recordings, the trained models
  assign higher mean change-head scores to exact RGB-changed pixels than to
  unchanged pixels. For example, ConvGRU+delta's mean scores were 0.643 on
  changed pixels and 0.054 on unchanged pixels.
- Against matched random initializations, spatial-only F1 increased from
  0.015 to 0.380 and ConvGRU F1 from 0.000 to 0.373 at their frozen
  validation-selected thresholds. ConvGRU+delta increased from 0.213 to 0.406.
  This is measurable learning for this narrow synthetic pixel-change task, not
  evidence of learning semantic scene events.
- At its validation-selected threshold of 0.50, ConvGRU+delta produced 6,549
  true-positive, 18,126 false-positive, and 1,075 false-negative pixels on the
  92 held-out transitions. Pixel precision was 0.265, recall 0.859, and F1
  0.406. The other two V1.1 variants also had only 0.259–0.260 precision.
- The model has a working recurrent state and showed nonzero recurrent
  gradients in the prior training diagnostic. On this particular synthetic
  split, ConvGRU+delta exceeded its reset-state ablation's pixel F1
  (0.406 vs 0.330), but the four-recording intervals overlap.
- The feature-delta variant has lower future-feature prediction MSE than the
  two alternatives on this synthetic test split. That is not evidence that it
  predicts meaningful future scenes: the target is the model's own next latent
  representation.

These facts do **not** establish semantic change detection, visual object
identity, occlusion continuity, harmless-noise rejection, event surprise
detection, or real-recording generalization. Every saved V1.1 evaluation
compared pixel predictions with exact adjacent-frame RGB differences. The
synthetic frames contain a moving colored square and occasional occluder; they
do not supply human-reviewed semantic event labels. No live-screen inference
outputs were persisted; the passive-live record contains aggregate runtime
telemetry only. No further capture was performed for V1.2.

`avcs.streaming_v1.diagnose` reruns the frozen validation-selected thresholds
on test and stores per-transition confusion counts and mean model scores
separately for changed and unchanged pixels, without writing frames or dense
prediction maps:

```powershell
python -m avcs.streaming_v1.diagnose `
  --comparison runs\weights\streaming_v1_1_comparison\controlled_comparison.json `
  --dataset datasets\streaming_v1_synthetic_0_1 `
  --split test `
  --output runs\weights\streaming_v1_1_comparison\v1_2_heldout_prediction_audit.json
```

The resulting audit is in
`runs/weights/streaming_v1_1_comparison/v1_2_heldout_prediction_audit.json`.
The false-positive counts are spread across all four recordings (4,881–5,431
for ConvGRU+delta). This establishes poor discrimination against the fixture's
pixel-change labels; it does not identify whether any particular pixel is a
semantically harmless or meaningful change.

### Version 2 sequence data contract

The new `avcs.streaming.ingest-index.v2` importer requires a source path whose
SHA-256 is checked against `source_id`, in addition to the prepared RGB NPZ.
The source path must remain local and is not copied into the output dataset.
Split assignment is keyed by that verified source hash, so separately
prepared clips from one original recording must share the same `source_id` and
cannot cross splits. Ingested manifests use
`avcs.streaming.sequence.v2`; the original V1 corpus and its hash are left
unchanged. The exact output manifest still starts with
`internal_training: PENDING`; written provenance is not owner approval.
A new Ed25519 approval must bind the final manifest hash before any training.

Each index row includes a prepared `path`, the original `source_path` relative
to the index, `source_id` equal to the original file's SHA-256, its source-level
split, all eight annotation statuses, and a `provenance` object with `kind`,
`rights_evidence`, and `permitted_uses`. The importer hashes the original
source in place, verifies that `source_id` matches, but never copies that
source file. A minimal command is:

```powershell
python -m avcs.streaming_v1.ingest `
  --index path\to\recording-index.json `
  --output datasets\avcs_real_sequences_v1
```

The example output directory must be new or empty. The resulting manifest hash
is the input to a separate rights approval; ingestion itself never unlocks
training.

Supported labels:

- `change_masks`: transition-indexed, per-pixel binary labels. Their meaning
  must be stated by the annotator; procedural V1 labels mean exact RGB
  inequality, not meaningful scene change.
- `semantic_event_labels`: transition-indexed multi-label int8 matrix using
  the manifest's fixed taxonomy (`stable`, `pixel_noise_only`, `motion`,
  `appearance`, `disappearance`, `occlusion`, `scene_transition`,
  `unexpected_event`). Values are `-1` unknown/unreviewed, `0` reviewed absent,
  and `1` reviewed present. Unknown is never converted to negative.
- `object_id_masks`: per-frame signed integer maps; `-1` means unknown, `0`
  means reviewed background, and positive values are stable object identities
  within a recording.
- `occlusion_masks`: per-frame binary masks with explicitly documented
  annotation meaning. They are not inferred from pixel differences.
- Existing motion displacement and validity masks remain supported.

`validate_manifest` checks annotation shapes, dtypes, value domains, status vs
array consistency, and the exact event taxonomy. The supervised evaluator
skips missing/ambiguous change labels, reports labelled and unlabelled
recordings separately, and emits null rather than a success-shaped score when
there are no labelled examples. Semantic event labels are counted for data
coverage and stable-pair latent cosine, but the model has no event-classification
head; therefore `semantic_event_capability` explicitly reports
`not_evaluated_no_semantic_event_head`. Self-supervised training on unlabeled
real sequences has not been implemented or run. The current supervised trainer
still requires pixel-change and motion labels for train and validation.

### Data required before training or generalization claims

No new real recording is present or newly approved in this workspace. The
existing approval applies only to the exact synthetic manifest; it does not
authorize personal desktop capture or unrelated licensed datasets. Before
ingestion, each new source needs:

1. A verifiable license or recording-consent reference explicitly permitting
   internal model training and local retention/processing. Distribution,
   commercial inference, and publication need separate rights decisions.
2. A source-file hash, stable source grouping, exact recording timestamps,
   documented decoding/color conversion, and a source-level split fixed before
   model selection.
3. Human-reviewed labels on an evaluation subset for meaningful change vs
   harmless pixel noise, stable intervals, motion, appearance/disappearance,
   tracked identities, partial/full occlusion, scene transitions, unexpected
   events, and ambiguous/unknown intervals. A negative is valid only when
   reviewed; missing annotations remain unknown.
4. Enough independent sources to prevent near-duplicate leakage. The existing
   engineering target of 30 independent sessions (20/5/5) at 2–5 minutes and
   5–10 FPS is a starting collection target, not a statistical guarantee.
   Increase sample size after estimating between-session variance and target
   confidence intervals.
5. A new validated manifest and separate owner-signed approval for its exact
   hash before training. No real-data training or checkpoint creation was
   justified or performed in V1.2.

### V1.2 evaluation status and next experiments

The current model can be evaluated for per-pixel change, labelled motion,
future-latent prediction, and latent continuity only where stable labels exist.
It cannot currently predict semantic event classes or object IDs. No
confidence-threshold-free unexpected-event method has been implemented; a
fixed pixel threshold would not establish surprise recognition. Future
comparisons should freeze data splits, use the same held-out recordings for
spatial/recurrent/delta variants, choose any operating point using validation
only, report per-recording results and source-cluster uncertainty, and include:

- pixel-level noise sensitivity as a distinct task from semantic event
  recognition;
- stable-scene feature drift and tracked-object identity continuity;
- event-level appearance/disappearance and occlusion recovery;
- future-feature prediction against matched random and persistence baselines;
- calibrated unexpected-event ranking with an explicitly defined event set
  and false-alarm rate, not an undocumented universal threshold.

The existing V1.1 weights, reports, and passive-live aggregate are preserved.
No V1.2 candidate was trained, and no new live capture was started.
