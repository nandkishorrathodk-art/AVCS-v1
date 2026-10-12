# Experimental temporal visual learning

AVCS now has an experimental self-supervised model prototype in
`avcs/temporal_cortex.py` and an in-memory pair buffer/training API in
`avcs/training/train_temporal.py`. It does not use click labels, OCR, DOM, or
application names.

## Model contract

`TemporalVisualCortex` accepts two adjacent RGB tensors in `[0, 1]` and returns:

- `latent`: a learned vector for the current frame;
- `predicted_latent`: a prediction of the current latent from the previous frame;
- `motion`: a bounded, dimensionless displacement field used to warp the prior
  frame; it is not a calibrated pixel-per-second optical-flow estimate;
- `surprise`: the per-pixel photometric residual after that learned warp.

The training objective combines latent prediction, VICReg-style batch variance
and covariance terms, photometric warp reconstruction, and motion smoothness.
The batch-statistic anti-collapse terms require at least two pairs per batch.
This is an initial research baseline, not proof that its latent is useful or
that its motion field corresponds to physical motion.

The live collector can pair samples separated by a configurable
`--pair-stride` while capturing at the same `--target-hz`. For example, stride
2 at 10 Hz pairs samples about 200 ms apart and yields about half as many
training pairs; capture-rate limits can make the actual interval longer.
This increases the observed time gap but does not itself produce calibrated
motion labels or guarantee more learnable movement.

An optional `--change-weight` applies only to photometric reconstruction.
Pixels whose mean absolute RGB difference exceeds `--change-threshold` get
weight `1 + change_weight`; unchanged pixels retain weight 1, and weights are
normalized per pair. The default weight is zero, preserving the uniform
baseline. `changed_pixel_fraction` is reported at the configured threshold.
This is a controlled loss ablation, not proof that changes are semantically
important or that the model learned correct motion.

## In-memory use

```python
from pathlib import Path

from avcs.temporal_cortex import TemporalVisualCortex
from avcs.training.train_temporal import FramePairBuffer, train_temporal_cortex

buffer = FramePairBuffer(capacity=256, frame_size=(128, 128))
buffer.append(previous_rgb_uint8, current_rgb_uint8)
buffer.append(next_previous_rgb_uint8, next_current_rgb_uint8)

model = TemporalVisualCortex()
loss_history = train_temporal_cortex(
    model,
    buffer,
    epochs=1,
    batch_size=8,
    checkpoint_path=Path("runs/temporal_cortex/experiment-01.pt"),
)
```

`FramePairBuffer` is a bounded ring buffer and writes no frames. When source
frames do not match `frame_size`, they are resized proportionally and centered
on a black canvas; content is not stretched or cropped. This preserves geometry
but introduces padded regions that are part of the model input. A checkpoint is
written only when `checkpoint_path` is explicitly given; it contains model
weights and training configuration but no input frames. The example uses only a
small recent window; it does not retain or train over 5–10 minutes of
full-resolution video. A live capture service and separately measured
inference budget are not implemented here. Training on a tiny recent buffer
can overfit; loss reduction alone is not a success criterion.

## Explicit live collection and first-run command

The Windows collector requires an explicit live-capture flag and checkpoint
path. It captures the full primary desktop, downsamples adjacent frames into a
bounded RAM buffer using aspect-preserving resize with letterbox padding, and
writes only model weights and metadata after training.
Close private windows before starting. Keep the desktop visually active during
collection if evaluating temporal change; do not use sensitive content.

```powershell
python -m avcs.training.collect_temporal `
  --confirm-live-capture `
  --duration-seconds 10 `
  --target-hz 10 `
  --pair-stride 1 `
  --change-weight 0 `
  --capacity 200 `
  --epochs 3 `
  --seed 17 `
  --checkpoint "runs\temporal_cortex\temporal_stream_a.pt"
```

The command prints capture progress, random-initialization and post-training
losses on the same buffer, and the checkpoint path. That comparison is a
training-fit diagnostic only; it is not held-out evaluation. `Ctrl+C` stops
capture before training/checkpoint saving.

Small real-desktop Stream A training runs and separate held-out captures have
been performed, including operator-labeled regimes. These are exploratory
single-session measurements, not independent scene-coverage evidence; labels
are not automatically verified, and the captures do not establish useful
general representations or real-time task performance. The collector records
the model initialization seed so held-out evaluation can reproduce the
matching random-initialization baseline.

## Stream A / held-out Stream B comparison

First train a fresh Stream A checkpoint using the explicit live collector above.
Then, at a separate time and on a visually distinct, non-sensitive scene, run
the held-out evaluator:

```powershell
python -m avcs.training.evaluate_temporal `
  --confirm-live-capture `
  --checkpoint "runs\temporal_cortex\temporal_stream_a.pt" `
  --regime static_micro_motion `
  --duration-seconds 10 `
  --target-hz 10 `
  --capacity 200 `
  --report "runs\temporal_cortex\stream_b_static.json"
```

Stream B stays in a bounded RAM buffer. The evaluator compares the frozen
Stream A checkpoint with a random-initialized model created from the exact
initialization seed recorded in that checkpoint. Its report includes each
self-supervised loss component, mean/minimum per-dimension latent standard
deviation, off-diagonal covariance RMS/Frobenius norm, model-predicted
dimensionless motion magnitude, and mean surprise. `--report` writes metrics
only, never frames. Checkpoints without a recorded initialization seed cannot
produce a matched baseline and are rejected.

Keep Stream B separate from Stream A in both time and content; the software
cannot verify that the operator chose a distinct scene. The three regime names
are operator annotations, not automatic scene detection. Run the command once
per regime after preparing each non-sensitive scene:

```powershell
python -m avcs.training.evaluate_temporal --confirm-live-capture --checkpoint "runs\temporal_cortex\temporal_stream_a.pt" --regime structured_scroll --report "runs\temporal_cortex\stream_b_scroll.json"
python -m avcs.training.evaluate_temporal --confirm-live-capture --checkpoint "runs\temporal_cortex\temporal_stream_a.pt" --regime dynamic_scene --report "runs\temporal_cortex\stream_b_dynamic.json"
```

## Temporal-gap and change-weight ablation

Do not combine the two interventions in the first comparison. To isolate
temporal gap, train a separate checkpoint with `--pair-stride 2` and
`--change-weight 0`; then evaluate using `--pair-stride 2`. To isolate
change-weighting, train another fresh checkpoint with stride 1 and a positive
weight (for example `--change-weight 4`), then evaluate with the same loss
settings. Each training run needs fresh explicit live-capture consent and a
different checkpoint path. The evaluator defaults to the loss configuration
stored in the checkpoint; optional `--change-threshold` or `--change-weight`
override only evaluation scoring, not model weights.

Example Stream A ablation commands (run separately, after preparing the
non-sensitive desktop scene):

```powershell
python -m avcs.training.collect_temporal --confirm-live-capture --duration-seconds 15 --target-hz 10 --pair-stride 2 --change-weight 0 --capacity 200 --epochs 3 --seed 17 --checkpoint "runs\temporal_cortex\temporal_stride2.pt"
python -m avcs.training.collect_temporal --confirm-live-capture --duration-seconds 15 --target-hz 10 --pair-stride 1 --change-weight 4 --change-threshold 0.03 --capacity 200 --epochs 3 --seed 17 --checkpoint "runs\temporal_cortex\temporal_change_weighted.pt"
```

After each run, use the held-out evaluator with matching `--pair-stride` and
operator-selected regime on a distinct scene. A change-weighted objective can
have a different numeric scale; compare trained and random models under the
same evaluation settings, and do not compare raw totals across different
weighting configurations.

These metrics do not establish zero physical motion, coherent directional
optical flow, or surprise calibration. The model's motion field is learned and
dimensionless; surprise is a photometric residual. A lower held-out loss is
evidence only about transfer on that objective and sample. Do not infer
general visual understanding, biological fidelity, calibrated optical flow,
60 FPS, or task success from it.

## Same-stream three-arm full-frame and crop comparison

To compare input views without changing the captured stream, run the region
comparison against one frozen checkpoint. It keeps adjacent frames only in
bounded RAM and first letterboxes each captured frame onto the model's square
input canvas. Changed pixels are detected there by raw RGB mean absolute
difference. The active crop is a fixed square, centered on the changed-pixel
centroid and clamped to the canvas bounds; default `--fixed-patch-size 32` can
be changed to 48 or another size from 32 through `--frame-size`. The active
patch and equal-size static control are each resized to the same model input.
It evaluates three arms on the exact same paired samples:

1. Full frame.
2. The changed-region crop.
3. An equal-dimension control crop selected outside the changed-region bounds,
   with every pixel checked to ensure it remains below the same change threshold.

Unchanged pairs are counted but excluded from all three scored arms. Changed
pairs without a valid non-overlapping static control are counted separately
and excluded from all three, preserving matched sample counts. At least two
valid triplets are required. Checkpoint loss settings are used for every arm
unless explicitly overridden.

```powershell
python -m avcs.training.compare_temporal_regions `
  --confirm-live-capture `
  --checkpoint "runs\temporal_cortex\temporal_stream_a.pt" `
  --duration-seconds 10 `
  --target-hz 10 `
  --capacity 200 `
  --fixed-patch-size 32 `
  --pixel-threshold 8 `
  --report "runs\temporal_cortex\region_comparison.json"
```

Use only after closing private windows and confirming the visible desktop is
non-sensitive. The optional JSON report contains aggregate metrics and timing,
not frames or crop coordinates. At least two valid paired triplets must be
captured for the batch-statistic comparison. If few pairs pass the raw-pixel
threshold or leave no room for a static control, the experiment stops rather
than reporting an invalid comparison.

The equal-sized active and control crops help distinguish changed-region input
from crop-and-resize geometry alone, but do not isolate geometry perfectly:
the two regions differ in spatial content, position, texture, and context. The
control-selection rule prefers a region far from the active bounds, which can
introduce screen-location effects. These scores test frozen-model input-view
sensitivity on one sample; they do not demonstrate better target grounding,
motion understanding, scene recognition, or task performance. Repeat across
independent sessions and report session-level paired differences and
uncertainty before drawing broader conclusions.

Each report includes a generated session ID, valid paired-triplet count, and a
per-session component table with all three arm metrics plus active-minus-control
and active-minus-full deltas. Between-session uncertainty remains `null` for a
single run; collect reports from independently prepared sessions before
estimating cross-session uncertainty. The 32/48 pixel sizes refer to patches on
the model input canvas, not source-monitor pixels or increased capture
resolution.
