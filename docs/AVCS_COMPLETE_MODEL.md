# AVCS Complete Cortex — Fully-Learned Model Card

This document describes the fully-learned version of the AVCS System 1 cortex.
It is the model the repo builds when the requirement is "no hardcoded
behaviour": every value the model reports at inference time is produced by a
trained tensor.

## What changed and why

The earlier cortex had learned components but kept several hand-written rules in
the inference path. Those are now learned heads (`forge/vision/cortex/core/learned.py`):

| Was hardcoded | Now learned | Where |
|---|---|---|
| Hashed character n-gram query embedding (parameter-free) | `LearnedTextEncoder` over raw UTF-8 bytes | `learned.py` |
| Greedy 2D non-maximum suppression over the affordance map | `SlotDetector` (DETR-style learned query slots) | `learned.py` |
| Single fixed category label | `SceneUnderstandingHead` multi-label attributes + regressed object count | `learned.py` |
| Controller chose the action | `ActionPolicyHead` (action distribution + action target) | `learned.py` |
| Schmitt-trigger thresholds `0.55 / 0.22 / 0.08` | `LearnedStabilityHead` with a learned previous-state embedding | `learned.py` |

The dense affordance map, the category head, the predictive world model and the
language modulation head were already learned and are unchanged.

## Outputs of a single forward pass

`ArtificialVisualCortex.perceive()` returns a `CortexState` with, in addition to
the original fields:

- `objects` — up to `num_object_slots` detected objects, each with a box,
  an objectness value, a class and a class confidence.
- `scene_attributes` — sigmoid probabilities over the multi-label attribute set.
- `object_count` — a regressed count (softplus, so never negative).
- `action` / `action_target` / `action_confidence` — the learned action choice
  and where to apply it.
- `stability_state_learned` — the learned temporal stability state.

## Parameter count and latency

| Retina input | Parameters | Latency (4 CPU cores) |
|---|---|---|
| 256x256 (default) | 1,678,779 | ~27 ms (~37 FPS) |
| 128x128 (`realtime_config()`) | 1,678,779 | ~10 ms (~95 FPS) |

The backbone dominates the cost and scales with the square of the retinal input
size; the learned heads together add under 1 ms. `realtime_config()` in
`forge/vision/cortex/config.py` selects the 128x128 preset.

## Training

The model trains end to end on CPU. Ten loss terms are active:

```
predictive + affordance + category + grounding
         + slots + attributes + count + action + stability
```

The slot loss uses greedy box matching and applies the box/class terms only to
matched slots. The affordance loss resamples the target grid to the logits'
spatial size, so it does not depend on the configured input resolution.

```bash
python3 -m forge.vision.cortex.training.trainer --epochs 4 --batch-size 8 --samples 200
```

## Honest limits

- The checkpoint in `runs/weights/avcs_cortex_weights.pt` was trained for four
  epochs on 200 synthetic samples. It is a working, fully-learned model, not a
  converged one. Category accuracy at that budget is roughly 4/8 on synthetic
  screens and the heads' outputs sit near their initialisation.
- Training data is synthetic. Real-desktop generalisation is untested here.
- The model is System 1 only: it perceives and proposes an action. It does not
  read arbitrary text, plan, or speak. That is System 2's role.
- No component sends frames off the machine; the model is CPU-only.
