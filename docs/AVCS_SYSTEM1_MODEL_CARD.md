# AVCS System 1 V1 — Model Card

**Model Name:** `AVCSSystem1V1`<br>
**Version:** `1.0.0-preview.1`<br>
**Organization:** Parth Labs<br>
**Date:** October 2026<br>
**Source Software License:** Apache 2.0<br>
**Checkpoint License:** Not granted / not cleared for redistribution<br>
**Model Type:** Unified Multi-Task Pure-Pixel Neural Visual Cortex

---

## 1. Model Summary

`AVCSSystem1V1` is an experimental local neural visual-perception model for RGB desktop-like images. It processes pixels without accessibility trees (UIA/MSAA), general-purpose OCR libraries, DOM inspection, or cloud vision APIs. CPU inference is supported. The trained checkpoint is not included in the source-only distribution, and no performance claim is made here.

The model unifies four visual-perception components into a shared-representation network:
1. **Multi-Scale Spatial Proposer:** Locates interactive UI elements (buttons, icons, input fields, tabs) across Stride 4, 8, and 16 feature pyramid levels.
2. **Goal-Conditioned Language Gaze Grounder:** Localizes natural-language target descriptions (e.g., *"Save button"*, *"Terminal icon"*) into calibrated 2D sub-pixel coordinates.
3. **Pixel-Native Word Recognizer:** Identifies functional UI words and text labels from raw pixel crops across a closed 132-class functional desktop vocabulary.
4. **Magnocellular Temporal Motion Engine:** Detects optical flow energy, motion velocity, modal surprise triggers, and scene stability across consecutive video frames.

---

## 2. Model Specifications & Checkpoint Provenance

| Metric / Parameter | Value | Verification Source |
| :--- | :--- | :--- |
| **Model Architecture** | `AVCSSystem1V1` | `avcs/perception/system1_v1.py` |
| **Total Parameter Count** | `1,663,564` (1.66M) | PyTorch Parameter Traversal |
| **Weights File** | `avcs_system1_v1.pt` | Local only; excluded from source-only release |
| **Checkpoint File Size** | `6,800,337 bytes` (6.48 MB) | Windows File System |
| **SHA-256 Hash** | `c30eb0686da96eb6f67981021bcb646b60b3ed6b050927ee97085e59ee6125a4` | Cryptographic Digest |
| **Sidecar Metadata** | `runs/weights/candidates/avcs_system1_v1.json` | JSON Schema v2 |
| **Execution Hardware** | CPU-only inference; measured on Windows 11, Intel Core Ultra 5 115U | Reproduced evaluation run |
| **Input Dimensions** | Dynamic screen size, normalized to 256×256 tensor | Standardized Preprocessing |
| **Output Contract** | Typed `VisualStateObservation` | `avcs/perception/visual_state.py` |

---

## 3. Subsystem Architecture

```
                    Raw Desktop RGB Frame [H x W x 3]
                                   │
             ┌─────────────────────┴─────────────────────┐
             ▼                                           ▼
   [Magnocellular Engine]                       [Spatial Resampling]
   - 16x24 Grid Optical Flow                     - Bilinear 256x256
   - Temporal Energy & Velocity                          │
   - Scene State & Modal Surprise                        ▼
                                                [Stem ConvBNAct 24ch]
                                                         │
                                               [Stage 1 DS-Conv 48ch] ──► P2 (Stride 4, 64x64)
                                                         │
                                               [Stage 2 DS-Conv 64ch] ──► P3 (Stride 8, 32x32)
                                                         │
                                               [Stage 3 DS-Conv 96ch] ──► P4 (Stride 16, 16x16)
                                                         │
                         ┌───────────────────────────────┼───────────────────────────────┐
                         ▼                               ▼                               ▼
                 [Category Head]               [Spatial Proposer]              [FiLM Grounder]
                 - 8 Desktop Types             - Affordance heatmaps           - 1D Byte Query Encoder
                 - Global Pool + MLP           - Bounding box spans            - 128x128 Dense Heatmap
                                               - Element kind logits           - Subpixel Peak + NMS
                                                         │                               │
                                                         ▼                               ▼
                                               [Word Recognizer Crops]         [Calibrated Presence]
                                               - 32x96 Crop Resampling         - Existence + Peak Gating
                                               - 132 UI Word Logits            - Fail-Closed Policy
                                                         │                               │
                                                         └───────────────┬───────────────┘
                                                                         ▼
                                                          [VisualStateObservation]
```

### Detailed Component Specifications

- **Stem & Backbone:** 3-stage depthwise separable convolutional backbone with 24, 48, 64, and 96 channels.
- **Feature Pyramid Network (FPN):** Lateral 1×1 projections and 3×3 smooth convolutions producing uniform 48-channel feature maps across P2 (stride 4), P3 (stride 8), and P4 (stride 16).
- **Goal-Conditioned Grounder:** Byte-level convolutional text encoder (embedding dimension 48, 1D conv layers, 128-dim projection), FiLM feature modulation, 2-layer conv heatmap head producing a 128×128 spatial logit map, and an existence projection head.
- **Subpixel Coordinate Decoder:** Discrete 2D spatial argmax followed by local 1D quadratic Taylor expansion along x and y coordinates, completely preventing DSNT center-collapse.
- **Spatial Peak Disambiguation:** 2D max pooling with an 8-pixel spatial exclusion radius and softmax-normalized candidate margin evaluation.
- **Word Recognizer:** 3-stage depthwise separable classifier accepting (B, 3, 32, 96) crops, classifying across 131 functional UI words and class 0 `<none>`.

---

## 4. Evaluation status

A local CPU evaluation was run against a procedurally generated held-out
dataset. Its JSON report and underlying frames/annotations are not part of the
source-only distribution while dataset and candidate-checkpoint lineage rights
remain undocumented. Consequently, this public model card makes no numeric
performance claim. The local report is identified by its checkpoint digest in
`docs/RELEASE_RIGHTS_AUDIT.md`; it is not independent real-world validation.

---

## 5. Decision Policy & Safety Contracts

The model enforces an uncompromising fail-closed safety policy governing the `action_authorized` field in `VisualStateObservation`:

1. **Target Presence Gating:**
   - Confidence `< 0.35`: Classify as `ABSENT`. Action strictly prohibited.
   - Confidence `0.35 .. 0.69`: Classify as `UNCERTAIN`. Action strictly prohibited.
   - Confidence `≥ 0.70`: Classify as `PRESENT`. Eligible for authorization checks.
2. **Spatial Ambiguity Check:** If multiple distinct spatial peaks exist and the softmax margin between Candidate #1 and Candidate #2 is `< 0.20`, `is_ambiguous = True` and action is prohibited.
3. **Temporal Stability Gate:** If the scene state is `busy` or `surprise` (modal dialog, animation, video playback), action is inhibited until the screen visually settles.
4. **Out-of-Vocabulary Gate:** If a query contains semantic modifiers outside the model's closed vocabulary, presence confidence is capped at `0.45` (`UNCERTAIN`) and action is inhibited.

---

## 6. Intended Uses and Known Limitations

### Intended Uses
- Standalone visual perception for desktop automation systems.
- Visual grounding layer for hybrid System 1 / System 2 cognitive agents.
- Real-time telemetry monitoring (scene stability, application categorization, UI element tracking).

### Out-of-Scope & Limitations
- **Not an OCR Library:** The model classifies UI words from a closed functional vocabulary (132 classes). It does not perform open-domain arbitrary text transcription.
- **Scale sensitivity:** Performance may vary with control size; no numeric recall claim is included in this rights-limited source release.
- **Multi-Monitor Display Topology:** Coordinate outputs are normalized `[0, 1]` relative to the captured screenshot. Multi-monitor offset transformations must be managed by the host display environment.
- **No action execution:** The model has no motor controller. `action_authorized` is only a model-side eligibility signal, not permission to click, type, or perform an irreversible action. A caller must independently enforce safety and verify outcomes.
- **Evaluation scope:** Reported results are on a small, procedurally generated synthetic dataset. They do not establish real-world accuracy, calibration, robustness, or successful task completion on arbitrary applications.

---

## 7. Data provenance, licensing, and release status

The System 1 training manifest identifies 240 procedural synthetic frames and
30 procedural negative frames, with annotations. It has no separate dataset
license or rights attestation. A separate, non-default training manifest
includes one Win32-captured frame. The training pipeline also loads three
optional candidate checkpoints; their source and license records are
incomplete. Therefore neither the training data nor the final checkpoint's
complete lineage is cleared for redistribution.

The root Apache-2.0 license covers the software source identified by that
license. It does not by itself license the checkpoint, inherited weights,
datasets, annotations, or captured images. The source-only bundle excludes
those materials and the evaluation report. See
[`RELEASE_RIGHTS_AUDIT.md`](RELEASE_RIGHTS_AUDIT.md) for evidence, statuses,
and the permissions needed for a future weight release.
