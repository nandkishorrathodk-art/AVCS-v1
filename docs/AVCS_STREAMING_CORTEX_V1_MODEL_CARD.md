# AVCS Streaming Visual Cortex V1 — Production Model Card

## Executive Summary
`StreamingVisualCortexV1` is the continuously observing, biological-inspired visual perception model for AVCS System 1.
It runs locally on CPU (~104 FPS, 9.34 ms P50 latency), maintaining an evolving recurrent spatiotemporal hidden state
across live desktop screen frames without saving raw screenshots or videos to disk, and communicating with System 2
via truthful, coordinate-free natural language dialogue.

## Model Details
- **Architecture**: `StreamingVisualCortexV1`
- **Parameter Count**: 501,612 parameters (clean random Kaiming/orthogonal initialization)
- **Checkpoints**:
  - Binary weights: `runs/weights/candidates/avcs_streaming_cortex_v1.pt`
  - SHA-256 Digest: `418c541e97b93615eccccd3cf488efaded84cd25f0384b865e3f5022c8c6032d`
  - Sidecar Metadata: `runs/weights/candidates/avcs_streaming_cortex_v1.json`
- **License**: Clean-lineage internal rights held by Nandkishor Rathod (Parth Labs)

## Key Technical Specifications
| Specification | Measured Value | Acceptance Threshold | Status |
|---|---|---|---|
| Continuous Soak Test | 1,000 ticks | >= 1,000 ticks | PASS |
| Sustained CPU Throughput | 95.41 FPS | >= 30.0 FPS | PASS |
| P50 Latency (Local CPU) | 9.74 ms | <= 35.0 ms | PASS |
| P95 Latency (Local CPU) | 17.43 ms | <= 50.0 ms | PASS |
| Memory RSS Stability | Flatline (0.0 MB growth) | < 25.0 MB | PASS |
| Latent Feature Variance | 1.0104 | >= 0.05 | PASS |
| Frozen Twin Parameter Drift | 0.00000000 | < 1e-6 | PASS |
| Coordinate Leakage to System 2 | Exactly 0 coordinate numbers | 0 numbers | PASS |
| Raw Video / Screenshot Disk Persistence | Exactly 0 bytes written to disk | 0 bytes | PASS |
| Legacy Checkpoint Protection | Baseline SHA-256 bit-for-bit preserved | 100% invariant | PASS |

## System 2 Integration & Natural Language Speech
- The model outputs purely qualitative, natural language scene descriptions (`SceneObservation`).
- Interactive controls are categorized into anatomical desktop regions (`header`, `footer`, `left_sidebar`, `right_sidebar`, `center`).
- Strictly ZERO raw coordinate numbers (`x, y`, bounding boxes) are ever exposed across the System 2 text interface.
