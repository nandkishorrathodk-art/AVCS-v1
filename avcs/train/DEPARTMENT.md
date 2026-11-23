# Department: Train (`avcs.train`)

## 1. Responsibilities & Boundaries
- Model training routines, optimization loops, and validation pipelines.
- Scaled behavioral cloning (BC) and Stage 3 DAgger training orchestration (`run_phase4_scaled_bc_training`).
- Learning rate schedulers, mixed-precision (torch.amp) training, checkpointing, and metrics reporting.
- Kaggle GPU automation and headless cluster training execution.

## 2. Allowed Imports
- Standard Library (`os`, `sys`, `time`, `pathlib`, `typing`, `json`).
- `torch`, `torch.optim`, `torch.cuda`.
- `avcs.protocol`, `avcs.models`, `avcs.data`.

## 3. Forbidden Imports
- STRICT LINUX/KAGGLE SAFE: NEVER eagerly import Windows-only modules (`win32gui`, `ctypes.windll`, `avcs.os_io`).
- NEVER import live Windows desktop runners (`avcs.runtime.runner`).

## 4. Sub-Agent Contract
- Ingests datasets from `avcs.data` and instantiates model architectures from `avcs.models`.
- Emits reproducible checkpoints and training diagnostics (`latency_stats`, `loss_curves`, `eval_metrics`).
- Runs headless across single-GPU (e.g. Kaggle T4/P100), multi-GPU, or CPU dry-run environments.
