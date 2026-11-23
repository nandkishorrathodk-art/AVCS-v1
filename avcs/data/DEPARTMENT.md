# Department: Data (`avcs.data`)

## 1. Responsibilities & Boundaries
- Multi-modal dataset representations, dataloaders, and batch collators.
- LeRobot v3.0 Canonical Parquet serialization/deserialization schemas.
- Data hygiene engines: `TrajectoryCleaner` (jitter smoothing, idle frame collapse, action coalescing) and `QADoctor` (temporal leakage detection, clock skew auditing, frame index clamping).
- Synthetic episode generation and Phase 4 multi-source mixture dataset builders.

## 2. Allowed Imports
- Standard Library (`pathlib`, `json`, `dataclasses`, `time`, `typing`).
- `torch`, `torch.utils.data.Dataset`, `torch.utils.data.DataLoader`.
- `numpy`, `PIL.Image`.
- `pyarrow`, `pyarrow.parquet`.
- `avcs.protocol` (for action and coordinate schemas).

## 3. Forbidden Imports
- STRICT LINUX/KAGGLE SAFE: NEVER eagerly import Windows-only modules (`win32gui`, `win32api`, `ctypes.windll`, `avcs.os_io`).
- NEVER import runtime autopilot execution engines (`avcs.runtime.runner`).

## 4. Sub-Agent Contract
- Produces normalized, sanitized trajectory batches for model training (`avcs.train`) and evaluation (`avcs.eval`).
- Guarantees 100% clean imports on headless Linux and Kaggle GPU notebook instances.
