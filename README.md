# AVCS — Artificial Visual Cortex System

AVCS is an experimental Python research project and SDK for screen visual perception.
It contains actual neural-network models, in-memory desktop capture pipelines, and empirical
evaluation harnesses.

**AVCS is not a single model**, and it is **not a proven, general-purpose computer-use model**
for arbitrary desktop tasks. Passing the test suite demonstrates internal software specification
adherence and tensor contract validity—not open-world visual understanding or automated task completion.

## Current status

- The `ScreenEye` SDK and supporting perception components provide the
  repository's desktop-facing API. Capture support and performance depend on
  the selected backend, hardware, operating system, and workload.
- `HumanEye2M` is an experimental model path. A checkpoint that loads
  successfully is not evidence of predictive accuracy; without suitable
  trained weights it must not be treated as a validated model.
- `GoalConditionedGrounder` and its demonstration-training pipeline are
  implemented, but a real-demonstration checkpoint and validated click
  performance are not included.
- `TemporalVisualCortex` is a self-supervised temporal research prototype.
  Its live-capture, training, and evaluation tools are documented in
  [docs/TEMPORAL_VISUAL_LEARNING.md](docs/TEMPORAL_VISUAL_LEARNING.md).
  Available desktop runs are small experiments, not evidence of general
  visual understanding, calibrated motion, or real-time task performance.

Benchmark figures and model claims should be accompanied by the exact
environment, command, workload, sample count, and report. Do not infer
end-to-end task success from unit tests, synthetic data, model checkpoint
loading, or a lower self-supervised loss alone.

## Install

Python 3.9 or later is declared by the package metadata. Create a virtual
environment, then install the project and development dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,desktop]"
```

PyTorch installation options vary by operating system and accelerator. If the
default package resolver does not select the PyTorch build you need, install
the appropriate PyTorch build for your environment before installing AVCS.
The `desktop` extra installs the optional `mss` dependency used by the
cross-platform real-screen capture backend. Without a working desktop capture
backend, live desktop operations may be unavailable; use explicit synthetic
sources only for tests and demonstrations.

## Verify

Run the repository test suite from the project root:

```powershell
python -m pytest tests -q
```

Test outcomes can vary by operating system and hardware. The tests are not a
substitute for isolated live-desktop acceptance tests.

## Temporal model research

The temporal model experiments require explicit operator consent before live
desktop capture. Close private windows and use a non-sensitive screen. Raw
frames are held in bounded memory by the documented tools; optional reports
contain aggregate metrics. See the protocol, limits, and commands in
[docs/TEMPORAL_VISUAL_LEARNING.md](docs/TEMPORAL_VISUAL_LEARNING.md).

## Contributing

Contributions are welcome. Before opening a pull request:

1. Describe the problem and the behavior change.
2. Add or update focused tests.
3. Run the relevant tests and report the exact command and result.
4. Avoid publishing screenshots, captured frames, credentials, or private
   desktop data.

## License

AVCS is made available under the Apache License 2.0. See [LICENSE.md](LICENSE.md).
The `avcs.auth` package contains optional application-level license-token
utilities; their presence does not change the repository's software license.
