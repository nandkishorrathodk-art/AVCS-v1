# AVCS System 1 v1.0.0-preview.1 — source-only preview

AVCS System 1 is a local PyTorch model for visual perception of desktop-like
RGB images. This developer preview returns typed visual observations,
interactive-region proposals, a closed-vocabulary UI-word prediction, and
optional query-conditioned target grounding. It does not click, type, capture a
desktop, call an LLM, or communicate with a cloud service.

This is an experimental preview, not a validated general-purpose desktop agent.
Predictions may be wrong. `action_authorized` is a model-side eligibility flag,
not permission to dispatch a computer action; callers must apply their own
independent safety and verification policy.

## Requirements and installation

Python 3.10–3.14 and PyTorch, NumPy, and Pillow are required. The model runs on
CPU; a GPU is optional. No external LLM, cloud API, OCR engine, accessibility
service, browser DOM, or motor controller is needed by System 1 inference.

Install the source-only wheel and its declared dependencies:

```powershell
python -m pip install .\artifacts\avcs_system1-1.0.0rc1-py3-none-any.whl
```

This source-only bundle does **not** contain a trained checkpoint. The local
checkpoint and the candidate weights used to initialize parts of it have no
documented redistribution grant, and the complete candidate/data lineage is
not established. Do not seek or redistribute a copy of those weights on the
basis of this package. `load_system1()` and the CLI require a compatible
checkpoint supplied under separate, documented authorization.

The source-only package can be installed and its API exercised with
developer-created model instances. It does not provide trained-model
predictions until authorized weights are available.

## Quick start

```python
from avcs import load_system1
from PIL import Image

model = load_system1(
    checkpoint_path=r".\authorized\avcs_system1_v1.pt",
    device="cpu",
)
observation = model.observe_frame(
    Image.open("screenshot.png").convert("RGB"),
    active_query="Save button",
)
print(observation.to_dict())
```

The input may be a Pillow image or an RGB `numpy.uint8` array in
`(height, width, 3)` order. The returned JSON-compatible mapping includes
screen dimensions, a `normalized_0_1_origin_top_left` coordinate convention,
target presence and confidence, pixel and normalized target coordinates when
localized, uncertainty flags, and the fail-closed `action_authorized` flag.
Pixel coordinates are `(x, y)`; screen dimensions are `(width, height)`.

CLI:

```powershell
python -m avcs.demo_system1 --image .\screenshot.png `
  --query "Save button" --checkpoint .\authorized\avcs_system1_v1.pt --json
```

Or, after installation, use `avcs-system1-demo` with the same arguments.

## Package contents

- `artifacts/avcs_system1-1.0.0rc1-py3-none-any.whl`: installable Python package.
- `artifacts/avcs_system1-1.0.0rc1.tar.gz`: Python source distribution.
- `artifacts/model_metadata.json`: version, architecture, and rights status.
- `README.md`, `LICENSE`, and `docs/`: install, API, limitations, and model card.
- `SHA256SUMS`: SHA-256 hashes for all release payload files other than itself.

The archive intentionally excludes all checkpoint/candidate weights, dataset
frames, annotations, evaluation report, screenshots, credentials, `.env` files,
and API keys. See `docs/RELEASE_RIGHTS_AUDIT.md` for artifact-level findings.

## Reproducible verification

From the repository root, install the test tools and runtime dependencies, then
run the focused tests and pure-pixel audit:

```powershell
python -m pip install "torch>=2.0" "numpy>=1.24,<3" "Pillow>=9" "pytest>=7"
python -m pip install "psutil>=5.9"
python -m pytest tests\test_system1_v1.py tests\test_demo_system1.py tests\test_system1_release_contract.py -q
```

The local benchmark command requires the non-distributed checkpoint and dataset
paths shown in `docs/SYSTEM1_DEVELOPER_GUIDE.md`; it is not reproducible from
this source-only archive. The benchmark report is withheld pending dataset
rights review. No model-weight performance claim is made by this package.
See `docs/RELEASE_RIGHTS_AUDIT.md` and `docs/AVCS_SYSTEM1_MODEL_CARD.md`.
