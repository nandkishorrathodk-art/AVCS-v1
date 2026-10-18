# AVCS System 1 v1.0.0-preview.1 — Developer Quickstart & Integration Guide

Welcome to the developer guide for **AVCS System 1 V1** (`AVCSSystem1V1`), a lightweight, pure-pixel neural visual perception model for desktop automation and computer-use agents.

---

## 1. Installation & Environment Requirements

AVCS System 1 is a local model. CPU inference is supported; a GPU is optional.

### Requirements
- Python 3.10–3.14
- PyTorch 2.0+
- NumPy 1.24–2.x and Pillow 9+

```bash
python -m pip install dist\avcs_system1-1.0.0rc1-py3-none-any.whl
```

The wheel declares and installs its PyTorch, NumPy, and Pillow dependencies.
The source-only distribution contains no checkpoint. `load_system1()` and the
CLI need a compatible checkpoint that you are separately authorized to use.
The local System 1 checkpoint and its inherited candidate weights are not
cleared for redistribution; see
[`RELEASE_RIGHTS_AUDIT.md`](RELEASE_RIGHTS_AUDIT.md). This preview requires no
LLM, cloud API, OCR engine, accessibility service, browser DOM, or motor
controller.

---

## 2. Quickstart: Python API

### Basic Loading & Scene Perception
Load the model and inspect a desktop screen image:

```python
import numpy as np
from PIL import Image
import avcs

# 1. Load System 1 from a checkpoint you are authorized to use.
model = avcs.load_system1(
    checkpoint_path="path/to/authorized/avcs_system1_v1.pt",
    device="cpu",
)

# 2. Load an image or screenshot
image = Image.open("path/to/desktop_screenshot.png").convert("RGB")
frame_np = np.array(image, dtype=np.uint8)

# 3. Observe the frame (unconditioned perception pass)
obs = model.observe_frame(frame_np)

print(f"Application Category : {obs.primary_category} ({obs.category_confidence * 100:.1f}%)")
print(f"Screen Resolution    : {obs.screen_dimensions}")
print(f"Detected UI Elements : {len(obs.detected_regions)}")

for region in obs.detected_regions[:5]:
    text_info = f" [Text: '{region.detected_text}']" if region.detected_text else ""
    print(f"  - {region.element_kind} @ {region.bbox_px} (conf={region.confidence:.2f}){text_info}")
```

### Query-Conditioned Target Grounding
To locate a specific UI target requested by a user or high-level agent:

```python
# Pass an active natural-language query
obs = model.observe_frame(frame_np, active_query="Save button")

print(f"Query            : '{obs.active_query}'")
print(f"Target Presence  : {obs.target_presence} (confidence: {obs.presence_confidence:.3f})")
print(f"Grounded Center  : {obs.selected_target_px} px")
print(f"Is Ambiguous     : {obs.is_ambiguous}")
print(f"Action Authorized: {obs.action_authorized}")

if obs.action_authorized:
    target_x, target_y = obs.selected_target_px
    print(f"Model eligibility point: ({target_x}, {target_y})")
    # This flag is not permission to dispatch an action.
else:
    print(f"Interaction blocked. Uncertainty flags: {obs.uncertainty_flags}")
```

---

## 3. Command-Line Interface (CLI Demo)

AVCS includes a standalone CLI tool for testing perception on images:

```bash
# Run visual grounding on an image
python -m avcs.demo_system1 --image screenshots/editor.png --query "Save button" `
  --checkpoint path\to\authorized\avcs_system1_v1.pt

# Run unconditioned perception (detect all interactive elements and text)
python -m avcs.demo_system1 --image screenshots/browser.png `
  --checkpoint path\to\authorized\avcs_system1_v1.pt

# Output structured JSON for piping into other tools
python -m avcs.demo_system1 --image screenshots/desktop.png `
  --query "Terminal icon" --checkpoint path\to\authorized\avcs_system1_v1.pt --json
```

---

## 4. Understanding VisualStateObservation

Every inference call returns a strongly-typed `VisualStateObservation` object containing:

```python
@dataclass(frozen=True)
class VisualStateObservation:
    frame_sequence: int                    # Monotonic sequence counter
    timestamp: float                       # UNIX timestamp
    screen_dimensions: Tuple[int, int]     # (Width, Height) in pixels
    coordinate_convention: str             # normalized_0_1_origin_top_left
    primary_category: str                  # 'desktop_canvas', 'editor_code', etc.
    category_confidence: float             # Softmax confidence [0.0, 1.0]
    detected_regions: List[VisualRegion]   # All proposed interactive regions
    target_presence: str                   # 'PRESENT', 'ABSENT', or 'UNCERTAIN'
    presence_confidence: float             # Calibrated presence probability
    active_query: Optional[str]            # Grounding prompt string
    selected_target_norm: Optional[Tuple]  # (cx, cy) normalized to [0, 1]
    selected_target_px: Optional[Tuple]    # (cx, cy) in absolute screen pixels
    target_candidates: List[SpatialCandidateInfo]
    target_margin: float
    is_ambiguous: bool                     # True if competing peaks lack margin
    action_authorized: bool                # Fail-closed execution gate
    temporal_change: TemporalChangeInfo    # Stability, optical flow, surprise
    uncertainty_flags: List[str]           # Reasons for abstention
```

---

## 5. Fail-Closed Safety Contracts

AVCS System 1 emits a fail-closed eligibility signal; it does not enforce the
host application's safety policy or dispatch actions:
- **`action_authorized` is `False` by default.**
- It is set to `True` **only** when all of the following conditions hold simultaneously:
  1. An `active_query` was provided.
  2. `target_presence == "PRESENT"` with `presence_confidence >= 0.70`.
  3. `is_ambiguous is False` (softmax margin between top spatial candidates is ≥ 0.20).
  4. The screen is stable (`temporal_change.stability_state in ("stable", "changed")`).
  5. There are **zero** uncertainty flags.

`action_authorized=True` means only that these model-side checks passed on this
inference. Treat it as a model signal, not authorization. A separate, current
observation, host-side safety policy, action validation, and independent
post-action verification are required before any external system acts.

---

## 6. Pure-Pixel Compliance Guarantee

System 1 enforces strict zero-cheating policies:
- **Zero OCR engines:** No Tesseract, PaddleOCR, or EasyOCR.
- **Zero Accessibility APIs:** No Windows UIAutomation, MSAA, or AT-SPI.
- **Zero DOM extraction:** No Selenium, Playwright, or web inspect.
- **Zero hardcoded coordinates:** All coordinates are generated dynamically by the neural model.
- **No external reasoning route:** System 1 inference itself returns a structured
  observation and does not call System 2 or send image data to a service.

## Reproducible evaluation and regression tests

Run the focused regression tests from the repository root:

```powershell
python -m pip install "torch>=2.0" "numpy>=1.24,<3" "Pillow>=9" "pytest>=7"
python -m pytest tests\test_system1_v1.py tests\test_demo_system1.py tests\test_system1_release_contract.py -q
```

Reproduce the benchmark only in a development checkout containing the local
checkpoint and dataset files, and only if you are authorized to use those data:

```powershell
python -m pip install "psutil>=5.9"
python -m benchmarks.benchmark_system1_v1 `
  --checkpoint runs\weights\candidates\avcs_system1_v1.pt `
  --manifest datasets\ui_grounding\manifest.json `
  --output dist\system1_v1_evaluation.json
```

The focused regression selection includes the pure-pixel AST audit in
`test_system1_v1.py` and the release API, coordinate, input-validation, and
checkpoint-integrity contracts in `test_system1_release_contract.py`.

The benchmark requires local checkpoint and dataset files that are not included
in the source-only release. The report is withheld pending the data-rights
review; no benchmark performance claim is made by this distribution. See
[`AVCS_SYSTEM1_MODEL_CARD.md`](AVCS_SYSTEM1_MODEL_CARD.md) for the exact scope,
and [`RELEASE_RIGHTS_AUDIT.md`](RELEASE_RIGHTS_AUDIT.md) for distribution
status. If a checkpoint is missing, the loader
raises `FileNotFoundError`; invalid schemas, incompatible weights, malformed
sidecars, and digest mismatches raise explicit errors rather than loading
silently.
