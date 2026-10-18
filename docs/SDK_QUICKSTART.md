# AVCS Developer SDK Quickstart Guide (`avcs-sdk`)

[![PyPI version](https://img.shields.io/badge/pypi-0.2.0-blue.svg)](https://pypi.org/project/avcs-sdk/)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-green.svg)](https://github.com/Parth-labs-in/AVCS)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE.md)

**AVCS (Artificial Visual Cortex System)** is an experimental Python research SDK and framework for screen visual perception, temporal visual modeling, and closed-loop motor experiments. It includes modular neural model architectures, in-memory desktop capture, and empirical evaluation harnesses.

> **Grounding safety boundary:** natural-language `look_at()` currently uses an
> untrained, randomly initialized grounder. It may be used for experimental
> gaze/foveal inspection only; its coordinate is not a validated semantic
> target. `ScreenEye` and `HumanEyeClient` therefore reject natural-language
> targets in motor-action methods. `ScreenEye.click_target()` and
> `type_into()` fail closed; click-on-arrival also requires an explicit
> normalized `(x, y)` coordinate and cannot default to the current fixation.
> `HumanEyeClient` likewise rejects string targets in `click`, `move_to`,
> `click_and_verify`, `type_into`, and `type_and_verify`. Explicit coordinates
> are caller-supplied motor targets, not proof that AVCS identified a semantic
> control. Both SDKs require real desktop capture by default; tests must opt
> into `SyntheticScreen` explicitly, and capture failure is not replaced with
> synthetic pixels.

---

## 1. Installation

Install the official SDK directly via pip:

```bash
pip install avcs-sdk
```

For real desktop screen grabbing on Windows, Linux, or macOS:
```bash
pip install avcs-sdk[desktop]
```

---

## 2. 10-Second Quickstart

```python
from avcs import ScreenEye

# 1. Initialize the experimental screen-perception loop (30 FPS target)
eye = ScreenEye(fps=30)
eye.start()

# 2. Check the model's current stability estimate
if eye.wait_until_stable():
    # 3. Inspect affordance candidates. These are not verified controls.
    candidates = eye.clickable_hotspots()
    print("Unvalidated clickable candidates:", candidates)

    # 4. Experimental gaze estimate only; it is not safe to use for motor input.
    target_x, target_y = eye.look_at("Submit Order")
    print(f"Unvalidated gaze estimate: ({target_x:.3f}, {target_y:.3f})")

eye.stop()
```

---

## 3. The 5 Core Developer Patterns

### Pattern 1: Experimental Language Gaze Inspection
The current natural-language gaze grounder is randomly initialized and has no
validated trained semantic grounding. `look_at()` can be inspected
experimentally, but its returned coordinate must not drive input actions:

```python
from avcs import ScreenEye

eye = ScreenEye()

# Experimental gaze/fovea estimate only; not a validated semantic target.
gaze_pos = eye.look_at("Confirm Payment")
print(f"Target normalized coordinates: {gaze_pos}")
```

Both SDKs block language-grounded motor actions until a validated grounding
model is available. For coordinate-based tests or controlled use, pass an
explicit normalized coordinate, for example
`client.click(target=(0.5, 0.5))` or
`eye.steer_hand_physical(target_gaze=(0.5, 0.5), click_on_arrival=True)`.
A supplied coordinate is not evidence that AVCS understood the target.

---

### Pattern 2: Stability Estimate (`wait_until_stable`)
`wait_until_stable` reports the current model's stability/surprise estimate.
It does not detect page-load completion or prove that a screen is safe to
interact with:

```python
from avcs import ScreenEye

eye = ScreenEye()

# Returns the model's estimate; not a page-load or task-success guarantee.
is_settled = eye.wait_until_stable(timeout=3.0)
if is_settled:
    print("Model reports a stable screen.")
else:
    print("No stable estimate before timeout.")
```

---

### Pattern 3: Coordinate-Based Servoing (`steer_hand`)
Servoing can demonstrate smooth motion to a supplied coordinate; it does not
validate that the coordinate corresponds to a semantic target. Do not feed the
untrained language-gaze estimate into motor control:

```python
from avcs import ScreenEye

eye = ScreenEye()
target = (0.5, 0.5)  # Explicit normalized coordinate, not a semantic detection.

# Current cursor position in normalized screen space
cursor = (0.2, 0.3)

while True:
    velocity, contact_acquired = eye.steer_hand(current_cursor=cursor, target_gaze=target)
    if contact_acquired:
        print("The servoing controller reports its alignment threshold was reached.")
        break
    
    # Update the simulated cursor estimate with a velocity step
    vx, vy = velocity
    cursor = (cursor[0] + vx * 0.016, cursor[1] + vy * 0.016)
```

---

### Pattern 4: Surprise-Threshold Event Hooks (`@eye.on_surprise`)
The callback reports that the current perception pipeline's surprise score
crossed a configured threshold. It does not classify the cause as a modal,
error, CAPTCHA, or other semantic event, and callback latency is not guaranteed:

```python
from avcs import ScreenEye, CortexState

eye = ScreenEye(fps=30)

# Receive a threshold signal; inspect the screen independently before acting.
@eye.on_surprise(threshold=0.45)
def handle_unexpected_event(state: CortexState):
    print(f"Surprise threshold crossed: {state.surprise_score:.3f}")

eye.start()
```

---

### Pattern 5: Affordance Candidate Maps (`clickable_hotspots`)
These methods return model-scored spatial candidates for affordance classes.
They are not verified controls or evidence that an element is clickable/typeable;
checkpoint loading and real-UI performance must be assessed separately:

```python
from avcs import ScreenEye

eye = ScreenEye()

# Get model-scored candidates: [(x, y, score), ...]
clickables = eye.clickable_hotspots()
for x, y, conf in clickables:
    print(f"Unvalidated candidate at ({x:.2f}, {y:.2f}) with score {conf:.2%}")

# Typeable-class candidates, not verified text fields
typeables = eye.typeable_hotspots()
for x, y, conf in typeables:
    print(f"Unvalidated candidate at ({x:.2f}, {y:.2f}) with score {conf:.2%}")
```

---

## 4. Context Manager (Automatic Cleanup)

`ScreenEye` fully supports Python's `with` statement:

```python
from avcs import ScreenEye

with ScreenEye(fps=25) as eye:
    state = eye.perceive_now()
    print("Active category:", state.primary_category)
# Perception engine background thread terminates cleanly upon exit
```

---

## 5. Performance Measurements

No general latency, throughput, memory, or real-UI accuracy claim is made here.
Performance depends on the selected capture source, checkpoint, hardware, and
workload. Record the exact command, environment, warm-up, sample count, and
result artifact before publishing a benchmark. Synthetic-input timing is not
evidence of live desktop task performance.

---

## 6. CLI Tools Included

The `avcs-sdk` package installs two turnkey CLI utilities:

### 1. Perception Benchmark CLI
Run the repository benchmark harness. Check its output for the selected
capture source; fixture/synthetic runs measure pipeline behavior, not
real-application task success:
```bash
avcs-benchmark
```

### 2. Visual HUD Dashboard Server
Launch the perception HUD at `http://localhost:8899`. Confirm the selected
capture source and checkpoint in the output; a running stream is not proof of
validated grounding or task completion:
```bash
avcs-hud --stream --fps 30 --port 8899
```

---

## 7. Complete API Reference

### `avcs.ScreenEye`
```python
class ScreenEye:
    def __init__(
        self,
        fps: int = 25,
        device: Optional[str] = None,
        source: Optional[object] = None,
        weights_path: Optional[str] = None,
        load_pretrained: bool = True,
    )
    def start(self) -> ScreenEye
    def stop(self) -> None
    def perceive_now(self, frame: Optional[np.ndarray] = None) -> CortexState
    def is_stable(self) -> bool
    def wait_until_stable(self, timeout: float = 4.0, poll_interval: float = 0.05) -> bool
    def look_at(self, query: str) -> Tuple[float, float]
    def locate(self, query: str) -> Tuple[float, float]
    def steer_hand(self, current_cursor: Tuple[float, float], target_gaze: Optional[Tuple[float, float]] = None) -> Tuple[Tuple[float, float], bool]
    def is_aligned(self, current_cursor: Tuple[float, float], target_gaze: Optional[Tuple[float, float]] = None) -> bool
    def clickable_hotspots(self) -> List[Tuple[float, float, float]]
    def typeable_hotspots(self) -> List[Tuple[float, float, float]]
    def on_change(self, callback: Callable[[CortexState], None]) -> Callable[[CortexState], None]
    def on_surprise(self, callback_or_threshold: Any = None, threshold: float = 0.3) -> Any
    def on_affordance(self, action_type: str = "clickable", min_confidence: float = 0.5) -> Any
```

---

## 8. License

Distributed under the **Apache-2.0 License**. See [LICENSE.md](../LICENSE.md) for details.
Built by **Parth Labs**.
