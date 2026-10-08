# MILESTONE 2: Production Developer SDK Packaging (`avcs-sdk`)

**Status:** ⚪ **PENDING** *(Locked until Milestone 1 completion)*  
**Repository:** [https://github.com/Praxis-Neural/AVCS](https://github.com/Praxis-Neural/AVCS)  
**Target Delivery:** Installable PyPI package (`pip install avcs-sdk`) with clean, typed, human-like developer APIs.

---

## 1. Objective

Package the AVCS biological perception engine into an ultra-clean, zero-friction developer library (`avcs-sdk`). Developers building computer-use agents, browser automation scripts, or UI testing suites should be able to install AVCS with a single command and interact with screen vision through simple, high-level methods (`look_at`, `wait_until_stable`, `steer_hand`, `on_surprise`).

---

## 2. Deliverables & Task Checklist

| Task ID | Item Description | Status | Evidence / Artifact |
| :--- | :--- | :--- | :--- |
| **M2.1** | **Core SDK Interface (`ScreenEye`) Implementation** | ✅ **COMPLETED** | `forge/vision/cortex/sdk.py` |
| **M2.2** | **Standalone `pyproject.toml` & Package Metadata** | ⚪ **PENDING** | `pyproject.toml` |
| **M2.3** | **Pre-Trained Weights Bundling / Remote Auto-Download** | ⚪ **PENDING** | Automatic checkpoint loader |
| **M2.4** | **100% Typed Python API (`py.typed` & Type Stubs)** | ⚪ **PENDING** | PEP 561 compliance |
| **M2.5** | **Zero-Latency Reactive Hooks (`on_surprise`, `on_change`)**| ⚪ **PENDING** | Asynchronous event dispatcher |
| **M2.6** | **Developer Documentation & Quickstart Examples** | ⚪ **PENDING** | `docs/SDK_QUICKSTART.md` |
| **M2.7** | **PyPI Packaging Verification (`build` & `twine check`)** | ⚪ **PENDING** | Wheel build verification |
| **M2.8** | **Milestone 2 Final Sign-Off** | ⚪ **PENDING** | Complete before opening Milestone 3 |

---

## 3. Developer Experience Target

```python
from avcs import ScreenEye

# 1. Initialize biological perception (30 FPS continuous stream)
eye = ScreenEye(fps=30)

# 2. Look at a target element (Language grounding without coordinates)
eye.look_at("Submit Button")

# 3. Wait until UI animations settle
eye.wait_until_stable(timeout_seconds=2.0)

# 4. Continuous hand-eye steering vector
velocity, is_aligned = eye.steer_hand(current_cursor_pos)

# 5. Reactive surprise event listener
@eye.on_surprise(threshold=0.4)
def on_alert(state):
    print("Modal or dynamic change detected!")
```

---

## 4. Acceptance Criteria
1. `pip install -e .` succeeds in a clean virtual environment without dependency conflicts.
2. `import avcs` executes in $< 100\text{ ms}$.
3. All SDK public methods pass automated unit tests.
