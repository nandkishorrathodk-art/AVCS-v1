# AVC — Artificial Visual Cortex (Neural Network)

A biomimetic visual cortex for screens and computer-use AI agents.

`avc` provides a fully trained **1.14M-parameter (1.68M with learned heads)** sensory-motor visual cortex neural network that runs on CPU at **20–50+ FPS** without requiring a GPU.

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │   SCREEN FRAME   (1920x1080x3  uint8 RGB)   │
                         └────────────────────┬────────────────────────┘
                                              │
              ┌───────────────────────────────▼───────────────────────────────┐
              │  (1) RETINAL FOVEATION  (RetinalFoveation — no weights)       │
              │     • saccade EMA alpha=0.7  • fovea covers 35% of screen     │
              │     • fixation <- motion/saliency peak (eye movement)         │
              └───────────────┬───────────────────────────┬───────────────────┘
                              │                           │
                   ┌──────────▼─────────┐      ┌──────────▼─────────┐
                   │ PERIPHERAL 256x256  │      │ FOVEAL 256x256      │
                   │ (global layout)     │      │ (high-detail)       │
                   └──────────┬─────────┘      └──────────┬─────────┘
                              │                           │
   ┌──────────────────────────▼───────────────────────────▼──────────────────────────┐
   │  (2) CORTICAL BACKBONE (CorticalBackbone) — 345,792 params (dual-path)         │
   │      Parallel MBConv InvertedResidual blocks + Squeeze-Excitation              │
   │      Fusion: Conv1x1 (256 -> 128) -> fused_feat (1, 128, 16, 16)               │
   └───────────────────────────────────┬─────────────────────────────────────────────┘
                                       │
                      fused_feat  (1, 128, 16, 16)   <-- Conscious Visual Field
                                       │
        ┌──────────────────┬───────────┴───────────┬────────────────────┬─────────────────┐
        │                  │                       │                    │                 │
        ▼                  ▼                       ▼                    ▼                 ▼
┌───────────────┐  ┌───────────────┐     ┌──────────────────┐  ┌──────────────┐  ┌────────────┐
│ (3) DORSAL    │  │ (4) VENTRAL   │     │ (5) LANGUAGE     │  │ (6) PREDICT. │  │ (7) SERVO  │
│ "WHERE / HOW" │  │ "WHAT"        │     │ MODULATION       │  │ CODING (GRU) │  │ (no wts)   │
│ 106,949 params│  │ 101,129 params│     │ 57,794 params    │  │527,368 params│  │ 0 params   │
└───────┬───────┘  └───────┬───────┘     └────────┬─────────┘  └──────┬───────┘  └─────┬──────┘
        │                  │                      │                   │                │
   (1,5,32,32)        latent(1,256)          relmap(1,1,16,16)    mem(1,256)      motor_flow
   affordances        cat(1,8)               coords(1,2)          surprise(1,1)   velocity
```

---

## Installation

```bash
pip install avc
```

For live desktop screen capture support:
```bash
pip install "avc[desktop]"
```

---

## Quickstart

### 1. High-Level Perception
```python
import avc
import numpy as np

# Load pretrained neural cortex with bundled weights
cortex = avc.load_cortex()

# Perceive any screen frame (numpy uint8 RGB or PIL Image)
frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
state = cortex.perceive(frame)

print("Primary Category :", state.primary_category)
print("Confidence       :", state.category_confidence)
print("Stability State  :", state.stability_state)
print("Surprise Score   :", state.surprise_score)
print("Affordance Map   :", state.affordance_map.shape)  # (32, 32)
print("Action Hotspots  :", state.action_hotspots)
```

### 2. Live Desktop ScreenEye
```python
import avc

with avc.ScreenEye() as eye:
    state = eye.capture_and_perceive()
    print("Active window visual state:", state.primary_category)
```

---

## Labels & Classes

- **Dorsal Affordances (5 classes)**: `static`, `clickable`, `typeable`, `scrollable`, `draggable`
- **Ventral UI Categories (8 classes)**: `desktop_canvas`, `editor_code`, `browser_webpage`, `terminal_shell`, `modal_dialog`, `form_input`, `media_playback`, `settings_panel`
- **Stability States (4 classes)**: `stable`, `animating`, `transitioning`, `surprise_event`

---

## License
Apache-2.0. Developed by Parth Labs.
