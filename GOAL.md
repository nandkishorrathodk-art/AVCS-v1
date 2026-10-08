# GOAL.md — Project North Star & Engineering Roadmap

## Mission: The Living Biological Eye for AI

Current AI models (GPT-4o, Claude Computer Use) operate like a **genius with a Polaroid camera**: every 2 to 4 seconds, they take a static screenshot, freeze for 2000ms while analyzing it, and output text instructions. During that lag, they are completely blind to transitions, animations, and real-time events.

**Our Goal:** Build **AVCS (Artificial Visual Cortex for Screens)** — a non-generative, biologically inspired, real-time visual sensory system that gives computers and robots **human-like continuous perception** (sub-40ms on GPU, sub-80ms on CPU).

---

## 1. Core Principles (What Makes Us Different)

1. **Non-Generative by Construction**:
   Perception does not generate text tokens. Text generation is a cognitive output; vision is a continuous spatial-temporal sensory field.
2. **Retinal Foveation & Saccades**:
   We do not downscale 4K screens into blurry thumbnails. We process a **Peripheral Stream** (global layout) and a dynamic **Foveal Stream** (high-res focus on active targets) with biological saccadic eye movements.
3. **Zero Coordinate Numbers (Hand-Eye Coordination)**:
   Humans do not calculate `x=542, y=318`. AVCS operates on **Gaze Fixation** and **Closed-Loop Visual Servoing**:
   $$\vec{\Delta} = \text{Point of Gaze} - \text{Hand/Cursor Position}$$
   Hand motion steers along the motor velocity flow vector until contact alignment is achieved.
4. **Predictive Coding (World Model)**:
   Inspired by Karl Friston's Free Energy Principle and Yann LeCun's V-JEPA. AVCS anticipates the next latent screen state $Z_{t+1}$ from $Z_t$. Unannounced popups, errors, or glitches trigger an instant **Temporal Surprise signal**.
5. **Decoupled System 1 & System 2**:
   - **System 1 (AVCS)**: Continuous real-time sensory reflex (30–60 FPS, sub-30ms).
   - **System 2 (LLM)**: Strategic planning and deep reasoning (runs asynchronously on state change).

---

## 2. Phased Roadmap

```
+-----------------------------------------------------------------------------+
| Phase 1: Architecture & Foundation (COMPLETED)                              |
| * Modular Core: Fovea, Backbone, Dorsal, Ventral, Predictive, Servoing      |
| * Developer SDK (ScreenEye) with Hand-Eye steering API                      |
| * Sub-80ms CPU benchmark verified; 13/13 unit tests passing                 |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| Phase 2: Cloud GPU Pre-Training & Real-World UI Datasets (ACTIVE)            |
| * Turnkey Kaggle Dual-T4 & Cloud GPU training runner (V-JEPA + Affordance)  |
| * Ingestion & benchmarking on real UI datasets (ScreenSpot, Mind2Web, RICO) |
| * TensorRT & ONNX sub-20ms compilation                                      |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| Phase 3: The Standalone Perception Product (SDK & API)                      |
| * PyPI Package (pip install avcs-sdk) & TypeScript SDK                      |
| * Zero-latency hooks: on_change, on_surprise, look_at, steer_hand           |
| * Drop-in adapters for Agent Frameworks (SONIC, LangChain, Playwright)      |
| * Interactive visual web dashboard (live gaze overlay & affordance heatmap) |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| Phase 4: Embodied AI & Physical Robotics                                    |
| * Porting AVCS to real-world camera streams (webcams, robotic arms, cars)   |
| * Closed-loop visual motor servoing for physical manipulation & grasp       |
+-----------------------------------------------------------------------------+
```

---

## 3. Measurable Success Metrics

| Metric | Target | Current Status |
| :--- | :--- | :--- |
| **Tick Latency (CPU)** | $\le 89\text{ ms}$ | **77.18 ms** (Achieved) |
| **Tick Latency (Cloud GPU)** | $\le 25\text{ ms}$ | **12–20 ms** (Projected / TensorRT) |
| **Perception FPS** | $\ge 25\text{ FPS}$ | **30–50 FPS** on CUDA |
| **Affordance mIoU** | $\ge 85\%$ | In training |
| **Surprise Detection Lag** | $\le 40\text{ ms}$ | Immediate on next tick |
| **Hand-Eye Alignment** | Contact acquired $\le 0.025$ radius | Verified in closed-loop servoing |

---

## 4. Guiding Discipline (Repository Rules)

- **Honest Metrics Only**: Always quote the *full System-1 tick* (foveation + neural forward + servoing), never isolated sub-steps that flatter the latency.
- **Pixels Only Mandate**: Perception must be grounded in raw pixels. No hidden accessibility trees, no DOM cheats.
- **Failures Recorded**: Never silently drop negative ablation results; keep them documented to prevent wasted re-trials.
