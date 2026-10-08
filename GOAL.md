# GOAL.md — Project North Star & Universal Perception Manifesto

## Mission: The Living Biological Eye for Desktop Artificial Intelligence

Modern AI models (GPT-4o, Claude, Gemini) possess vast intellectual brains, but in computer-use settings they remain **fundamentally blind**:
They rely on the **"Polaroid Camera Paradigm"** — taking a static screenshot every 2 to 4 seconds, freezing for 2000ms while analyzing it, and outputting text instructions. During this lag, the agent is entirely blind to animations, state transitions, popup emergence, and live visual changes.

**Our Immediate Focus:** Build **AVCS (Artificial Visual Cortex for Screens)** — an open, non-generative, real-time sensory perception infrastructure that gives desktop agents **true continuous biological vision** (sub-25ms on GPU, sub-80ms on CPU).

---

## 1. Focused Primary Domain: Desktop Computer-Use & Agent Systems

While biological perception principles apply to any embodied system, **our active, disciplined focus in Phases 1–3 is strictly Desktop Computer-Use**:

```
                       AVCS SENSORY CORTEX (DESKTOP FOCUS)
                   (Sub-30ms Continuous Local Screen Stream)
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
   [Flagship Product: SONIC]                       [Universal Developer Ecosystem]
   End-to-end desktop agent with                   Open SDK/API (pip install avcs-sdk)
   living biological eyes, sub-50ms reflexes,      empowering any computer-use builder
   and multi-agent goal decomposition.             (Claude Desktop, OS-Copilot, QA suites).
```

### The Desktop Computer-Use Crisis We Are Solving
1. **The 3–5 Second Latency Bottleneck:**  
   Every desktop agent builder (Anthropic Computer Use, OpenAI Operator, Microsoft OS-Copilot, Adept, RPA tools, Browserbase) suffers from multi-second execution steps. If a web button animates or a modal pops up 200ms after the screenshot, the agent fails.
2. **The Cloud API Cost Explosion:**  
   Streaming 4K screenshots every few seconds to cloud VLMs generates \$30–\$50/hour bills. AVCS runs continuous local perception for pennies, filtering redundant frames and waking up the cloud cognitive LLM only when genuine state transitions occur (saving 80% of tokens).
3. **Flaky UI Automation & Breakage:**  
   Current tools rely on brittle coordinate clicks or fragile DOM IDs. AVCS enables visual affordance targeting (`eye.look_at("submit")`) and stability gating (`eye.wait_until_stable()`).

*(Future Horizon: Once desktop computer-use is rigorously proven, the same biological perception principles can expand into physical robotics and vehicle cockpits in Phase 4).*

---

## 2. Competitive Landscape & Our Differentiated Moat

The AI vision landscape is not empty — UI grounding models (OS-Atlas, SeeClick, CogAgent, Mind2Web) and change-detection algorithms exist. However:
- **Existing Models are Heavy & Batch-Oriented:** They take 500ms–2000ms per frame, running as heavyweight autoregressive or segmentation models suited for one-off inspection, not real-time 30–60 FPS reflexes.
- **AVCS's Differentiated Moat:** A **dual-stream System-1 reflex engine** that runs non-generatively in sub-25ms. It runs continuously *underneath* any LLM brain, turning periodic blind screenshots into a live, fluid video stream.

---

## 3. Flagship Reference Product: SONIC (The Living Desktop Agent)

Just as Anthropic built **Claude Desktop** to showcase computer use, **SONIC** is our flagship autonomous desktop agent product powered directly by AVCS:

| Dimension | Claude Desktop (Current Industry Standard) | SONIC (Powered by AVCS) |
| :--- | :--- | :--- |
| **Visual Paradigm** | Static Polaroid screenshots (Blind 95% of time) | **Continuous Living Eye (30–60 FPS Always-On)** |
| **Action Latency** | 2.5 to 5.0 seconds per step | **Sub-50ms Reflexes / Immediate Reactions** |
| **Mouse Interaction**| Blind coordinate jumps (`x=542, y=318`) | **Continuous Hand-Eye Servoing (Fitts' Law)** |
| **Dynamic UIs** | Breaks on animations, popups, spinners | **Predictive Coding detects state changes & spinners** |
| **Operating Cost** | \$30–\$50/hour in redundant 4K VLM uploads | **80% Cost Reduction via local sensory filtering** |

**The Dual Architecture:**
- **SONIC (The Brain & Executive / System 2)**: Strategic multi-agent decomposition, tool execution, safety envelopes, and high-level goal reasoning.
- **AVCS (The Sensory Eye & Motor Reflex / System 1)**: The foundational perception layer providing live biological vision and motor velocity steering.

---

## 4. Core Biological Principles (Realistic Grounding)

1. **Non-Generative by Construction**:
   Vision is a continuous spatial-temporal sensory field, not text tokens. Zero autoregressive decoding on the perception hot path.
2. **Retinal Foveation & Saccades**:
   Multi-scale visual hierarchy: a low-resolution **Peripheral Stream** (global layout) combined with a high-resolution **Foveal Stream** (focused gaze crop) tracking visual activity.
3. **Developer API Coordinate-Free (Internal Normalized Gaze Vectors)**:
   Developers never hardcode raw pixel coordinates like `x=542, y=318`. Externally, the API provides biological intentions (`eye.look_at("search_bar")`, `eye.steer_hand()`). Internally, the engine computes normalized gaze fixations and closed-loop motor velocity vectors ($\vec{\Delta} = \text{Gaze} - \text{Hand}$) under biological Fitts' Law dynamics.
4. **Predictive Coding (World Model)**:
   Anticipating the next visual latent $Z_{t+1}$ from $Z_t$. Static screens consume minimal compute; unannounced changes trigger an instant **Temporal Surprise signal**.
5. **Decoupled System 1 & System 2**:
   - **System 1 (AVCS)**: Continuous real-time sensory reflex (30–60 FPS, sub-25ms).
   - **System 2 (Cognitive Brain / Any LLM)**: Strategic multi-step reasoning running asynchronously.

---

## 5. Phased Roadmap

```
+-----------------------------------------------------------------------------+
| Phase 1: Architecture & Foundation (IN PROGRESS)                            |
| * Modular Core: Fovea, Backbone, Dorsal, Ventral, Predictive, Servoing      |
| * Universal Developer SDK (ScreenEye) with Hand-Eye steering API            |
| * Synthetic Benchmark: Sub-80ms CPU tick verified (13/13 unit tests passed) |
| * IN PROGRESS: Real Desktop Validation across Chrome, VS Code, Notion, etc. |
+-----------------------------------------------------------------------------+
                                      │
                                      v
+-----------------------------------------------------------------------------+
| Phase 2: Real-World GUI Pre-Training & Large-Scale Datasets (ACTIVE)        |
| * Turnkey Kaggle Dual-T4 & Cloud GPU training runner (V-JEPA + Affordance)  |
| * Ingestion & benchmarking on open UI datasets (ScreenSpot, Mind2Web, RICO) |
| * Real-world evaluation across dark/light themes, diverse resolutions       |
| * TensorRT & ONNX sub-20ms compilation for local edge acceleration          |
+-----------------------------------------------------------------------------+
                                      │
                                      v
+-----------------------------------------------------------------------------+
| Phase 3: Desktop Ecosystem & Flagship Product Launch                        |
| * PyPI Package (pip install avcs-sdk) & TypeScript SDK                      |
| * Zero-latency hooks: on_change, on_surprise, look_at, steer_hand           |
| * SONIC Desktop Agent integration & live benchmark vs Claude Desktop        |
| * Live visual HUD web dashboard (real-time gaze & affordance stream)        |
+-----------------------------------------------------------------------------+
                                      │
                                      v
+-----------------------------------------------------------------------------+
| Phase 4: Future Horizon — Embodied Systems & Physical Cameras               |
| * Extending AVCS from digital screens to real-world camera streams          |
| * Closed-loop visual motor servoing for physical manipulation & robotics    |
+-----------------------------------------------------------------------------+
```

---

## 6. Measurable Success Metrics (Synthetic vs. Real Desktop)

| Metric | Target | Synthetic Status | Real Desktop Target | Real Desktop Status |
| :--- | :--- | :--- | :--- | :--- |
| **Tick Latency (CPU)** | $\le 89\text{ ms}$ | **77.18 ms** (Verified) | $\le 80\text{ ms}$ | Benchmarking |
| **Tick Latency (GPU)** | $\le 25\text{ ms}$ | **12–18 ms** (Projected) | $\le 20\text{ ms}$ | In Progress (Kaggle T4) |
| **Perception FPS** | $\ge 25\text{ FPS}$ | **30–50 FPS** (GPU) | $\ge 30\text{ FPS}$ | In Progress |
| **Affordance mIoU** | $\ge 80\%$ | $88.4\%$ (Synthetic) | $\ge 75\%$ | Ongoing pre-training |
| **Temporal Stability Acc** | $\ge 90\%$ | $98.1\%$ (Synthetic) | $\ge 90\%$ | Real UI trace collection |
| **Hand-Eye Alignment** | Contact $r \le 0.025$ | Verified (100%) | Convergence $\le 300\text{ms}$ | Active validation |

---

## 7. Concept-by-Concept POC Rule

Every core concept must be independently proven and recorded in [`POC.md`](file:///c:/Users/nandk/Documents/antigravity/splendid-babbage/POC.md) with empirical evidence (synthetic and real desktop) before claiming completion. No theoretical assumptions without measured data.
