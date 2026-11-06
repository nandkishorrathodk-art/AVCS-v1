# ACE AI Operating System: Master Architecture Specification (v0.4.0)

> **Document Type:** System Architecture Specification & Engineering Blueprint  
> **Target System:** AVCS (Artificial Visual Cortex System) / Ace Autopilot  
> **Status:** Active Master Architecture Blueprint (v0.4.0)  
> **Canonical Path:** `layer_context.md` (alias: `leyer_context.md`)

---

## 1. Executive Summary & Core Paradigm Shift

Current industry computer-use agents (Anthropic Computer Use, OpenAI Operator, Microsoft Copilot Actions) operate on a flawed, monolithic paradigm:

$$\text{Full Screen Screenshot (2MB)} \longrightarrow \text{Upload to Cloud} \longrightarrow \text{Multi-second CoT Essay} \longrightarrow \text{Guess Coordinates} \longrightarrow \text{Blind Click}$$

This causes three catastrophic failure modes:
1. **Excessive Latency ($6,000\text{--}12,000\text{ ms}$):** Generating 250+ tokens of Chain-of-Thought (CoT) reasoning before every micro-action makes human-like computer control impossible.
2. **Blind Open-Loop Actuation:** Models emit raw $(x, y)$ coordinates without sensory verification; if a click is dropped by OS debounce or a loading spinner appears, the agent is completely blind until the next screenshot.
3. **Privacy & Bandwidth Hazards:** Continuously streaming raw $1920\times 1080$ uncompressed pixels to a remote cloud datacenter wastes bandwidth and compromises sensitive desktop data.

### The AVCS v0.4 Breakthrough: Decoupled Dual-Stream Intelligence
Inspired by **General Agents' Ace** (`ace-control-medium` and `ace-control-small`), AVCS strictly decouples high-level cognitive reasoning from low-level motor actuation. 

Instead of a single monolithic model doing everything, the system is organized into **5 specialized, interconnected layers** backed by a **Local 400K Parameter Visual Cortex** and an **Offline Learning Pipeline**.

---

## 2. The Master Architecture (v0.4.0)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 AVCS v0.4 MASTER ARCHITECTURE BLUEPRINT                                │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                        │
│  [ LAYER 1: COGNITIVE INTELLIGENCE ] (High-Level Planner @ 0.5 - 1.0 Hz)                               │
│  • Role: Multilingual Reasoning (Hindi, English, Hinglish), Goal Decomposition, Task Strategy          │
│  • Input: User prompt (e.g., "Start menu kholo aur Notepad search karo")                               │
│  • Output: Ordered Sub-Goal Directives: [1. Open Start Menu, 2. Focus Search, 3. Type Notepad]          │
│  • Invariant: NEVER predicts low-level (x, y) coordinates. Focuses purely on semantic intent.          │
│                                                   │                                                    │
│                                                   ▼                                                    │
│  [ LAYER 2: ORCHESTRATION & PERSISTENT MEMORY ]                                                        │
│  • Role: State Machine, Skill Routing, Anchor-Verified Caching, Loop Breakout                          │
│  • Anchor-Verified Skill Cache: Matches semantic goal AND verifies visual anchors before replay        │
│       ├── CACHE HIT (Anchors Verified) ──▶ Instant Deterministic Replay (0 ms Model Latency!)          │
│       └── CACHE MISS / ANCHOR MISMATCH ──▶ Route sub-goal to Perception & Control Engines              │
│                                                   │                                                    │
│                                                   ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                            PERCEPTION & COMPUTER CONTROL SUBSYSTEM                               │  │
│  │                                                                                                  │  │
│  │   [ SYSTEM 1: LOCAL VISUAL CORTEX ]                 [ LAYER 3: ACE CONTROL ENGINE ]              │  │
│  │   • 400K-parameter ConvGRU (On-device @ 160 FPS)    • Specialized VLA Policy Model               │  │
│  │   • Fast optical change detection & motion flow     • Ingests 1280x800 frame + sub-goal          │  │
│  │   • UI stability detection (`is_stable` gate)       • Predicts raw computer action (300-533 ms)  │  │
│  │   • 5-class UI morphology & affordance hotspots     • Emits action via `ActionAdapter`           │  │
│  │   • Evaluates localized click ripples (<2 ms)       • Direct Action Token generation             │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                   │                                                    │
│                                                   ▼                                                    │
│  [ LAYER 4: SAFE DESKTOP EXECUTOR ] (Win32 Biological Motor Actuator)                                  │
│  • Role: Native OS Execution, Dynamic Coordinate Calibration, Biological Kinematics                    │
│  • Dynamic Coordinate Calibration: Multi-monitor, DPI scaling (100%-200%), virtual desktop mapping     │
│  • 5th-Order Minimum-Jerk Kinematics: Calibrated flight (50-120 ms at 100 Hz, zero teleportation)       │
│  • Windows 11 XAML Protocol: 35 ms hover dwell + 65 ms button hold (experimentally calibrated)         │
│  • Hardware Keystroke Scan Codes (`MapVirtualKeyW`)                                                    │
│  • Expanded Safety Firewall: Console protection + Gating on irreversible ops (delete, pay, send)       │
│                                                   │                                                    │
│                                                   ▼                                                    │
│  [ LAYER 5: VERIFICATION & SENSORY MEMORY ] (Dual-Stage Closed-Loop)                                   │
│  • Stage 1 (Physical Verification): Visual Cortex detects optical delta / click ripple in <2 ms        │
│  • Stage 2 (Semantic Verification): Confirms intended UI state emerged (e.g. Notepad window opened)    │
│       ├── SUCCESS: Advance sub-goal; on completion, save trajectory to `skills_vault/`                 │
│       └── STUCK / FAIL: Save failure trajectory for training; send reflection feedback to L1/L2        │
│                                                   │                                                    │
└───────────────────────────────────────────────────┼────────────────────────────────────────────────────┘
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE BEHAVIORAL LEARNING & CURATION PIPELINE                                 │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  1. Trajectory Collector: Logs all live runs (both SUCCESS and FAILURE trajectories)                   │
│  2. Telemetry Curator: Curation filter separates golden demonstrations from negative failure cases     │
│  3. Model Training: Supervised Fine-Tuning (SFT) & Direct Preference Optimization (DPO) on demonstrations│
│  4. Validation Gate: Candidate checkpoints evaluated on Showdown-Clicks benchmark before deployment    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Deep Dive into the 5 Layers & Engineering Upgrades

### Layer 1: Cognitive Intelligence (High-Level Planner)
* **Invocation Frequency:** Low frequency ($0.5\text{--}1.0\text{ Hz}$). Only invoked at workflow initialization or when Layer 5 reports a verified obstacle.
* **Model Class:** Frontier Large Language Model (e.g. Nemotron 120B, Claude 3.5/3.7, GPT-4o).
* **Core Responsibilities:**
  1. Multilingual reasoning and intent extraction (Hindi, English, Hinglish).
  2. Decomposes high-level intent into ordered, atomic sub-goals (e.g. `[1: Open Start, 2: Focus Search Bar, 3: Type Notepad, 4: Launch App]`).
  3. Receives failure reflection diagnostics from Layer 5 and synthesizes alternative strategic routes.

---

### Layer 2: Orchestration & Persistent Memory
* **Core Responsibilities:**
  1. **Anchor-Verified Skill Cache (P0 Upgrade):**
     * **The Risk:** Blindly replaying cached coordinates based solely on a semantic goal hash causes misclicks if window positions, display resolutions, or app layouts differ.
     * **The Solution:** Before replaying a cached trajectory, Layer 2 checks **UI State Anchors** (e.g. active window title, target bounding box features). Only if anchors match does it replay; otherwise, it falls back to dynamic Layer 3 prediction.
  2. **Loop Breakout State Machine:** Tracks action history across a sliding window. If repetition is detected ($\ge 2$ identical actions without progress), halts execution and prompts Layer 1 for dynamic replanning.

---

### System 1 Visual Cortex vs. Layer 3 Ace Control Engine
A critical architectural principle of AVCS v0.4 is the clear separation of responsibilities between the **400K Parameter Visual Cortex** and the **Ace Control VLA Model**:

| Dimension | System 1: Visual Cortex (`avcs/cortex/`) | Layer 3: Ace Control Engine (`avcs/ace/`) |
|---|---|---|
| **Model Type** | 400,765 parameter ConvGRU Neural Network | Vision-Language-Action (VLA) Foundation Model |
| **Execution Host** | 100% Local on CPU ($2.4\text{ ms}$ tick, 160 FPS) | Local Edge GPU ($300\text{--}500\text{ ms}$) or Cloud NIM |
| **Input** | Continuous stream of raw desktop pixels | Canonical $1280 \times 800$ frame + active sub-goal |
| **Output** | Optical change, motion flow, `is_stable`, UI hotspots | Direct computer interaction tokens (click, type, key) |
| **Role** | **Sensory Reflex & Optical Retina** | **Action Prediction Brain** |

They are **complementary partners**:
* The Visual Cortex monitors screen stability; it signals `is_stable=True` before Ace Control is queried, preventing prediction on half-loaded screens.
* Ace Control predicts the action.
* The Visual Cortex immediately verifies whether a physical ripple occurred ($<2\text{ ms}$).

---

### Layer 3: Ace Control Engine & `ActionAdapter`
* **Model Sizing (Estimated):**
  - Small variants (`ace-control-small`): Inferred $\sim 1.5\text{B}\text{--}3\text{B}$ parameters.
  - Medium variants (`ace-control-medium`): Inferred $\sim 7\text{B}$ parameters.
  *(Note: Parameter counts are inferred engineering estimates based on latency profiles and hardware ceilings; General Agents has not officially published raw model specs).*
* **`ActionAdapter` Pattern:**
  Different models output different formats (Anthropic Computer Use JSON, OpenAI tool calls, General Agents API responses, or AVCS custom prefix tokens `ACT:CLICK x y`). 
  The `ActionAdapter` normalizes all incoming model outputs into the internal deterministic `Action` schema without coupling the system to any single vendor's grammar.

---

### Layer 4: Safe Desktop Executor (Win32 Biological Motor Actuator)
* **Module Path:** `avcs/ace/computer.py`
* **Dynamic Coordinate Calibration (P1 Upgrade):**
  - Dynamically queries `SM_XVIRTUALSCREEN`, `SM_YVIRTUALSCREEN`, `SM_CXVIRTUALSCREEN`, `SM_CYVIRTUALSCREEN` to handle multi-monitor desktop bounds.
  - Automatically compensates for Per-Monitor V2 DPI scaling ($100\%, 125\%, 150\%, 200\%$) across primary and secondary displays.
* **5th-Order Minimum-Jerk Kinematics:**
  - Implements Flash & Hogan (1985) biological trajectory modeling at $100\text{ Hz}$.
  - Fitts's Law duration calibrated to **$50\text{--}120\text{ ms}$** natural human flight, eliminating artificial 400ms lag.
* **Windows 11 XAML Island Timing:**
  - **$35\text{ ms}$ Hover Dwell:** Injects `WM_MOUSEMOVE` and `WM_POINTERENTER` to let modern WinUI 3 XAML containers enter the active hover state.
  - **$65\text{ ms}$ Button Hold Duration:** Guarantees down-up transitions are not filtered by DWM 60Hz quantization or switch-bounce debounce.
  *(Note: These timings are experimentally calibrated parameters for Windows 11, subject to display refresh rates).*
* **Expanded Safety Firewall (P1 Upgrade):**
  - **Console Protection:** Rejects ungrounded typing if active window is a terminal/console.
  - **Irreversible Operation Gate:** Mandatory user confirmation prompt before executing high-risk operations (permanent file deletion, payments, submitting sensitive forms, sending messages).

---

### Layer 5: Verification & Sensory Memory (Dual-Stage Closed-Loop)
* **Stage 1: Physical Verification (Optical Ripple):**
  - Evaluated in $<2\text{ ms}$ by Visual Cortex: $\Delta = |O_{\text{post}} - O_{\text{pre}}|$. Confirms physical button depression or focus ring change.
* **Stage 2: Semantic Verification (Task Success) (P0 Upgrade):**
  - Physical change alone does not guarantee task success (e.g. clicking an icon might cause a ripple, but the app may crash or take 3 seconds to launch).
  - Stage 2 verifies that the **intended semantic outcome** materialized (e.g., verifying `Notepad` process active or top-level window title present).
* **Failure Trajectory Logging (P1 Upgrade):**
  - Logs both **successful sessions** and **failed sessions** (with error traces and stuck points). Negative trajectories are essential for Direct Preference Optimization (DPO) and hard-negative mining in the offline training pipeline.

---

## 4. Benchmark Alignment & Data Provenance

### General Agents Showdown-Clicks Benchmark Data
To maintain strict empirical honesty, the benchmark figures published by General Agents are documented with exact split provenance:
* **Full Benchmark Suite:** 5,679 total human desktop clicks across diverse real-world tasks.
* **Published Development Benchmark:** **557-click held-out development subset** evaluating Correct Left-Click Predictions (95% CI) and Action Prediction Latency (ms):

```
                      SHOWDOWN-CLICKS BENCHMARK (557-Click Development Split)
Model Architecture            Click Accuracy (95% CI)    Action Latency (ms)    Speed vs Ace Small
--------------------------------------------------------------------------------------------------
ace-control-small                     72.89%                   324 ms             1.0x (Baseline)
ace-control-medium                    77.56%                   533 ms             1.6x slower
UI-TARS-72B-SFT                       52.10%                 1,977 ms             6.1x slower
Gemini 2.0 Flash                      43.50%                 3,069 ms             9.5x slower
Qwen2.5-VL-72B-Instruct               41.80%                 3,790 ms            11.7x slower
Operator (OpenAI)                     61.20%                 6,385 ms            19.7x slower
Molmo-72B-0924                        58.40%                 6,599 ms            20.4x slower
Claude 3.7 Sonnet                     55.60%                 9,656 ms            29.8x slower
OmniParser V2 + GPT-4o                48.90%                12,642 ms            39.0x slower
```

---

## 5. Offline Behavioral Learning Pipeline

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE BEHAVIORAL TRAINING PIPELINE                            │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   [ Live Desktop Execution ] ──▶ Records (Ot, At, Ot+1, Reward, Verification)          │
│                                            │                                           │
│                                            ▼                                           │
│   [ Trajectory Vault ] ───────────────────┬───────────────────────────┐                │
│                                           ▼                           ▼                │
│                                 [ Golden Demonstrations ]    [ Failure Trajectories ]  │
│                                 (Task Verified = True)       (Task Verified = False)   │
│                                           │                           │                │
│                                           ▼                           ▼                │
│                                 [ Supervised SFT ]           [ Direct Preference DPO ] │
│                                 (Imitation Learning)         (Contrastive Training)    │
│                                           │                           │                │
│                                           └───────────┬───────────────┘                │
│                                                       ▼                                │
│                                           [ Showdown-Clicks Eval Gate ]                │
│                                           (Must exceed 77.5% click acc)                │
│                                                       │                                │
│                                                       ▼                                │
│                                           [ Production Deployment ]                    │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **State-Action Trajectory Logging:** Low-level Windows hooks (`WH_MOUSE_LL`, `WH_KEYBOARD_LL`) record user and agent actions paired with desktop frames.
2. **Contrastive Curation:** Pairs successful paths with negative recovery paths for DPO training.
3. **Rigorous Validation Gate:** Candidate checkpoints must reproduce $\ge 77.5\%$ accuracy on the benchmark suite before promotion to production.

---

## 6. Implementation Roadmap for AVCS v0.4

- [x] **v0.3.0 Release:** Dual dispatch minimum-jerk kinematics, Windows 11 XAML hover/hold protocol, JPEG capture acceleration ($6\text{ ms}$), safety firewall, 85 passing tests.
- [ ] **Sprint 1 (P0 Upgrades):**
  - Implement `ActionAdapter` in `avcs/ace/actions.py` to decouple model wire protocols from internal actions.
  - Implement Semantic Verification in `avcs/ace/runner.py` (checking process/window state beyond raw pixel deltas).
  - Implement Anchor Verification for Skill Caching in `avcs/ace/session.py`.
- [ ] **Sprint 2 (P1 Upgrades):**
  - Calibrate Fitts's Law duration clamp in `avcs/ace/computer.py` to **$50\text{--}120\text{ ms}$**.
  - Add Dynamic DPI & Virtual Desktop Coordinate Calibration in `avcs/ace/computer.py`.
  - Expand Execution Safety Firewall to guard irreversible operations (file deletion, financial, outbound messages).
  - Wire negative failure trajectory serialization into `skills_vault/failures/`.
- [ ] **Sprint 3 (Visual Cortex Coupling):**
  - Integrate `StreamingVisualCortexV2` into `AceRunner` as a pre-actuation stability gate (`is_stable`) and sub-2ms post-actuation ripple verifier.
