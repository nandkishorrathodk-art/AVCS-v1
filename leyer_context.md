# ACE AI Operating System: 5-Layer Master Architecture

> **Document Type:** System Architecture Specification & Context Blueprint  
> **Target System:** AVCS (Artificial Visual Cortex System) / Ace Autopilot  
> **Status:** Active Reference Blueprint (v0.3.0+)  
> **Canonical Path:** `layer_context.md`

---

## 1. Executive Summary & Core Paradigm Shift

Current industry computer-use agents (Anthropic Computer Use, OpenAI Operator, Microsoft Copilot Actions) operate on a flawed, monolithic paradigm:

$$\text{Full Screen Screenshot (2MB)} \longrightarrow \text{Upload to Cloud} \longrightarrow \text{Multi-second CoT Essay} \longrightarrow \text{Guess Coordinates} \longrightarrow \text{Blind Click}$$

This causes three catastrophic failure modes:
1. **Excessive Latency ($6,000\text{--}12,000\text{ ms}$):** Generating 250+ tokens of Chain-of-Thought reasoning before every micro-action makes human-like computer control impossible.
2. **Blind Open-Loop Actuation:** Models emit raw $(x, y)$ coordinates without sensory verification; if a click is dropped by OS debounce or a loading spinner appears, the agent is completely blind until the next screenshot.
3. **Privacy & Bandwidth Hazards:** Continuously streaming raw $1920\times 1080$ uncompressed pixels to a remote cloud datacenter wastes bandwidth and compromises sensitive desktop data.

### The General Agents Ace & AVCS Breakthrough
Inspired by **General Agents' `ace-control-medium` (533 ms / 77.56% click accuracy)** and **`ace-control-small` (324 ms / 72.89% click accuracy)**, AVCS decouples high-level cognitive reasoning from low-level motor actuation. 

Instead of a single monolithic model doing everything, the system is organized into **5 specialized, interconnected layers**.

---

## 2. The 5-Layer Master Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  ACE AI OPERATING SYSTEM ARCHITECTURE                                  │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                        │
│  [ LAYER 1: MAIN REASONING INTELLIGENCE ] (Cognitive Planner @ 0.5 - 1.0 Hz)                          │
│  • Role: High-Level Language Understanding (Hindi / English), Goal Decomposition, Strategy            │
│  • Input: User natural language prompt (e.g., "Start menu kholo aur Notepad search karo")              │
│  • Output: Sub-goal Plan Sequence: [Sub-Goal 1: Activate Start Menu, Sub-Goal 2: Type Notepad]         │
│                                                   │                                                    │
│                                                   ▼                                                    │
│  [ LAYER 2: TASK ORCHESTRATOR & SKILL CACHE ]                                                          │
│  • Role: Dynamic Routing, Session Management, Persistent Skill Replay                                 │
│  • Cache Check: Is this goal/sub-goal already recorded in `skills_vault/`?                             │
│       ├── CACHE HIT  ──▶ Instant Deterministic Replay (0 ms Model Latency!)                            │
│       └── CACHE MISS ──▶ Route sub-goal directive to Layer 3 Computer Control Engine                   │
│                                                   │                                                    │
│                                                   ▼                                                    │
│  [ LAYER 3: SPECIALIZED COMPUTER CONTROL ENGINE ] (VLA Model @ 300 - 533 ms)                          │
│  • Role: Direct Vision-Language-Action (VLA) Policy                                                    │
│  • Input: Canonical 1280x800 Desktop Frame (JPEG in-memory, <8ms) + Active Sub-Goal                     │
│  • Output: Direct Action Tokens (<= 5 tokens, zero CoT text bloat)                                     │
│            Format: `ACT:CLICK 292 768` | `ACT:TYPE Notepad` | `ACT:KEY win`                             │
│                                                   │                                                    │
│                                                   ▼                                                    │
│  [ LAYER 4: DESKTOP EXECUTOR ] (Win32 Biological Motor Actuator)                                      │
│  • Role: Neuromorphic Motor Physics & Native OS Protocol Execution                                   │
│  • 5th-Order Minimum-Jerk Kinematics (Flash & Hogan @ 100 Hz, 50-120 ms natural flight)                │
│  • Dual Dispatch: `SetCursorPos` + `SendInput(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)`                │
│  • Windows 11 XAML Protocol: 35 ms hover dwell + 65 ms button hold with normalized [0, 65535] coords   │
│  • Hardware Keystroke Scan Codes (`MapVirtualKeyW`) + Execution Safety Firewall                        │
│                                                   │                                                    │
│                                                   ▼                                                    │
│  [ LAYER 5: VERIFICATION & SENSORY MEMORY ] (Closed-Loop Reflex)                                       │
│  • Sub-2ms Optical Delta / Click Ripple Probe (`HandMotor.verify_ripple`)                              │
│  • Verification: Did target UI element physically change / open?                                       │
│       ├── SUCCESS: Advance to next sub-goal; on task complete, serialize trajectory to `skills_vault/` │
│       └── STUCK / FAIL: Emit honest reflection feedback to Layer 1 for dynamic replanning              │
│                                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Deep Dive into the 5 Layers

### Layer 1: Main Reasoning Intelligence (Cognitive Planner)
* **Frequency:** Low frequency ($0.5\text{--}1.0\text{ Hz}$). Only invoked when initiating a task, decomposing a goal, or replanning after a verified obstacle.
* **Model Class:** Frontier Large Language Model (e.g. Claude 3.5/3.7, Nemotron 120B, GPT-4o).
* **Core Responsibilities:**
  1. Parse multilingual and dialect user prompts (Hindi, English, Hinglish: *"Chrome open karke OpenAI website visit karo"*).
  2. Break complex multi-app workflows into atomic sub-goals.
  3. Maintain global task objective and detect high-level completion criteria.
* **Key Invariant:** Layer 1 **NEVER** predicts low-level $(x, y)$ coordinates. It outputs semantic directives only.

---

### Layer 2: Task Orchestrator & Skill Cache
* **Module Path:** `avcs/ace/orchestrator.py` & `avcs/ace/skills_vault/`
* **Core Responsibilities:**
  1. **Persistent Skill Caching (0 ms Replay):** When a user requests a common task (e.g., "Open Notepad", "Launch VS Code", "Open Chrome"), Layer 2 computes a semantic hash of the goal. If a verified trajectory exists in `skills_vault/`, it immediately replays the cached motor trajectory without any model inference.
  2. **Sub-Goal State Machine:** Manages step budgets, timeout gates, and transitions between atomic sub-goals.
  3. **Loop Detection & Recovery:** Detects repetitive patterns ($\ge 2$ identical actions) and triggers breakout reflection rather than executing dumb infinite clicks.

---

### Layer 3: Specialized Computer Control Engine (Ace-Control VLA)
* **Model Class:** Compact Vision-Language-Action Policy Network ($1.5\text{B}\text{--}7\text{B}$ parameters, e.g. `ace-control-small`, `ace-control-medium`, `Qwen2.5-VL-3B/7B`, `SmolVLM2-2.2B`).
* **Latency Target:** $300\text{ ms} \le T_{\text{infer}} \le 533\text{ ms}$.
* **Core Responsibilities:**
  1. Ingests canonical $1280 \times 800$ desktop frame (delivered from memory via `ScreenStreamer` in $<0.1\text{ ms}$).
  2. Emits **ultra-compact action tokens** ($\le 5$ tokens total, $<40\text{ ms}$ decoding time) without verbose JSON or Chain-of-Thought essays:
     ```text
     ACT:CLICK 292 768
     ACT:TYPE Notepad
     ACT:KEY win
     ACT:STOP Task completed
     ```
  3. Uses fixed visual token budgets ($256\text{--}576$ visual tokens) to ensure prefill completes in under $200\text{ ms}$.

---

### Layer 4: Desktop Executor (Biological Win32 Motor Actuator)
* **Module Path:** `avcs/ace/computer.py` (606 lines)
* **Core Responsibilities:**
  1. **5th-Order Minimum-Jerk Kinematics:** Implements biological neuromuscular trajectory modeling (Flash & Hogan, 1985):
     $$\tau = \frac{t}{T}, \quad s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$$
     Scaled via Fitts's Law to **$50\text{--}120\text{ ms}$ duration at $100\text{ Hz}$**, eliminating unnatural cursor teleportation.
  2. **Dual Dispatch:** Injects simultaneous `SetCursorPos(cur_x, cur_y)` and `SendInput(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK)` with normalized coordinates in $[0, 65535]$.
  3. **Windows 11 XAML Island Protocol:**
     * **$35\text{ ms}$ Hover Dwell:** Ensures WinUI 3 XAML Islands (`DesktopWindowXamlSource`) receive `WM_MOUSEMOVE` and `WM_POINTERENTER` to enter hover state.
     * **$65\text{ ms}$ Button Hold Duration:** Prevents DWM 60Hz frame quantization and touch switch-bounce debounce filters from discarding clicks.
  4. **Hardware Scan Code Injection:** Maps virtual keys via `MapVirtualKeyW` to ensure modern Windows applications (Windows Terminal, Windows Search) receive valid keystrokes.
  5. **Execution Safety Firewall:** Intercepts ungrounded typing when a console window (`ConsoleWindowClass`, `CASCADIA`, `WindowsTerminal`) is foreground, preventing terminal text corruption.
  6. **Hardware Failsafe:** `emergency_release_hardware()` releases all held buttons and modifier keys if interrupted (`Ctrl+C` or 4-corner mouse push).

---

### Layer 5: Verification & Closed-Loop Sensory Memory
* **Latency Target:** $<2\text{ ms}$ local check.
* **Core Responsibilities:**
  1. **Optical Delta & Click Ripple Verification:** Immediately after button release, evaluates localized pixel difference:
     $$\Delta = |O_{\text{post}} - O_{\text{pre}}|$$
     Confirms that the UI element visually reacted (e.g., button depressed, menu opened, cursor focused).
  2. **Behavioral Trajectory Recording (OpenAI VPT Style):**
     * When a workflow completes successfully, records the full sequence $(O_0, A_0, O_1, A_1, \dots, O_N, \text{STOP})$ into `skills_vault/trajectories/`.
     * Curates these demonstrations to continuously build offline training datasets for local model fine-tuning.
  3. **Fail-Closed Reflection:** If a click fails to trigger visual response, passes honest reflection feedback to Layer 1 rather than forging a successful state.

---

## 4. Benchmark Alignment: General Agents' Showdown-Clicks

General Agents evaluated `ace-control-medium` and `ace-control-small` across 5,679 held-out expert desktop clicks:

| Model Architecture | Parameter Scale | Action Latency | Click Accuracy (95% CI) | Operational Mode |
|---|---|---|---|---|
| **ace-control-small** | $\sim 1.5\text{B}\text{--}3\text{B}$ | **324 ms** | **72.89%** | Direct VLA Action Tokens |
| **ace-control-medium** | $\sim 7\text{B}$ | **533 ms** | **77.56%** | Direct VLA Action Tokens |
| **UI-TARS-72B-SFT** | 72B | 1,977 ms | 52.10% | Supervised Fine-Tuned LLM |
| **Gemini 2.0 Flash** | Frontier | 3,069 ms | 43.50% | Generalist Multimodal API |
| **Qwen2.5-VL-72B-Instruct** | 72B | 3,790 ms | 41.80% | Generalist Multimodal API |
| **OpenAI Operator** | Frontier | 6,385 ms | 61.20% | Multi-step CoT Agent |
| **Claude 3.7 Sonnet** | Frontier | 9,656 ms | 55.60% | Multi-step CoT Agent |
| **OmniParser V2 + GPT-4o** | Hybrid Pipeline | 12,642 ms | 48.90% | OCR + BBox + LLM |

### The Power-Law Scaling Law
General Agents proved that computer control accuracy scales monotonically with behavioral expert training data (1x, 2x, 4x, 8x, 16x log scale). Fine-tuning compact models on clean state-action demonstrations beats 72B+ conversational models by over **20% absolute accuracy** at **10x–30x lower latency**.

---

## 5. Codebase Mapping: Where Each Layer Lives in AVCS

```
AVCS-v1/
├── avcs/
│   ├── ace/
│   │   ├── __init__.py        # Official SDK entrypoint (Agent, Computer, Action)
│   │   ├── actions.py         # Layer 3 Action definitions, grammar, alias normalization
│   │   ├── agent.py           # Layer 3 Agent coordinator & canonical canvas validation
│   │   ├── client.py          # Layer 1/3 Multi-protocol client & fast grammar parser
│   │   ├── computer.py        # Layer 4 Biological Win32 Motor Actuator (Fitts's Law, XAML)
│   │   ├── runner.py          # Layer 2 Execution orchestrator, safety firewall, closed loop
│   │   ├── session.py         # Layer 2/5 Multi-turn session tracking & repetition detector
│   │   ├── streamer.py        # Layer 3 Continuous 30 FPS background desktop grabber (JPEG)
│   │   └── run.py             # CLI entrypoint (`python -m avcs.ace.run`)
│   ├── cortex/                # System 1 Neuromorphic Visual Cortex (400k parameter ConvGRU)
│   └── control/               # Neuromuscular servoing and capability management
├── layer_context.md           # This Master Architecture Document
├── pyproject.toml             # v0.3.0 configuration with `avcs-ace` CLI script
└── tests/
    └── test_ace_engine.py     # 85 automated unit & regression tests (100% passing)
```

---

## 6. Implementation Checklist for Full Ace Parity

- [x] **v0.3.0 Release:** Win32 minimum-jerk kinematics, dual dispatch, Windows 11 XAML hover dwell ($35\text{ ms}$), hold duration ($65\text{ ms}$), JPEG capture acceleration ($6\text{ ms}$), 85 tests passing.
- [ ] **Pillar 1:** Implement Ultra-Compact Action Grammar (`ACT:CLICK x y`, `ACT:TYPE text`, `ACT:KEY win`) in `avcs/ace/client.py` to eliminate JSON decode overhead.
- [ ] **Pillar 2:** Re-calibrate Fitts's Law duration clamp in `avcs/ace/computer.py` to **$50\text{--}120\text{ ms}$** (matching PRD SLA).
- [ ] **Pillar 3:** Wire `skills_vault/` SQLite/JSON persistent skill cache into `avcs/ace/runner.py` for 0 ms replay of verified tasks.
- [ ] **Pillar 4:** Add local vLLM / TensorRT-LLM support for local 3B/7B VLA models (`http://127.0.0.1:8000/v1`) to achieve native sub-500ms end-to-end execution.
- [ ] **Pillar 5:** Create `benchmarks/run_showdown_benchmark.py` evaluating top-1 click accuracy directly against `generalagents/showdown-clicks`.
