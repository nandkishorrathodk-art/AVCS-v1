# Product Requirements Document (PRD) — AVCS (Artificial Visual Cortex System)

**Document Status:** Approved & Baseline Frozen  
**Project Owner & Sole Rights Holder:** Nandkishor Rathod (Parth Labs)  
**Target Platform:** Windows 11 / Linux (Standard 4-Core CPU, 15 GB RAM, No GPU Required)  
**Version:** 1.0.0-draft  
**Last Updated:** October 18, 2026  

---

## 1. Executive Summary & Vision

### 1.1 The Dream
Modern AI computer-use agents (Anthropic Computer Use, OpenAI Operator) interact with computers either by sending heavy full-screen screenshots to cloud multimodal APIs every 3–5 seconds, or by "cheating" through OS accessibility trees (UIAutomation / DOM inspection). This approach has three fatal flaws:
1. **High Latency & Blindness:** 3–5 second periodic screenshots cannot track fast animations, cursor movements, real-time typing, or sudden confirmation pop-ups.
2. **Fragility & Cheating:** Accessibility APIs (DOM, UIAutomation) fail completely on custom UI frameworks, canvas applications, video players, games, and Electron web views.
3. **Severe Privacy Leaks:** Uploading raw screenshots of sensitive user desktops (passwords, emails, banking, source code) to remote cloud servers violates enterprise privacy.

**AVCS (Artificial Visual Cortex System)** solves this by building an AI computer-use architecture modeled after **human biological vision and cognition**:
- A fast, continuous, local visual reflex arc that watches desktop pixels in real time (~100 FPS on CPU).
- A biological neuromuscular motor controller that moves the mouse in natural human curves.
- A cognitive reasoning layer (System 2) that understands goals and talks to the user without managing coordinate numbers.

---

## 2. The Two-Tier Biological Architecture

Human vision does not process every single rod and cone signal in the conscious prefrontal cortex. The retina and superior colliculus handle high-speed continuous motion and reflex tracking at millisecond speed, while the conscious cortex performs high-level reasoning and language processing.

AVCS mirrors this exact split:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   SYSTEM 2 — Cognitive Brain & Speech                   │
│  - Powered by: Multimodal Vision-Language Model (Gemini / Claude / VLM)│
│  - Runs: Cloud GPU or high-tier host (change-triggered, ~1-2 sec clock)│
│  - Responsibilities:                                                   │
│    • High-level reasoning, intent planning, and task decomposition     │
│    • Reading arbitrary free-form text, code, sentences, and articles   │
│    • Conversational natural-language speech with the operator          │
│    • Emitting semantic subgoals (e.g., "Click Save in the header")     │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │ Natural language subgoals
                                    │ & qualitative scene observations
                                    │ (STRICTLY ZERO COORDINATE NUMBERS)
┌───────────────────────────────────▼────────────────────────────────────┐
│                   SYSTEM 1 — Visual Cortex & Motor Hands                │
│  - Powered by: StreamingVisualCortexV1 (~501k params) + HumanEye       │
│  - Runs: Local 4-core CPU, ZERO GPU, real-time continuous clock (10ms) │
│  - Responsibilities:                                                   │
│    • Continuous 95–100 FPS screen capture with 1-slot mailbox buffer   │
│    • Spatiotemporal memory (ConvGRU) tracking motion & pop-up surprise │
│    • Spatial peak grounding & foveal 1:1 patch boundary refinement     │
│    • 131-word closed UI button vocabulary classifier                   │
│    • 5th-order minimum-jerk kinematics (smooth human-like mouse curves)│
│    • Caret blink focus detection (0.8–2.0 Hz) for typing security      │
│    • Closed-loop optical ripple verification (fresh post-action frame) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Current Implementation Status (Where We Are Today)

Over the past 20 days of engineering, **System 1 (The Eyes & Hands)** has been implemented, validated, and packaged:

### 3.1 Proven & Verified Capabilities
1. **Continuous Real-Time Screen Capture:**
   - Native Windows desktop capture via `ThreadedScreenCapture` (MSS + Win32 GDI).
   - Sustained throughput: **95.4 – 103.9 FPS** on a standard 4-core CPU at 1920×1200 resolution.
   - 1-slot atomic mailbox buffer: zero queue lag, zero stale frame buildup, zero GDI handle leaks.
2. **Spatiotemporal Neural Cortex (`StreamingVisualCortexV1`):**
   - 501,612 parameters (~2.1 MB weight file).
   - Multi-scale spatial convolutional encoder down to 16×16 feature grid.
   - Recurrent `ConvGRUCell` carrying temporal memory across observations.
   - Stage A certified: 38/38 unit tests, 20/20 challenger stress tests.
3. **Empirical Online Learning Superiority (Stage B):**
   - Pre-update evaluation protocol with Smooth L1 predictive loss and VICReg anti-collapse variance regularization ($\text{Var}(z) = 0.9673 \gg 0.05$).
   - **+70.37% prediction gain** ($\Delta_{\text{pred}} = -0.36082 < 0$) over an identical frozen reference twin model under identical live streams.
   - Bit-for-bit frozen twin stability: parameter drift strictly $0.00000000$.
4. **Biological Motor Kinematics & Verification:**
   - Flash & Hogan (1985) 5th-order minimum-jerk polynomial curves with human biological jitter (10.5 Hz micro-tremor).
   - Fail-closed caret-blink temporal frequency verification before typing.
   - `VisualRippleVerifier` verifying postconditions from fresh-frame photometric deltas.
5. **Packaged Checkpoints:**
   - Candidate weights: `runs/weights/candidates/avcs_streaming_cortex_v1.pt` (SHA-256: `418c541e97b93615eccccd3cf488efaded84cd25f0384b865e3f5022c8c6032d`).
   - 90/90 cumulative test suite passing 100% on CPU.

---

## 4. The Brutal Problems & Roadblocks (Why the Live Test Failed)

In live testing on the operator's actual 1920×1200 physical display, four critical bottlenecks were exposed:

### Roadblock 1: The Zero-OCR Reading Barrier
* **The Mandate:** Clean-room rules strictly prohibit OCR libraries (Tesseract, EasyOCR, PaddleOCR) and Windows accessibility APIs.
* **The Problem:** System 1's visual word classifier (`PixelWordRecognizer`, 239k params) only knows **exactly 131 functional UI action words** (`Save`, `Run`, `Cancel`, `Close`, `Terminal`, `Settings`, etc.).
* **The Reality:** When the operator writes arbitrary free-form text, notes, code, or prompts on the screen, **System 1 is completely blind to it**. It cannot read sentences.

### Roadblock 2: Fragile Synthetic Scene Categorization
* **The Problem:** `HumanEye`'s ventral scene category head was trained on synthetic UI templates across only 8 broad classes (`browser_webpage`, `code_editor`, `terminal`, `settings_panel`, `media_player`, `document`, `desktop_canvas`, `modal_dialog`).
* **The Reality:** When exposed to a real-world dark-mode VS Code IDE window with split editors, terminal tabs, and file trees, the model became confused: it predicted `browser_webpage` with only **50.1% confidence** and flagged 49.9% uncertainty (*"Visual layout appears ambiguous or unclassified"*).

### Roadblock 3: The "Insaan Ki Tarha Speech" Illusion
* **The Problem:** System 1 has no generative language model inside it.
* **The Reality:** The generated observations (e.g. *"Display is active. Primary context: Browser Webpage (stable)..."*) are **deterministic Python string templates** slotting in predicted hotspot counts and category labels. There is no conversational reasoning in System 1.

### Roadblock 4: The Missing System 2 Bridge
* **The Core Gap:** System 1 was never designed to be an all-in-one conversational AI. It was designed to be the ocular sensory reflex and motor hands for **System 2**. Because the System 2 bridge has not been connected to a live VLM, System 1 had no "mouth" to read text or speak with the user.

---

## 5. System Requirements & Specifications

### 5.1 System 1 (Visual Cortex & Motor Layer)
| Metric / Requirement | Target Specification | Measured Status |
| :--- | :--- | :--- |
| **Execution Hardware** | 1–4 CPU cores, 15 GB RAM, No GPU | Met (AMD64 4-core CPU) |
| **Inference Latency** | $\le 15\text{ ms}$ per tick | **9.3 – 10.2 ms** (Met) |
| **Sustained Capture FPS** | $\ge 30\text{ FPS}$ | **95.4 – 103.9 FPS** (Met, >3x target) |
| **Memory Footprint** | Bounded RAM buffer, flat RSS over 1k ticks | **Met (Flat memory line)** |
| **Prohibited Dependencies** | 0 OCR, 0 UIAutomation, 0 DOM | **Met (Clean AST audit)** |
| **Coordinate Encapsulation** | 0 coordinates exported to System 2 | **Met (Qualitative regions only)** |
| **Vocabulary Size** | 131 functional UI action words | **Met (Closed vocabulary)** |

### 5.2 System 2 (Reasoning & Language Layer) — PENDING
| Requirement | Description | Status |
| :--- | :--- | :--- |
| **VLM Integration** | Bridge to Gemini 1.5 / Claude 3.5 Sonnet / Local VLM | **Open / In Progress** |
| **Arbitrary Text Reading** | Read free-form user typing, terminal logs, code syntax | **Requires System 2** |
| **True Human Dialogue** | Conversational responses explaining desktop context | **Requires System 2** |
| **High-Level Planning** | Break high-level user tasks into System 1 subgoals | **Requires System 2** |

---

## 6. Product Roadmap & Next Milestones

### Phase 1: System 1 Hardening & Candidate Freeze (COMPLETED)
- [x] Decoupled 1-frame live screen capture engine.
- [x] Spatiotemporal ConvGRU recurrent cortex (`StreamingVisualCortexV1`).
- [x] Dual-model frozen twin online learning benchmark (+70.37% gain).
- [x] 5th-order minimum-jerk kinematics & ripple verification.
- [x] Packaging candidate checkpoints and baseline hash preservation.

### Phase 2: System 2 Cognitive Bridge (ACTIVE GOAL)
- [ ] Connect the live observation engine to System 2 VLM (Gemini / Claude / local VLM).
- [ ] Implement change-triggered foveal crop dispatch: when System 2 needs to read arbitrary text or code, System 1 extracts the high-res region of interest without sending full-resolution uncompressed video.
- [ ] Replace template speech with true natural language conversational explanations.

### Phase 3: Real-World UI Dataset & Grounding Expansion
- [ ] Expand scene classification training beyond synthetic mockups to real-world enterprise applications (VS Code, Chrome, Terminal, Windows Settings, Office).
- [ ] Expand patch word recognition from 131 words to 500+ UI tokens using clean-room synthetic renderers.
- [ ] Full end-to-end task demonstration: User instructs System 2 in natural language $\to$ System 2 plans subgoals $\to$ System 1 executes motor actions $\to$ Postconditions verified $\to$ Outcome reported.

