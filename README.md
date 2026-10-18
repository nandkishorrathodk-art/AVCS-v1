# AVCS — Artificial Visual Cortex System

**An end-to-end, pure-pixel neural visual perception and motor control architecture for desktop automation.**

Developed by Parth Labs under the direction of **Nandkishor Rathod** (Project Owner & Sole Rights Holder).

---

### Core Documentation Links
* 📄 **[Product Requirements Document (PRD.md)](PRD.md)** — Comprehensive architecture, biological vision design, and functional specs.
* 🤝 **[Contribution Guide (CONTRIBUTING.md)](CONTRIBUTING.md)** — Clean-room rules, open challenges, test instructions, and PR guidelines.
* 🎯 **[Goals & Brutal Reality (GOAL.md)](GOAL.md)** — Unvarnished engineering reality, live test findings, and hardware limits.
* 📊 **[Project Status & Milestones (PROJECT.md)](PROJECT.md)** — Feature inventory, test results, and milestone ledger.

---

## 1. What This Model Actually Is (Technical Identity & Specs)

Strip away all marketing, hype, and academic sugar-coating. Here is the exact technical reality of what this model is:

* **Model Name:** `StreamingVisualCortexV1` + `HumanEye` Subsystem (AVCS System 1).
* **Parameter Count:** **501,612 parameters (~0.5 Million / ~2.1 MB weight file)**.
  *(For comparison: Claude 3.5 Sonnet or GPT-4 are 100+ Billion parameters; small local LLMs like Llama-3 8B are 8,000,000,000 parameters. This model is 0.006% the size of an 8B LLM).*
* **Execution Hardware:** Runs on a standard **4-core AMD64 CPU, 15 GB RAM, ZERO GPU**.
* **Latency & Speed:** **9.3 – 10.2 ms CPU forward pass**, sustaining **95.4 – 103.9 FPS** continuous live capture and inference.
* **Internal Neural Components:**
  1. **Multi-Scale Spatial Encoder:** 2D Convolutions (`scale_half`, `scale_quarter`, `scale_eighth`) downsampling raw desktop pixels into a 16×16 feature grid with 48 channels.
  2. **Recurrent Spatiotemporal Memory (`ConvGRUCell`):** Gated recurrent neural memory retaining visual context across consecutive frames to track motion and state changes.
  3. **Predictive Latent Heads:** 64-dimensional latent projection head predicting future frame visual representations.
  4. **Dense Change & Motion Heads:** Spatial change probability head and 2D grid displacement head detecting visual surprise.
  5. **Patch Word Recognizer (`PixelWordRecognizer`):** 239,000-parameter ConvNet classifying small UI button crops over a **closed vocabulary of 131 functional UI words**.
  6. **Neuromuscular Servoing Controller:** Flash & Hogan (1985) 5th-order minimum-jerk kinematics generator ($10\tau^3 - 15\tau^4 + 6\tau^5$) with 10.5 Hz micro-tremor biological jitter.
  7. **Visual Ripple Verifier:** Postcondition verifier capturing fresh post-action frames to verify photometric/optical changes.

---

## 2. What Problems Does This Model Actually Solve? (Kis Kis Problem Ko Solve Karta Hai)

This model was engineered to solve 6 specific, fatal bottlenecks that plague existing computer-use AI agents:

### Problem 1: The 3–5 Second Screenshot Lag
* **The Industry Flaw:** Tools like Anthropic Computer Use and OpenAI Operator grab a full screenshot every 3 to 5 seconds. In that multi-second gap, the AI is completely blind: it cannot see cursor movements, hover tooltips, smooth scroll animations, typing progress, or sudden confirmation pop-ups.
* **How AVCS Solves It:** AVCS captures and processes desktop screen pixels continuously at **95.4 – 103.9 FPS (~10ms per tick)** on a standard 4-core CPU. Its 1-slot latest-value mailbox buffer drops stale backlog frames so the model always perceives the exact present instant with zero latency lag.

### Problem 2: The "Cheating" Trap (UIAutomation / DOM Scraping)
* **The Industry Flaw:** Most automation bots cheat by reading OS accessibility trees (Windows UIAutomation, MSAA) or web page DOMs (Selenium, Playwright). When an application uses custom GPU rendering, HTML5 Canvas, Electron without accessibility tags, video players, or games, these tools break completely.
* **How AVCS Solves It:** AVCS operates on **pure RGB display pixels alone**. It has zero dependencies on Windows accessibility APIs, zero DOM inspection, and zero accessibility hooks. If a pixel is on the screen, AVCS can perceive it regardless of what framework rendered it.

### Problem 3: Privacy Leaks & Massive Cloud Video Bandwidth
* **The Industry Flaw:** Uploading 1080p desktop video or uncompressed screenshots to cloud vision APIs consumes gigabytes of bandwidth and exposes confidential user data (passwords, emails, bank accounts, proprietary source code) to third-party cloud servers.
* **How AVCS Solves It:** System 1 runs **100% locally on CPU in bounded RAM buffers**. Zero raw frames, zero screenshots, and zero video recordings are ever written to disk or uploaded to the cloud. Only high-level qualitative text observations (with zero coordinate numbers) leave the local machine.

### Problem 4: Robotic "Bot-Like" Cursor Movement
* **The Industry Flaw:** Automated bots move cursors in dead-straight lines or teleport instantaneously between coordinates (`x=100, y=200` to `x=500, y=600`). Anti-bot systems and OS heuristics flag this immediate non-human behavior.
* **How AVCS Solves It:** AVCS implements biological **5th-order minimum-jerk polynomial trajectory generation** (Flash & Hogan, 1985), producing natural bell-shaped velocity curves with zero start/end acceleration and an optional 10.5 Hz biological micro-tremor resonator that mimics human muscle jitter.

### Problem 5: Blind "Fire-and-Forget" Clicking
* **The Industry Flaw:** Ordinary bots click a coordinate and immediately assume the click worked. If the window was lagged, unresponsive, or an unexpected modal dialog blocked the button, the bot proceeds blindly and breaks the workflow.
* **How AVCS Solves It:** AVCS enforces **closed-loop verification (`VisualRippleVerifier`)**. After every physical mouse click, it grabs a fresh post-action frame and measures the photometric delta and optical ripple energy at the target site. It marks the action successful only if an actual visual state transition occurred.

### Problem 6: Online Representation Collapse in Continuous Streams
* **The Industry Flaw:** Continuous self-supervised streaming models frequently collapse into predicting constant or all-zero latent embeddings when observing static or repetitive screens.
* **How AVCS Solves It:** AVCS implements **VICReg anti-collapse variance regularization** ($\text{Var}(z) \ge 0.05$) combined with Smooth L1 future prediction loss. In empirical testing against an identical frozen reference twin model under identical live streams, the online updating model achieved **+70.37% prediction gain** ($\Delta_{\text{pred}} = -0.36082 < 0$) while maintaining robust latent variance ($\text{Var}(z) = 0.9673$).

---

## 3. Where Does This Model Fail? (Model Kaha Fail Ho Raha Hai — Zero Sugar-Coating)

In live testing on the operator's actual 1920×1200 physical display, the model's hard limits and failure modes were exposed directly:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        THE 4 BRUTAL FAILURES                           │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Zero Arbitrary Text Reading (Cannot read typed text, code, or docs) │
│ 2. Fragile Scene Classification (Mistook dark-mode VS Code for Browser)│
│ 3. Fake "Speech" (Hardcoded Python string templates, NOT an LLM)       │
│ 4. Zero Cognitive Planning (Cannot reason or break down tasks alone)   │
└────────────────────────────────────────────────────────────────────────┘
```

### Failure 1: Complete Blindness to Arbitrary Text (The Zero-OCR Reality)
* **The Reality:** **This model CANNOT read arbitrary text typed by the user, cannot read paragraphs, cannot read terminal outputs, and cannot read source code lines.**
* **Why it fails:** Clean-room rules strictly ban OCR libraries (Tesseract, EasyOCR, PaddleOCR). System 1's visual patch classifier (`PixelWordRecognizer`) has a closed vocabulary of **only 131 fixed UI button words** (`Save`, `Run`, `Cancel`, `Close`, `Terminal`, `Settings`, `Browse`, `Files`). If you type a sentence, write a note, or open an unmapped button, the model is completely blind to the text content. Full text reading strictly requires a **System 2 VLM**.

### Failure 2: Fragile Scene Classification on Real-World IDEs
* **The Reality:** When tested on the operator's live 1920×1200 desktop running **Visual Studio Code** in dark mode, the model predicted `browser_webpage` with only **50.1% confidence** and flagged 49.9% uncertainty (*"Visual layout appears ambiguous or unclassified"*).
* **Why it fails:** `HumanEye`'s ventral scene category head was trained on synthetic UI templates across only 8 broad classes (`browser_webpage`, `code_editor`, `terminal`, `settings_panel`, `media_player`, `document`, `desktop_canvas`, `modal_dialog`). Real-world IDEs with split panes, dark backgrounds, terminal tabs, and file trees look ambiguous to its small 16×16 spatial feature grid.

### Failure 3: The "Insaan Ki Tarha Speech" Illusion
* **The Reality:** **System 1 does NOT possess conversational intelligence and does not speak like an AI.**
* **Why it fails:** There is NO language model inside System 1. The emitted natural language text:
  > *"Display is active. Primary context: Browser Webpage (stable). Visible interactive controls: 2 clickables in center, 1 clickable in footer... Visual layout appears ambiguous or unclassified."*  
  is generated by **deterministic Python string templates** in `avcs/observation.py` that simply slot detected category names and hotspot counts into static sentences.

### Failure 4: Zero Cognitive Reasoning & Task Planning
* **The Reality:** If you tell System 1 *"Open Chrome and search for news"*, it cannot do it.
* **Why it fails:** A 500k-parameter neural network on CPU is an **ocular reflex arc and motor cerebellum** (like a human retina, visual cortex, and motor nerves). It is NOT a prefrontal cortex. It has no conceptual understanding of goals, sequential logic, or user intent. It requires **System 2 (an LLM/VLM)** to tell it what subgoal to execute.

---

## 4. The Two-Tier Architecture: Why System 1 and System 2 Are Both Needed

To achieve real human-like computer use, AVCS divides responsibilities strictly between two systems:

```
┌────────────────────────────────────────────────────────┐
│             SYSTEM 2 (Cognitive Brain & Speech)         │
│  - Powered by: Multimodal VLM (Gemini / Claude / VLM)  │
│  - Clock: Change-triggered (~1-2 seconds)              │
│  - Reads: Arbitrary text, code, sentences, & articles  │
│  - Speaks: True conversational natural dialogue        │
│  - Plans: High-level task decomposition & subgoals     │
└───────────────────────────▲────────────────────────────┘
                            │ Natural language subgoals
                            │ & qualitative observations (ZERO coordinates)
┌───────────────────────────▼────────────────────────────┐
│             SYSTEM 1 (Visual Cortex & Motor Hands)     │
│  - Powered by: StreamingVisualCortexV1 (~501k params)  │
│  - Clock: Continuous real-time (95–100 FPS, 10ms)      │
│  - Hardware: Local 4-core CPU, No GPU, Bounded RAM     │
│  - Perceives: Motion, diffs, pop-up surprises, hotspots│
│  - Acts: Smooth 5th-order minimum-jerk mouse curves    │
│  - Verifies: Optical ripples & frame transitions       │
└────────────────────────────────────────────────────────┘
```

### Responsibility Comparison

| Capability | System 1 (This Local Model) | System 2 (External VLM / LLM) |
| :--- | :--- | :--- |
| **Model Size** | 501,612 params (~2.1 MB) | 10B – 100B+ params |
| **Hardware** | Standard 4-core CPU (<10ms) | Cloud GPU / High-memory accelerator |
| **Real-Time Stream** | 95–104 FPS continuous capture | Change-triggered (periodic on demand) |
| **Visual Awareness** | Pure pixels, motion, surprise, hotspots | High-level semantic scene understanding |
| **Text Reading** | 131 fixed UI button words only | Arbitrary free-form text, code, sentences |
| **Speech / Conversation** | Formatted Python string templates | Genuine natural language reasoning |
| **Motor Execution** | Physical mouse curves & keystrokes | None (never manages hardware or coordinates) |
| **Verification** | Frame diffs & optical ripple energy | Task completion & logical postconditions |

---

## 5. Quickstart & Verification

### Install Dependencies
```powershell
python -m pip install numpy torch pillow mss pytest
```

### Run Core Stage Tests (90/90 Tests Pass)
```powershell
py -3.14 -m pytest tests/test_stage_a_live_cortex.py tests/test_stage_b_online_learning.py tests/test_stage_c_human_dialogue.py -q
```

### Run Live Desktop Screen Inspection
To see what the model actually perceives on your live desktop screen right now:
```powershell
py -3.14 -c "from avcs.control.controller import AVCSController; c = AVCSController(); obs = c.describe_screen(); print('DESCRIPTION:', obs.scene_description); print('CATEGORY:', obs.primary_category); print('CONFIDENCE:', obs.confidence)"
```

### Run 1,000-Tick CPU Soak Benchmark
```powershell
py -3.14 benchmarks/benchmark_soak_test.py
```

---

## 6. License & Ownership

All rights, architecture designs, trained candidate checkpoints, and code are the proprietary intellectual property of **Parth Labs**, directed by **Nandkishor Rathod** (Project Owner & Sole Rights Holder).
Distributed under clean-room provenance and non-cheating verification rules.
