# AVCS — Artificial Visual Cortex System

**An end-to-end, pure-pixel neural visual perception and motor control architecture for desktop automation.**

Developed by Parth Labs under the direction of **Nandkishor Rathod** (Project Owner & Sole Rights Holder).

---

## 1. What We Are Building (Kya Bana Rahe Hain)

AVCS is designed as a **two-tier biological vision and cognition architecture** for computer interaction:

```
┌────────────────────────────────────────────────────────┐
│             SYSTEM 2 (Cognitive Brain & Speech)         │
│  - Vision-Language Model (Gemini / Claude / Local VLM) │
│  - Reads arbitrary text, code, sentences, & articles   │
│  - High-level reasoning, task planning, & dialogue     │
└───────────────────────────▲────────────────────────────┘
                            │ Natural language subgoals
                            │ & textual observations (ZERO coordinates)
┌───────────────────────────▼────────────────────────────┐
│             SYSTEM 1 (Visual Cortex & Motor Hands)     │
│  - StreamingVisualCortexV1 (~501k parameters, CPU-only)│
│  - Real-time 95–100 FPS screen capture (1-frame buffer)│
│  - ConvGRU spatiotemporal memory & surprise detection  │
│  - 5th-order minimum-jerk kinematics (smooth mouse)    │
│  - Visual ripple postcondition verification            │
└────────────────────────────────────────────────────────┘
```

---

## 2. Model Architecture & Requirements (Kaisa Model Hona Chahiye)

1. **Pure Pixels Only (No Cheating):**
   - Strictly ZERO external OCR libraries (Tesseract, EasyOCR, PaddleOCR).
   - Strictly ZERO OS accessibility APIs (UIAutomation, MSAA) or browser DOM inspection (Selenium/Playwright).
   - Everything perceived must be derived from raw RGB display pixels.
2. **Lightweight CPU Execution:**
   - Runs on a standard 4-core CPU without requiring a GPU.
   - Forward pass execution latency $\le 10\text{ ms}$ sustaining 60–100 FPS throughput.
   - Model size: $\le 1\text{ Million}$ parameters (Candidate `StreamingVisualCortexV1` is 501,612 parameters, ~2.1 MB).
3. **Temporal Recurrence & Visual Memory:**
   - ConvGRU recurrent hidden state carrying visual context across consecutive frames.
   - Real-time optical flow and surprise alerts when unexpected dialogs or pop-ups appear.
4. **Coordinate Encapsulation:**
   - Coordinates (`x, y`, bounding boxes) remain strictly internal to AVCS motor hands.
   - System 2 and the operator receive natural descriptions of layout and elements without raw numbers.

---

## 3. Current Progress & Milestones (Kahan Tak Pahonch Chuka Hai)

* **Continuous Live Screen Capture:**
  - Real Windows 1920×1200 desktop capture running at **95.4 – 103.9 FPS** on local 4-core CPU via `ThreadedScreenCapture` (MSS + Win32 GDI).
  - 1-slot latest-value mailbox buffer: zero queue backlog, zero stale frame lag.
* **Stage A (Spatiotemporal Architecture & Permissions):**
  - Clean random weight initialization without legacy weight contamination.
  - Fail-closed tri-state permissions (`enable_observation`, `enable_online_learning`, `authorize_checkpoint_save`).
  - Passed: 38/38 unit tests, 20/20 challenger stress tests.
* **Stage B (Dual-Model Frozen Twin Learning Benchmark):**
  - Online stream learner with Smooth L1 future prediction + VICReg anti-collapse variance loss ($\text{Var}(z) = 0.9673 \gg 0.05$).
  - **Empirical Learning Superiority Proven:** $\Delta_{\text{pred}} = -0.36082 < 0$ (**+70.37% prediction improvement** over an identical frozen reference model under identical live streams).
  - Frozen reference twin drift is strictly `0.00000000` (zero contamination).
  - Passed: 90/90 cumulative unit, integration, stress, and regression tests.
* **Packaged Checkpoint:**
  - `runs/weights/candidates/avcs_streaming_cortex_v1.pt` (SHA-256: `418c541e97b93615eccccd3cf488efaded84cd25f0384b865e3f5022c8c6032d`).
  - Sidecar metadata: `runs/weights/candidates/avcs_streaming_cortex_v1.json`.
  - Baseline weights (24 checkpoints) bit-for-bit preserved and verified.

---

## 4. Honest Bottlenecks & Known Limitations (Kya Problem Hai Abhi)

Strip away all marketing and sugar-coating: here is the exact unvarnished status from live testing on a real 1920×1200 display:

1. **System 1 CANNOT Read Arbitrary Free-Form Text:**
   - System 1 has no OCR engine.
   - Its visual patch classifier (`PixelWordRecognizer`) has a closed vocabulary of **exactly 131 functional UI action words** (`Save`, `Run`, `Cancel`, `Close`, `Terminal`, `Settings`, etc.).
   - If the operator writes arbitrary text, notes, or code on screen, System 1 **cannot read it**. Full-text reading requires the System 2 VLM bridge.
2. **Coarse Scene Classification is Fragile on Real IDEs:**
   - System 1's scene classifier has 8 broad synthetic categories.
   - In live testing with VS Code active in dark mode, the model predicted `browser_webpage` with only **50.1% confidence** and flagged 49.9% uncertainty (*"Visual layout appears ambiguous or unclassified"*).
3. **"Natural Dialogue" in System 1 is Template-Driven:**
   - Observations emitted by System 1 are generated by deterministic Python string templates formatting predicted categories and hotspot counts, not by a conversational LLM.
4. **Mandatory Need for System 2:**
   - True human-level understanding, reading what the user typed, and natural conversational dialogue require connecting System 1's fast sensory stream to System 2 (Gemini / Claude / local VLM).

---

## 5. Quickstart & Verification

### Install Dependencies
```powershell
python -m pip install numpy torch pillow mss pytest
```

### Run Full Test Suite
```powershell
py -3.14 -m pytest tests/test_stage_a_live_cortex.py tests/test_stage_b_online_learning.py tests/test_stage_c_human_dialogue.py -q
```

### Run Live Desktop Screen Inspection
```powershell
py -3.14 -c "from avcs.control.controller import AVCSController; c = AVCSController(); obs = c.describe_screen(); print(obs.scene_description)"
```

---

## 6. License & Ownership

Proprietary rights, model architecture, and experimental assets belong to **Parth Labs** / **Nandkishor Rathod**.
Distributed under clean-room provenance and authorization rules.
