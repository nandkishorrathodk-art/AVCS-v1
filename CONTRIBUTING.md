# Contributing to AVCS (Artificial Visual Cortex System)

Welcome to the **AVCS** open engineering effort. This project is owned and directed by **Nandkishor Rathod** (Parth Labs).

Before submitting code, please read this document carefully. This repository has strict, non-negotiable architectural mandates, clean-room contracts, and rigorous verification requirements.

---

## 1. What We Are Building & Why It Is Hard

AVCS is an **autonomous computer-use system** that perceives desktop screens from **raw RGB pixels alone** and controls the mouse and keyboard like a human, without relying on operating-system cheat codes.

Existing AI computer-use agents cheat by using:
* External OCR engines (Tesseract, EasyOCR).
* Operating-system accessibility trees (Windows UIAutomation, MSAA).
* Browser DOM inspection (Selenium, Playwright).

**Why is cheating prohibited here?**  
Because accessibility trees and DOM inspectors fail completely on games, video editors, custom graphics, canvas elements, and native Electron apps. Furthermore, uploading full uncompressed desktop screenshots to remote cloud APIs every few seconds creates severe privacy vulnerabilities and massive latency lag.

AVCS solves this using a **two-tier biological architecture**:
1. **System 1 (Local Sensory Cortex & Motor Hands):** Runs continuously on a standard 4-core CPU at 95–100 FPS with $\le 10\text{ ms}$ latency, tracking visual motion, pop-up surprises, clickable hotspots, and moving the cursor in natural 5th-order minimum-jerk curves.
2. **System 2 (Cognitive Brain & Speech):** High-level reasoning layer (VLM) that reads arbitrary text, plans tasks, and converses with the operator without managing coordinate numbers.

---

## 2. Non-Negotiable Clean-Room Rules (Strict Prohibitions)

Every pull request is automatically audited via AST and regex inspection. Violating any of these rules will result in immediate rejection:

* ⛔ **NO External OCR Libraries:** Never import `pytesseract`, `easyocr`, `paddleocr`, or Windows OCR APIs.
* ⛔ **NO Accessibility / DOM Cheating:** Never import `pywinauto`, `uiautomation`, `win32com`, `selenium`, `playwright`, or query OS accessibility trees.
* ⛔ **NO Raw Pixel Leakage:** Never export raw RGB frames, image crops, or neural embeddings across the System 2 text boundary without explicit user authorization.
* ⛔ **NO Coordinate Number Exposure:** Never emit raw screen coordinate numbers (`x, y`, bounding boxes, normalized floats) to System 2 or in natural language dialogue. Geometry remains strictly internal to AVCS motor hands.
* ⛔ **NO Hardcoded Coordinate Overrides:** Never hardcode button positions or static lookup tables for benchmarks.
* ⛔ **CPU-Only Efficiency Mandate:** All System 1 neural modules must execute on a standard 4-core CPU without requiring a GPU. Inference latency must remain $\le 15\text{ ms}$ per tick.

---

## 3. High-Priority Open Challenges (Where We Need Your Help)

If you want to contribute, these are the active engineering bottlenecks identified during live testing:

### Challenge A: Real-World IDE Scene Classification
* **Current Issue:** In live testing on a real 1920×1200 desktop, `HumanEye`'s ventral scene category head misclassified a dark-themed VS Code window as `browser_webpage` with 50.1% confidence and 49.9% uncertainty.
* **Goal:** Extend the scene classifier training beyond synthetic mockups to recognize real-world multi-pane desktop layouts: code editors (VS Code, JetBrains), terminals, file explorers, and complex multi-window desktops.

### Challenge B: Pure-Pixel Character / UI Text Patch Recognition
* **Current Issue:** System 1's `PixelWordRecognizer` has a closed vocabulary of only 131 fixed functional UI words (`Save`, `Run`, `Cancel`, etc.). It is blind to arbitrary sentences, code syntax, or free-form text typed by the user.
* **Goal:** Design and train a lightweight pure-pixel sequence recognizer (e.g., small CTC-based CNN/RNN or compact ViT under 1M parameters) capable of reading short arbitrary text patches at 1:1 physical resolution, completely without external OCR libraries.

### Challenge C: System 2 VLM Bridge Integration
* **Current Issue:** System 1's "speech" is currently generated using Python string templates formatting category and hotspot variables.
* **Goal:** Implement the official System 2 bridge connecting `SceneObservation` and change-triggered foveal crops to multimodal models (Gemini 1.5, Claude 3.5 Sonnet, or local open-weight VLMs like Qwen-2-VL) to enable true human conversational dialogue and arbitrary text reading.

### Challenge D: Multi-Monitor & Fractional DPI Coordinate Mapping
* **Current Issue:** Multi-monitor virtual desktop arrangements with negative coordinates or mixed DPI scaling (125%, 150%, 200%) require explicit affine coordinate transformation matrices.
* **Goal:** Harden `NeuromuscularServoingController` to seamlessly map internal neural heatmap peaks to physical input coordinates across complex multi-monitor setups.

---

## 4. Development & Testing Workflow

### 4.1 Setup Environment
```powershell
# Clone the repository
git clone https://github.com/Parth-labs-in/AVCS.git
cd AVCS

# Install core and desktop dependencies
python -m pip install numpy torch pillow mss pytest
```

### 4.2 Running Tests
Ensure all unit, integration, and stress tests pass before submitting code:
```powershell
# Run all core AVCS tests
python -m pytest tests/test_stage_a_live_cortex.py tests/test_stage_b_online_learning.py tests/test_stage_c_human_dialogue.py -q

# Run the complete regression suite
python -m pytest tests/ -q
```

### 4.3 Running Live Desktop Screen Inspection
To test what the model actually sees on your live desktop screen:
```powershell
python -c "from avcs.control.controller import AVCSController; c = AVCSController(); obs = c.describe_screen(); print(obs.scene_description)"
```

### 4.4 CPU Latency & Memory Profiling
To verify that your changes do not degrade latency or introduce memory leaks:
```powershell
python benchmarks/benchmark_soak_test.py
```
* Acceptance criteria: Sustained throughput $\ge 60\text{ FPS}$, P50 latency $\le 12\text{ ms}$, memory RSS flatline across 1,000 ticks.

---

## 5. Pull Request & Verification Checklist

When opening a PR, include the following in your description:
- [ ] Description of the problem solved and architecture change.
- [ ] Exact test commands executed and output pass count.
- [ ] Measured CPU latency (P50 / P95) on your local hardware.
- [ ] Confirmation that baseline checkpoints in `runs/weights/candidates/` remain untouched.
- [ ] Confirmation that zero OCR libraries or accessibility APIs were introduced.
- [ ] Evidence that no raw screenshots or user data are persisted to disk.

---

## 6. License & Code Ownership

By contributing to AVCS, you agree that your contributions are licensed under the repository's clean-room terms and belong to the project directed by **Parth Labs** / **Nandkishor Rathod**.
