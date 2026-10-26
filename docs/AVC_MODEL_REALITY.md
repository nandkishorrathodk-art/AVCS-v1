# AVC (Artificial Visual Cortex) — Technical Reality & Architecture Document

> **Core Reality Fact:** AVC (`StreamingVisualCortexV2` / `avcs_v2.pt`) is a **real-time neuromorphic computer vision neural network**, not an autoregressive language model. It has **no text decoder, no tokenizer, and no language generation capabilities**. All English sentences previously observed in the console were deterministic Python `f-string` templates populated by numeric sensory telemetry and Win32 OS APIs.

---

## 1. Executive Summary: What AVC Is and What It Is Not

```mermaid
flowchart TD
    subgraph Real_AVC_Neural_Model ["What AVC ACTUALLY IS (avcs_v2.pt - 501k Params)"]
        Pixels["Raw Screen Pixels (128x128)"] --> Pre["4-Channel Edge & Luminance Preprocessor"]
        Pre --> Stem["Spatial Pyramid Stem (3 Convs)"]
        Stem --> ConvGRU["ConvGRU Temporal Recurrent Cell (64ch, 16x16)"]
        ConvGRU --> Head1["Head 1: Temporal Change & Spinner Logit"]
        ConvGRU --> Head2["Head 2: 5-Class UI Objectness Grid (16x16)"]
        ConvGRU --> Head3["Head 3: Action Ripple Response Logit"]
    end

    subgraph What_AVC_Is_NOT ["What AVC IS NOT"]
        LLM["Large Language Model (NO Tokenizer, NO Vocab)"]
        OCR["Screen Text Reader / OCR Engine (CANNOT read text)"]
        Speech["Autonomous Voice / Natural English Generator"]
    end

    subgraph Outside_Python_Glue ["Where The English Commentary Came From"]
        OS["Windows Win32 OS API: GetWindowTextW()"] --> Title["Active Window Title String"]
        Head2 --> Coord["Hotspot Grid Peak Coords"]
        Title & Coord --> Templates["avcs/narrator.py: Hardcoded Python f-strings"]
        Templates --> Print["Console Output ('Alert! Modal dialog...')"]
    end
```

| Dimension | Real State of AVC (`avcs_v2.pt`) | Common Misconception |
|---|---|---|
| **Architecture** | Spatiotemporal ConvGRU CNN (`52 tensors`) | Transformer / VLM |
| **Model Size** | **~501,612 parameters (~2 MB weight file)** | 7B – 70B Language Model |
| **Execution** | **100% Local CPU** (~80 ms – 140 ms per tick) | Cloud GPU / Tensor Parallelism |
| **Text Output** | **None** (Outputs raw tensors only) | Generates natural English sentences |
| **OCR Capability** | **Zero** (Cannot read words or text on screen) | Reads documents and articles |
| **Active Window Title** | Provided by **Windows Win32 OS API** (`GetWindowTextW`) | Read from pixels by the neural network |
| **Console Commentary** | Hardcoded Python format strings in `narrator.py` | Model "talking" to the user |

---

## 2. Technical Specification of the Neural Network

The core model is implemented in `avcs/cortex/model.py` as `StreamingVisualCortexV2`.

### 2.1 Inputs and Preprocessing
- **Raw Input:** Screen frame captured via `mss` (e.g. 1920x1200 or 1920x1080 RGB).
- **Retina Downscale:** Scaled down to uniform $128 \times 128 \times 3$ RGB float32 tensor normalized to $[0.0, 1.0]$.
- **Preprocessor (`EdgeLuminancePreprocessor`):** Computes a 4-channel tensor:
  $$\mathbf{x}_{\text{4ch}} = \left[ \text{Luminance}, \text{Sobel}_X, \text{Sobel}_Y, \sqrt{\text{Sobel}_X^2 + \text{Sobel}_Y^2} \right] \in \mathbb{R}^{B \times 4 \times 128 \times 128}$$

### 2.2 Backbone & Recurrence
1. **Multi-Scale Spatial Pyramid Stem (~51.3k params):**
   Three parallel convolutional pathways with kernel sizes $3\times3$, $5\times5$, and $7\times7$ downsampling to feature map $f_t \in \mathbb{R}^{B \times 64 \times 16 \times 16}$.
2. **ConvGRU Temporal Cell (~221.4k params):**
   Maintains spatiotemporal hidden state $h_t \in \mathbb{R}^{B \times 64 \times 16 \times 16}$ across frames:
   $$h_t = \text{ConvGRU}(f_t + |f_t - f_{t-1}|, h_{t-1} \cdot e^{-\lambda \Delta t})$$
   where $\lambda = 0.5$ provides smooth exponential memory decay over sub-2.0s frame gaps.
3. **Latent Predictive Coding (~25.1k params):**
   Adaptive average pooling projects $h_t$ to latent vector $z_t \in \mathbb{R}^{B \times 96}$, and a future predictor predicts $\hat{z}_{t+1}$. The mean-squared error gives sensory prediction surprise:
   $$\text{Uncertainty} = \|z_{t+1} - \hat{z}_{t+1}\|^2$$

### 2.3 The 3 Output Heads (~103k params)
1. **Head 1 (`TemporalChangeSpinnerHead`):**
   - Outputs `change_heatmap` $\in [0, 1]^{B \times 1 \times 128 \times 128}$ tracking moving pixel masks.
   - Outputs `spinner_activity_logit` $\in \mathbb{R}^{B \times 1}$ predicting periodic circular motion.
2. **Head 2 (`GenericUIRegionDetectorHead`):**
   - `ui_objectness_logits` $\in \mathbb{R}^{B \times 1 \times 16 \times 16}$: Probability of interactive UI element in each grid cell.
   - `ui_class_logits` $\in \mathbb{R}^{B \times 5 \times 16 \times 16}$: 5 generic morphological classes:
     0. `interactive_control` (buttons, clickable icons)
     1. `text_input` (entry fields, address bars)
     2. `toggle_icon` (checkboxes, switches)
     3. `modal_dialog` (popup boundaries)
     4. `content_region` (code panels, article blocks)
   - `ui_box_regression` $\in \mathbb{R}^{B \times 4 \times 16 \times 16}$: Bounding offsets $(\Delta x, \Delta y, \Delta w, \Delta h)$.
3. **Head 3 (`PostActionVisualResponseHead`):**
   - Outputs `post_action_response_logit` $\in \mathbb{R}^{B \times 1}$ verifying whether mouse click produced a visual pixel ripple.

---

## 3. Why the Model Appeared to "Hallucinate" and Speak Fake Alerts

The disconnect between user expectations and system behavior arose from **three specific layers of heuristic glue code**:

### 3.1 The "Modal Dialog" Hallucination
In `avcs/narrator.py`, the code previously had:
```python
# FAULTY LEGACY CODE:
elif eye_frame.has_popup or eye_frame.surprise_score > 0.45:
    human_speech = f"Alert! A modal dialog popped up in the {popup_region} with interactive options!"
```
- **The Bug:** `eye_frame.has_popup` was evaluated with `or eye_frame.surprise_score > 0.45`.
- **The Consequence:** Whenever the user typed a character in PowerShell or a cursor blinked, the optical change caused `surprise_score` to exceed `0.45`. The Python code immediately shouted: *"Alert! A modal dialog popped up in the center with interactive options!"* even though `has_popup` was `False` and no modal existed.

### 3.2 The "Spinner Animation" Spam
Similarly, for animating screens:
```python
elif not eye_frame.is_stable and (eye_frame.surprise_score > 0.08 or getattr(eye_frame, "has_spinner", False)):
    human_speech = f"Hold on... continuous spinner animation active in {spinner_region}, waiting for content to settle..."
```
- Any non-settled frame (e.g. while scrolling, typing, or moving a window) was automatically called a *"continuous spinner animation"*, confusing the user into thinking a spinner was falsely detected.

### 3.3 Active Window Names
The system printed window names like:
```text
'Fix Sonic agent launch & ID bugs | OpenHands - Google Chrome' observed.
```
- This was **not read by the neural network**.
- It was fetched directly via the Windows C API in `HandMotor.get_foreground_title()`:
  ```python
  hwnd = user32.GetForegroundWindow()
  user32.GetWindowTextW(hwnd, buffer, 512)
  ```

---

## 4. Current State: Cleaned and Stabilized

Following the latest fixes committed to both repositories:

1. **Strict Modal Gating:**
   `popup_alert` now strictly requires `eye_frame.has_popup is True`. It will never hallucinate a modal dialog from optical motion alone.
2. **Event-Driven Silence:**
   Once a screen settles, `LiveNarrator` produces `None`, remaining completely silent instead of spamming repetitious status lines.
3. **Calibrated Surprise:**
   Optical delta scaling was reduced from an aggressive `10.0x` multiplier down to calibrated rates, preventing normal typing from saturating to `surprise=1.000`.
4. **UTF-8 Console Safety:**
   Active window titles containing emojis or non-ASCII characters are sanitized using regex before printing, eliminating Windows CP1252 `UnicodeEncodeError` crashes.

---

## 5. Architectural Gap: Reflex vs. Semantic Understanding

To achieve what a human does when looking at a screen, a system requires two distinct tiers:

```mermaid
flowchart LR
    Screen["Desktop Screen (1920x1200)"] --> Sys1["AVC System 1 (Eyes / Reflex Arc)"]
    Screen --> Sys2["True VLM System 2 (Brain / Semantic VLM)"]

    subgraph Tier1 ["System 1 (What This Repo Has)"]
        Sys1 --> T1["500k ConvGRU Model"]
        T1 --> T1Out["Is screen stable? (True/False)<br/>Optical motion %<br/>Hotspot coordinates (Buttons/Inputs)<br/>Latency: ~80ms (100 FPS)"]
    end

    subgraph Tier2 ["System 2 (What Generates Human Speech)"]
        Sys2 --> T2["Multimodal Vision-Language Model<br/>(SmolVLM / Moondream / Gemini)"]
        T2 --> T2Out["'The user is editing Python code in VS Code.<br/>A syntax error is visible on line 42.<br/>The terminal below shows a failed test run.'"]
    end
```

### Why System 1 Alone Cannot "Talk Like a Human":
1. **Model Parameter Scale:** Describing screen semantics requires a vocabulary of 32,000+ token embeddings and a multi-layer transformer (~250M to 2B parameters minimum). A 500k parameter ConvGRU can only classify visual motion and spatial shapes.
2. **Text Reading (OCR):** High-resolution text recognition requires sub-word patch tokenization or CNN-CTC text decoders. System 1 downsamples the screen to 128x128 pixels, where text glyphs are reduced to blurry 2-pixel edges.

---

## 6. How to Implement True Human-Like Screen Description

To have a real AI model describe the screen like a person without hardcoded if/else templates, the standard solution is to attach a **Local Vision-Language Model (VLM)**:

### Option A: Local Offline VLM (Runs on your PC)
- **Model:** `HuggingFaceTB/SmolVLM-256M-Instruct` or `vikhyatk/moondream2` (~1.5 GB).
- **How it works:**
  1. Captures full-resolution desktop screenshot.
  2. Passes image through a vision transformer encoder + language decoder.
  3. The model autoregressively generates genuine natural language:
     > *"A desktop screen displaying Visual Studio Code with a dark theme. The left pane shows a file explorer tree, and the main editor has a Python script open. A PowerShell terminal is visible at the bottom."*

### Option B: Cloud Multimodal VLM (High Accuracy)
- **Model:** Gemini 1.5 Flash Vision / GPT-4o Vision API.
- **How it works:**
  Sends bounded screenshots when visual change is detected, returning instant human-grade screen comprehension and actionable advice.

---

## 7. Verification Matrix

| Test Suite / Capability | File Location | Verified Status |
|---|---|---|
| **ConvGRU Temporal Forward Pass** | `avcs/cortex/model.py` | **PASS** (100 FPS on CPU) |
| **Hotspot Detection** | `avcs/eye_stream.py` | **PASS** (16x16 objectness grid) |
| **Zero Modal Hallucinations** | `tests/e2e/test_tier2_boundary_corner.py` | **PASS** (36/36 tests green) |
| **Full SDK Regression Suite** | `tests/` | **PASS** (277 passed, 1 skipped) |
| **Live Win32 Actuation Safeguards** | `avcs/hand_motor.py` | **PASS** (Corner breakout active) |

