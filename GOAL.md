# AVCS — Honest Model Specification, Live Findings & System Roadmap

## 1. The Raw Reality: What This Model Actually Is

Strip away all marketing, hype, and academic sugar-coating. Here is the unvarnished truth of what this model is:

* **It is NOT an LLM, VLM, or thinking conversational AI:** It has no general intelligence, cannot reason, and cannot generate free-form conversational text.
* **It is a microscopic ~501,612-parameter (0.5M) ConvGRU reflex model:** Modern vision-language models (Claude 3.5 Sonnet, GPT-4o, Gemini) are 10 Billion to 100+ Billion parameters. This model is **~2.1 MB in weight size**.
* **Its entire job is low-latency visual reflex and motor coordination:**
  1. Captures live desktop frames at 95–100 FPS on a 4-core CPU without GPU.
  2. Tracks temporal motion, visual diffs, and unexpected pop-ups (surprise score) via ConvGRU recurrence.
  3. Detects high-contrast interactive hotspots (buttons, input fields).
  4. Executes smooth biological mouse trajectories (5th-order minimum-jerk kinematics) without exposing coordinate numbers.
  5. Verifies click postconditions via optical ripple deltas.
* **Who does the real cognitive work?**  
  * **Reading arbitrary text, code, or user input:** Requires **System 2 (External VLM / LLM)**.
  * **Planning, Reasoning, & Natural Dialogue:** Requires **System 2**.
  * **Safety, Error Handling, & Verification:** Handled 100% by deterministic code (`ControlKernel`, `VisualRippleVerifier`, permissions).
  * **System 1 (This Model):** Acts purely as the biological retina, visual reflex arc, and motor hands.

---

## 2. Live Desktop Test Findings (October 18, 2026)

Tested live on the operator's actual 1920×1200 physical display with active dark-themed windows:

1. **Resolution & Capture:** Real frame shape `(1200, 1920, 3)` grabbed at ~100 FPS with zero GDI handle leaks.
2. **Category Classification Reality:**
   - Predicted `browser_webpage` with only **50.1% confidence** (49.9% uncertainty) on a VS Code editor window.
   - *Why?* The category head was trained on synthetic layouts with basic templates; complex real-world IDEs (file trees, terminal tabs, dark-mode code editors) exceed its 8-category synthetic training.
3. **Arbitrary Text Reading Reality:**
   - **System 1 CANNOT read arbitrary text or sentences typed by the user.**
   - By clean-room design, all external OCR engines (Tesseract, EasyOCR, PaddleOCR) and Windows UIAutomation trees are strictly prohibited.
   - The visual patch classifier (`PixelWordRecognizer`) has a closed vocabulary of **exactly 131 functional UI action words** (`Save`, `Run`, `Cancel`, `Close`, `Terminal`, `Settings`, etc.).
   - Any free-form text, notes, code syntax, or sentences are outside its closed vocabulary and cannot be read without System 2.
4. **"Natural Dialogue" Reality:**
   - Observations emitted by System 1 (e.g., *"Display is active. Primary context: Browser Webpage (stable)..."*) are **deterministic Python string templates** slotting in predicted categories and hotspot counts, not generative natural language.

---

## 3. Division of Labor: System 1 vs System 2

| Capability | System 1 (This Local Model) | System 2 (External VLM / LLM) |
| :--- | :--- | :--- |
| **Model Size** | ~501,612 params (~2.1 MB) | 10B – 100B+ params |
| **Execution Hardware** | Local 4-core CPU (<10 ms, ~100 FPS) | Cloud GPU / High-memory host |
| **Visual Awareness** | Pure pixels, motion deltas, hotspots | High-level semantic scene understanding |
| **Text Reading** | 131 fixed UI button words only | Arbitrary free-form text, code, sentences |
| **Speech / Conversation** | Formatted string templates | True natural language reasoning & dialogue |
| **Motor Execution** | Physical mouse curves & keystrokes | None (never touches coordinates or hardware) |
| **Verification** | Frame diffs, optical ripples | Logical task postconditions |

---

## 4. Current Status of Codebase & Checkpoints

* **Stage A (Live Spatiotemporal Model & Mailbox):** PASSED (38/38 unit tests, 20/20 challenger tests).
* **Stage B (Dual-Model Frozen Twin Benchmark):** PASSED (+70.37% prediction gain over frozen twin, latent variance 0.9673, 90/90 tests pass).
* **Packaged Checkpoint:** `runs/weights/candidates/avcs_streaming_cortex_v1.pt` (SHA-256: `418c541e97b93615eccccd3cf488efaded84cd25f0384b865e3f5022c8c6032d`).
* **Preserved Baselines:** All 24 baseline checkpoints bit-for-bit preserved.
* **Integrity Audits:** Zero OCR imports, zero DOM inspection, zero raw screenshots saved to disk.

---

## 5. Roadmap & Next Steps

1. **Clean-room System 1 Candidate Packaging:** Freeze and maintain `avcs_streaming_cortex_v1.pt` as the local, pure-pixel sensorimotor reflex layer.
2. **System 2 VLM Bridge Integration:** Connect the live observation stream to a multimodal language model (Gemini / Claude / local VLM) to handle:
   - Full-text reading of arbitrary screen content.
   - Natural language dialogue explaining what is on screen.
   - High-level task planning that feeds discrete subgoals to System 1.
3. **No Overclaiming Mandate:** Document all capabilities and limitations truthfully. Never claim System 1 has general vision or language capabilities on its own.
