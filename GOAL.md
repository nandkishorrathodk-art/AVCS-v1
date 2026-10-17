# AVCS — Honest Model Specification & Brutal Engineering Reality

## 1. The Raw Reality: What This Model Actually Is

Strip away all marketing, hype, and academic sugar-coating. Here is the unvarnished truth of what this model is:

* **It is NOT an LLM, VLM, or thinking AI:** It has no general intelligence, no reasoning, no concept of time, and zero memory of what happened a second ago.
* **It is a microscopic 1.35M-parameter CNN compass:** Modern vision models (Claude 3.5 Sonnet, GPT-4o, Qwen-2-VL) are 7 Billion to 70 Billion parameters. This model is **1,354,050 parameters (~5.5 MB file)**.
* **Its entire job is simple coordinate guessing:** It takes a 256×256 thumbnail of the desktop, takes a short query string (e.g. *"close button"*), and outputs a 64×64 heatmap guessing `(x, y)` coordinates.
* **Who does the real work?**  
  * **Planning & Thinking:** Handled 100% by **System 2 (External LLM)**.
  * **Safety, Error Handling, & Verification:** Handled 100% by **Python code (`VisualRippleVerifier`, `ControlKernel`)**.
  * **The Model:** Acts only as a dumb, fast visual pointer (a mathematical cursor compass).

---

## 2. The 5 Brutal Flaws & Bottlenecks (No Sugar-Coating)

### Flaw 1: The Model is Legally Blind (256×256 Resolution Blur)
* Real displays are **1920×1080** or 4K. The model forces the desktop into a **256×256 pixel thumbnail**.
* At this resolution, standard 10–12px system text or small 16×16 icons become **3 to 4 blurry grey pixels**.
* The model **cannot read text** (Zero OCR). It only matches fuzzy visual blobs and shapes. If two buttons have similar shapes or icons, it has zero ability to distinguish them without high-res foveal cropping.

### Flaw 2: The Softmax Hallucination Trap (Clicks on Blank Screens)
* Neural networks use a spatial `softmax` over the 64×64 grid. By mathematical definition, **softmax probabilities must sum to 1.0 (100%)**.
* If the screen is a completely blank solid color or empty wallpaper and you query *"click save button"*, the model **cannot output "nothing" on its own**. It will pick a random pixel of background noise and claim with confidence that the button is there.
* *Reality in code:* This is why Gate 4 failed on blank screens (2 false positive clicks). A separate target-presence classifier and calibrated abstention loss are strictly required to stop it from clicking into the void.

### Flaw 3: Synthetic Training Bias (It Has Never Seen the Real World)
* The model was trained almost entirely on **procedurally drawn Python graphics** (rectangles, synthetic buttons, basic Tkinter/Pillow mockups).
* It has **not** been trained on dirty, complex real-world enterprise software (SAP, Salesforce, custom Electron/WPF apps, dark-mode themes, canvas elements).
* On clean synthetic benchmarks, it achieves ~90.8% accuracy. On an unseen, cluttered, real-world enterprise screen, its raw accuracy will drop significantly unless constrained by code.

### Flaw 4: Zero Memory & Zero Context (Stateless Reflex)
* Each frame is processed in total isolation.
* If a click causes an unexpected error popup (*"Session Expired"*, *"Access Denied"*), the model does not know what happened. It will keep searching for its original target or click blindly unless Python code halts motor execution.

### Flaw 5: Scaling Parameter Myth (More Params Won't Magically Fix This)
* The repository's own measured facts proved that scaling parameters from 2.7M to 31M actually made the planner **worse** (Semantic rate dropped 56.7% → 10.0%).
* Simply making this network larger does not solve the grounding problem. Data quality, spatial calibration, and closed-loop verification are what keep it functioning.

---

## 3. Division of Labor: Who Actually Does What?

To make this system work without delusions, duties are strictly separated:

| Responsibility | Who Owns It | Why |
| :--- | :--- | :--- |
| **High-level Goal & Strategy** | **System 2 (Cloud LLM)** | Decides *what* needs to be done (e.g. "open notepad, write test, close without saving"). The local model cannot think. |
| **Coarse Target Guessing** | **AVCS Model (`GoalConditionedGrounder`)** | Fast sub-15ms reflex on CPU. Points the cursor in the rough direction of the visual blob. |
| **Sub-Pixel Precision** | **Python Foveal Engine (`grounding.py`)** | Takes an uncompressed 1:1 patch around the model's guess to find exact boundaries ($\pm 2\text{ px}$). |
| **Focus & Typing Security** | **Python Controller (`focus.py`)** | Checks caret blinking (0.8–2.0 Hz). Fails closed if focus is missing. Model cannot verify typing safety. |
| **Action Containment & Gates** | **Python Kernel (`ControlKernel`)** | Blocks unauthorized mutating operations (`mkdir`, `write`, dangerous clicks). |
| **Postcondition Verification** | **Python Verifier (`verifier.py`)** | Checks optical delta / window appearance after a click. Never trusts the model's self-reported success. |

---

## 4. What the Model MUST DO (Strict Requirements)

1. **Fast CPU Forward Pass:** Must execute in $\le 15\text{ ms}$ on 1 CPU thread (ensuring 60+ FPS real-time reflex).
2. **Dense Heatmap Prediction:** Must output a normalized 64×64 spatial distribution over the 256×256 canvas.
3. **Calibrated Presence Score:** Must predict a scalar presence probability $\in [0.0, 1.0]$. If target is absent, score must drop below threshold ($0.50$).
4. **Hard-Zero Blank Screen False Positives (Gate 4):** Under zero circumstances may the model authorize a click on an empty desktop or blank canvas.
5. **Fail-Closed Default:** If confidence is low or query is ambiguous, `action_authorized` must evaluate to `False`.

---

## 5. What the Model MUST NOT DO (Prohibitions & Anti-Cheating)

* ⛔ **NO Cheating with OCR:** Do NOT import Tesseract, EasyOCR, PaddleOCR, or Windows OCR.
* ⛔ **NO Cheating with Accessibility Trees / DOM:** Do NOT query Windows UIAutomation, accessibility APIs, or browser DOM (Selenium/Playwright).
* ⛔ **NO Hardcoded Coordinates:** Do NOT hardcode button positions in Python scripts.
* ⛔ **NO Raw Pixel Leakage:** Zero raw RGB frames, screenshots, or embeddings may leave the local machine to System 2.
* ⛔ **NO Self-Authorization:** The model's raw heatmap must never directly move the hardware mouse without passing through the `ControlKernel` and `VisualRippleVerifier`.
* ⛔ **NO Overclaiming:** Do NOT report synthetic test passes as "universal computer use solved". Report real numbers from real runs.

---

## 6. What Needs to Be Done (The Realistic Action Plan)

1. **Fix Blank Screen Rejection (Gate 4):**  
   Train with calibrated target-absence loss (margin penalty + negative spatial entropy regularization) on solid, gradient, and wallpaper frames so the model outputs near-zero logits on missing targets.
2. **Foveal Saccade Hand-off:**  
   Because 256×256 downsampling blinds the model to fine text, enforce that all small targets ($< 32\times32$) must be cropped at 1:1 physical resolution and refined by `FovealAcuityEngine`.
3. **Strict Verifier-Gated Execution:**  
   Treat the model as an unreliable advisor. Every click proposed by the model must be verified by `VisualRippleVerifier` against real screen deltas before System 2 is told that progress occurred.
4. **Natural Model-to-Model Communication:**  
   System 2 sends plain-language subgoals; System 1 executes substeps locally and reports real, observed visual facts back without robotic coordinate dumps.
