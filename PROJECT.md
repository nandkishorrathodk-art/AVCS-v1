# Project: AVCS (Artificial Visual Cortex System)

**Project Lead & Rights Holder:** Nandkishor Rathod (Parth Labs)  
**Status:** System 1 Candidates Trained & Verified; System 2 VLM Bridge Pending  
**Operating Environment:** Windows 11, Local 4-core CPU, No GPU, Pure-Pixel Clean-Room Contract  

---

## 1. Core Architecture

AVCS operates as a two-tier vision and action framework:

1. **System 1 (Local Sensory-Motor Cortex):**
   - **Continuous Pixel Ingestion:** 95–100 FPS screen capture via `ThreadedScreenCapture` with 1-frame atomic mailbox buffer.
   - **Spatiotemporal Neural Memory:** Multi-scale spatial encoder + `ConvGRUCell` recurrent hidden state tracking motion, diffs, and unexpected visual surprise.
   - **Dynamic UI Grounder & Fovea:** `GoalConditionedGrounder` + `FovealAcuityEngine` mapping UI targets to physical pixels.
   - **Closed Vocabulary Word Recognizer:** 239k-parameter patch ConvNet recognizing 131 distinct functional UI action words.
   - **Biological Motor Kinematics:** 5th-order minimum-jerk trajectory generator + 10.5 Hz micro-tremor toggle + fail-closed typing focus verification.
   - **Action Broker & Verifier:** `VisualRippleVerifier` verifying postconditions via fresh-frame photometric deltas.
   - **Zero Coordinate Leakage:** Absolute coordinate encapsulation; System 2 receives zero raw numbers.

2. **System 2 (Cognitive Language & Planning Brain):**
   - External VLM / LLM (Gemini / Claude / local multimodal model).
   - Responsible for full-text reading of arbitrary documents, articles, code, and user typing.
   - Responsible for high-level goal breakdown, strategy, and conversational speech with the operator.

---

## 2. Milestone Ledger & Current Status

| # | Milestone Name | Scope | Status | Verification Evidence |
|---|----------------|-------|--------|-----------------------|
| **M1** | Dataset & Benchmarks | Synthetic desktop generation, bounding box annotations, blank screen negatives | **DONE** | Cleanroom datasets, zero-leak splits |
| **M2** | Neural Grounding Head | Replace SHA-256 features, eliminate center collapse, train cross-attention | **DONE** | `GoalConditionedGrounder`, >=85% accuracy |
| **M3** | Foveal Acuity & Word Recognizer | High-res 1:1 patch refinement, 131-word closed UI vocabulary classifier | **DONE** | `PixelWordRecognizer` (239k params), foveal edge tests |
| **M4** | Motor Kinematics & Verification | 5th-order min-jerk curves, caret blink focus detector, optical ripple verifier | **DONE** | Verified in `tests/test_controller.py`, `tests/test_servoing_verification.py` |
| **Stage A** | Continuous Live Stream Model | `StreamingVisualCortexV1` (501k params), 1-frame mailbox, clean random initialization | **DONE** | 38/38 unit tests, 20/20 challenger tests |
| **Stage B** | Online Learning Twin Benchmark | SmoothL1 future prediction, VICReg anti-collapse variance (>0.05), identical frozen twin comparison | **DONE** | +70.37% prediction gain, 0.00000000 frozen drift, 90/90 tests pass |
| **Stage C** | Human-Like Dialogue (System 1) | Coordinate-free anatomical layout descriptions (`header`, `footer`, `sidebar`, `center`) | **DONE (Template)** | Formatted string templates; true generative dialogue awaits System 2 |
| **Stage D** | Production Qualification & Packaging | 1,000-tick soak test (zero leak), <10ms CPU latency, baseline hash preservation | **DONE** | Candidate checkpoint packaged, hashes verified |
| **System 2**| Full-Text Reading & Cognitive Bridge | Wire System 1 observation stream to VLM for arbitrary text reading and human conversation | **PENDING** | Next architectural phase |

---

## 3. Honest Live Test Findings & Current Limitations

1. **System 1 is Blind to Arbitrary Text:**
   - There is NO general OCR engine. It recognizes only 131 fixed functional UI buttons.
   - If the operator writes arbitrary text, notes, or code on screen, System 1 **cannot read it**.
2. **Synthetic Scene Categorization Fragility:**
   - Tested live on operator's actual 1920×1200 desktop running VS Code.
   - The model predicted `browser_webpage` with 50.1% confidence and 49.9% uncertainty because real-world IDEs were not part of its 8 synthetic scene categories.
3. **Dialogue Reality:**
   - System 1 emits formatted string templates based on detected hotspots. It cannot reason or answer open-ended questions without System 2.

---

## 4. Checkpoints & Provenance

* **Candidate Model:** `runs/weights/candidates/avcs_streaming_cortex_v1.pt` (SHA-256: `418c541e97b93615eccccd3cf488efaded84cd25f0384b865e3f5022c8c6032d`)
* **Metadata Sidecar:** `runs/weights/candidates/avcs_streaming_cortex_v1.json`
* **Baseline Preservation:** 24 legacy checkpoints bit-for-bit preserved and verified.
* **Prohibited Tech Audit:** Clean ast/grep audit confirming 0 OCR imports and 0 DOM/UIAutomation accessibility APIs.
