# MILESTONE 1: Core Perception Weights & Real Desktop Grounding

**Status:** 🟡 **IN PROGRESS**  
**Repository:** [https://github.com/Praxis-Neural/AVCS](https://github.com/Praxis-Neural/AVCS)  
**Target Delivery:** High-accuracy pre-trained AVCS neural weights (`avcs_cortex_weights.pt`) validated across 5 core desktop applications.

---

## 1. Objective

Train the ultra-compact 1.12M parameter AVCS model end-to-end using multi-task loss (Predictive V-JEPA + Dorsal Affordance + Ventral Category + Top-down Grounding) on Cloud GPU. Verify that real-time tick latency ($\le 25\text{ ms}$ GPU / $\le 80\text{ ms}$ CPU) and classification/affordance accuracy ($\ge 80\%$) are achieved across diverse desktop applications (VS Code, Chrome, Excel, Notion, PowerShell).

---

## 2. Deliverables & Task Checklist

| Task ID | Item Description | Status | Evidence / Artifact |
| :--- | :--- | :--- | :--- |
| **M1.1** | **Retinal Dual-Scale Foveation Architecture** | ✅ **COMPLETED** | `forge/vision/cortex/core/fovea.py` (0.42 ms crop) |
| **M1.2** | **Inverted Residual Mobile Backbone with SE Attention** | ✅ **COMPLETED** | `forge/vision/cortex/core/backbone.py` (1.12M params) |
| **M1.3** | **32x32 Dense Dorsal Affordance Segmentation Head** | ✅ **COMPLETED** | `forge/vision/cortex/core/dorsal.py` (5-class action fields) |
| **M1.4** | **Ventral Semantic Categorization Head** | ✅ **COMPLETED** | `forge/vision/cortex/core/ventral.py` (8 UI categories) |
| **M1.5** | **Temporal Predictive Coding (V-JEPA / Friston)** | ✅ **COMPLETED** | `forge/vision/cortex/core/predictive.py` (Surprise detection) |
| **M1.6** | **Automated Unit Test Suite (16 Tests)** | ✅ **COMPLETED** | `pytest tests/test_cortex.py` (16 passed in 9.87s) |
| **M1.7** | **Real-World UI Grounding Dataset Adapter** | ✅ **COMPLETED** | `forge/vision/cortex/data/real_ui.py` (ScreenSpot/RICO) |
| **M1.8** | **Real Desktop Screen Diversity Benchmark (POC-7)** | ✅ **COMPLETED** | `forge/vision/cortex/benchmark_real_desktop.py` (86.6ms CPU) |
| **M1.9** | **Turnkey Standalone Cloud GPU Training Script** | ✅ **COMPLETED** | `kaggle/avcs/train_avcs_gpu.py` (Zero-dependency) |
| **M1.10**| **Kaggle API Authentication & Remote Push** | ✅ **COMPLETED** | Kernel `137716045` pushed & verified |
| **M1.11**| **Cloud GPU Multi-Task Pre-Training Execution** | 🟡 **IN PROGRESS**| [Kaggle Kernel Running](https://www.kaggle.com/code/nandkishorrathod/avcs-visual-cortex-training) |
| **M1.12**| **Download Converged Weights (`avcs_cortex_weights.pt`)** | ⚪ **PENDING** | Awaiting Kaggle kernel completion |
| **M1.13**| **Re-Run POC-7 Benchmark with Trained Weights** | ⚪ **PENDING** | Measure validation accuracy gain vs baseline |
| **M1.14**| **Milestone 1 Final Verification & Sign-off** | ⚪ **PENDING** | Lock before opening Milestone 2 |

---

## 3. Measurable Acceptance Criteria

To declare Milestone 1 officially **COMPLETED**, the following conditions must be met:
1. **Loss Convergence:** Training loss drops steadily across 12 epochs on Kaggle Dual-T4 with no NaN/divergence.
2. **Category Accuracy:** Validation UI category classification reaches $\ge 80\%$ on held-out procedural and real screens (improving from the untrained 12.5% uniform baseline).
3. **Affordance Detection:** Dense 32x32 affordance head reliably detects clickable CTA buttons and typeable fields in test apps.
4. **Latency Budget:** Single forward-pass tick latency remains $\le 25\text{ ms}$ on GPU and $\le 85\text{ ms}$ on CPU.

---

## 4. Progress Log

* **2026-10-08 22:30:** Implemented `benchmark_real_desktop.py` testing VS Code, Chrome, Excel, Notion, and PowerShell at 1280x720. Recorded untrained baseline ($p50 = 86.67\text{ ms}$, uniform $12.5\%$ confidence).
* **2026-10-08 23:05:** Configured user Kaggle API token and packaged standalone `train_avcs_gpu.py`.
* **2026-10-08 23:12:** Pushed Version 3 kernel to Kaggle. Training started with PyTorch AMP FP16.
* **2026-10-08 23:36:** Added remote `https://github.com/Praxis-Neural/AVCS.git` and successfully pushed `main` and `forge-screen-vision` branches.
* **Current Status:** Kernel actively training in the cloud.
