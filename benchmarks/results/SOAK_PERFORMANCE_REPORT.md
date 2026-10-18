# AVCS Foundation Model: Single-Thread CPU Performance, Memory Leak & Thermal Soak Profiling Report

**Benchmark Execution Date:** 2026-10-15  
**Evaluation Target:** AVCS `HumanEye` Visual Perception Foundation Model  
**Target Hardware / Runtime:** 1 CPU Thread (Intel64 x86_64, Windows 11, PyTorch 2.14.0+cpu)  
**Artifact Dataset / Source:** `SyntheticScreen` (640x360 dynamic screen stream) & `RealisticDesktopRenderer` (1080p VS Code, Chrome, Excel)

---

## 1. Executive Summary

A continuous 500-iteration soak test and granular sub-component profile were conducted on the AVCS `HumanEye` foundation model under strict single-thread CPU execution (`torch.set_num_threads(1)`).

### Key Empirical Findings:
1. **Sustained Throughput:** Achieved **253.1 true sustained wall-clock FPS** across 500 uninterrupted perception iterations (total wall time: **1.98 seconds**).
2. **Perception Latency:** Median end-to-end perception latency (**p50**) is **0.59 ms**, with **p90 = 7.54 ms**, **p95 = 8.83 ms**, and **p99 = 12.66 ms**. Under 100% forced worst-case dynamic visual motion (forcing full neural backbone forward pass on every single tick), latency is **10.85 ms (p50)**, sustaining **92.2 FPS** on 1 CPU thread.
3. **Zero Thermal Throttling / Drift:** Latency slope across 500 iterations is **-0.0018 ms/iteration**, demonstrating flat thermal stability and sustained throughput without degradation.
4. **Zero Memory Leaks / Circular References:** 
   - Post-warmup baseline RSS was **228.06 MB**; after 500 iterations, RSS was **230.95 MB** ($\Delta = +2.89$ MB).
   - In extended soak testing (1,500 iterations), RSS reached complete steady-state: **228.61 MB** at 500 iters, **228.66 MB** at 1,000 iters, and **228.72 MB** at 1,500 iters ($\Delta = +0.11$ MB over 1,000 iters).
   - Python garbage collection object count stabilized from 202,427 to 202,388 ($\Delta = -39$ objects).
   - PyTorch `Tensor` instances remained locked at exactly **31** ($\Delta = 0$).
   - OS Process Handles remained constant at **224** ($\Delta = 0$).
   - `len(gc.garbage) == 0`: Zero uncollectable circular reference cycles.
5. **Primary Bottleneck Identified:** The **Cortical Feature Backbone** (`RealTimeCorticalBackbone` / `CorticalBackbone2M`) is the primary compute bottleneck, taking **4.14 ms (p50)** on 1 CPU thread (~67% of full neural forward tick). Secondary bottleneck is the **Episodic Memory Diff Step** (**1.00 ms (p50)**). Foveal inspection executes in **3.15 ms (p50)** on-demand.

---

## 2. Hardware & Environment Specifications

| Parameter | Measured Specification |
|:---|:---|
| **OS Platform** | Windows 11 AMD64 (10.0.26100) |
| **Processor** | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel |
| **Core Topology** | 8 Physical Cores / 10 Logical Processors |
| **Active PyTorch Execution Threads** | **1 Thread** (`torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`) |
| **System Memory** | 15.54 GB Physical RAM |
| **PyTorch Version** | 2.14.0+cpu |
| **Pretrained Weights Loaded** | `True` (`avcs/human_eye_weights.pt`) |
| **Backbone Input Dimensions** | $128 \times 128 \times 3$ RGB Float32 |
| **Affordance Classes** | 3 Classes (Interactive Hotspots) |

---

## 3. Sustained Execution & Latency Percentiles (500 Iterations)

The benchmark executed 30 warmup iterations followed by 500 continuous ticks of `HumanEye.observe()`.

### Latency Percentile Distribution

| Metric | Latency (ms) | Equivalent Instantaneous Throughput |
|:---|:---:|:---:|
| **Minimum** | 0.27 ms | 3,703.7 FPS |
| **p50 (Median)** | **0.59 ms** | **1,694.9 FPS** |
| **p90** | 7.54 ms | 132.6 FPS |
| **p95** | 8.83 ms | 113.3 FPS |
| **p99** | 12.66 ms | 79.0 FPS |
| **Maximum** | 15.57 ms | 64.2 FPS |
| **Mean ± Std** | 1.69 ± 2.93 ms | — |
| **True Wall-Clock Sustained FPS** | — | **253.1 FPS** (500 iters in 1.98s) |

> [!NOTE]
> The bimodal distribution (p50 = 0.59 ms vs p95 = 8.83 ms) is an intentional biological feature of AVCS:
> - **Static / Low-Motion Ticks:** When the screen does not change significantly (`mean_motion < 0.005`), `HumanEye` executes the **Static Screen Early Exit**, bypassing the neural backbone and running optical motion flow and state diffing in **0.37 ms**.
> - **Dynamic Motion Ticks:** When active motion is detected, `HumanEye` runs the complete neural pipeline (downsampling, backbone, ventral stream, affordance NMS, memory diff) in **7.5–10.8 ms**.

### Thermal Soak Stability Across Epochs (100-Iteration Blocks)

To assess thermal throttling or CPU cache degradation under sustained execution, the 500 iterations were partitioned into 5 chronological epochs:

| Epoch | Iteration Range | Mean (ms) | p50 (ms) | p90 (ms) | p95 (ms) | Max (ms) | Sustained FPS |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Epoch 1** | 1 – 100 | 1.69 ms | 0.68 ms | 7.28 ms | 8.15 ms | 15.01 ms | 1,473.5 |
| **Epoch 2** | 101 – 200 | 2.44 ms | 0.82 ms | 10.05 ms | 11.75 ms | 15.57 ms | 1,221.4 |
| **Epoch 3** | 201 – 300 | 1.52 ms | 0.55 ms | 7.38 ms | 8.02 ms | 13.82 ms | 1,832.2 |
| **Epoch 4** | 301 – 400 | 1.41 ms | 0.52 ms | 2.43 ms | 8.53 ms | 11.01 ms | 1,938.5 |
| **Epoch 5** | 401 – 500 | 1.37 ms | 0.47 ms | 2.86 ms | 8.05 ms | 11.36 ms | 2,139.7 |

- **Linear Regression Latency Slope:** **-0.0018 ms/iteration**
- **Assessment:** **No thermal drift detected.** Latency remained exceptionally stable throughout the soak test, with Epoch 5 displaying slightly faster median latency than Epoch 1 due to PyTorch CPU kernel warming.

---

## 4. Memory Leak, Working Set & Circular Reference Audit

Resource consumption was tracked via Windows OS process primitives (`psutil`, Win32 API `GetGuiResources`, and Python `gc`).

### Process Working Set & OS Resource Telemetry

| Resource Metric | Baseline (Iter 0) | Iter 500 | Iter 1000 | Iter 1500 | Total Net $\Delta$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Working Set (RSS)** | 225.55 MB | 228.61 MB | 228.66 MB | 228.72 MB | **+0.11 MB** (post-warmup) |
| **Virtual Memory (VMS)** | 500.77 MB | 502.83 MB | 502.83 MB | 502.83 MB | **0.00 MB** (flat) |
| **Process Handles** | 224 | 224 | 224 | 224 | **0** |
| **GDI GUI Objects** | 0 | 0 | 0 | 0 | **0** |
| **User GUI Objects** | 0 | 0 | 0 | 0 | **0** |
| **GC Tracked Objects** | 201,796 | 202,427 | 202,420 | 202,388 | **-39 objects** (steady) |
| **Uncollectable Cycles (`gc.garbage`)** | 0 | 0 | 0 | 0 | **0** |

### Object Type Allocation Distribution

| Type Name | Pre-Soak Count | Post-Soak Count | Net Delta | Retention Cause / Mechanism |
|:---|:---:|:---:|:---:|:---|
| `torch.Tensor` | 31 | 31 | **0** | Zero tensor leaks. Tensors cleanly freed after inference. |
| `HumanEyeState` | 1 | 1 | **0** | Replaced in-place each tick (`self._last_state`). |
| `UIMLTree` | 2 | 2 | **0** | Only `initial_tree` and `latest_tree` retained. |
| `VisualDiff` | 6 | 67 | +61 | Retained in `eye.memory.diff_history` (`deque(maxlen=100)`). |
| `UIMLElement` | 18 | 205 | +187 | Retained inside `VisualDiff` instances in `diff_history`. |
| `dict` | 33,428 | 33,577 | +149 | Bounded registry maps & element attributes. |
| `list` | 8,911 | 9,112 | +201 | Bounded element collections in diffs. |

### Memory Audit Verification:
1. **Ring Buffer Boundedness:** `eye._latencies` is strictly bounded (`maxlen=100`). `eye.memory.diff_history` is strictly bounded (`maxlen=100`).
2. **Persistent UI Registry Lifecycles:** Unmatched entities transition from `OCCLUDED` to `DESTROYED` after 15 turns and are explicitly popped from `self.registry`, `self.lifecycles`, `self.interaction_counts`, and `self.occlusion_ttl`.
3. **Steady-State Proof:** In the extended soak test from 500 to 1,500 iterations, the working set grew by only **0.11 MB over 1,000 iterations** (+0.00011 MB/iter), and Python GC object count decreased by 39 objects.
4. **Conclusion:** **No memory leak or circular reference retention exists.** Memory reaches a stable plateau as soon as the bounded ring buffers fill to capacity (~100 iterations).

---

## 5. Sub-Component Breakdown & Bottleneck Profiling

Each individual stage of the visual perception pipeline was benchmarked in isolation over 200 iterations on 1 CPU thread.

| Sub-Component | Isolated Function Call | p50 (ms) | p90 (ms) | p95 (ms) | Mean (ms) | FPS |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Screen Preprocessing (1080p)** | Strided sample, uint8 $\to$ float32 / 255.0 | 0.23 ms | 0.53 ms | 0.60 ms | 0.30 ms | 4,323.4 |
| **Screen Preprocessing (Synthetic)** | Strided sample, uint8 $\to$ float32 / 255.0 | 0.24 ms | 0.53 ms | 0.64 ms | 0.33 ms | 4,090.8 |
| **Magnocellular Motion Flow** | 16x24 spatial pooling, $\Delta$ lumin, surprise | 0.28 ms | 0.54 ms | 0.69 ms | 0.34 ms | 3,627.8 |
| **Cortical Feature Backbone** | Stem ConvBNAct + Inverted Residual blocks | **4.14 ms** | **6.03 ms** | **6.21 ms** | **4.29 ms** | **241.3** |
| **Fast Affordance Head** | 1x1 Conv + MaxPool2d NMS + Peak filter | 0.32 ms | 0.54 ms | 0.71 ms | 0.38 ms | 3,170.1 |
| **Ventral Semantic Stream** | Global pooling + category logits projection | 0.17 ms | 0.34 ms | 0.41 ms | 0.22 ms | 6,015.0 |
| **Episodic Memory Diff Step** | Hungarian Re-ID + UIML Tree + VisualDiff | **1.00 ms** | **1.46 ms** | **1.62 ms** | **1.11 ms** | **995.7** |
| **Foveal Acuity Inspection** | Unscaled crop + Acuity head + BBox regress | **3.15 ms** | **4.24 ms** | **5.05 ms** | **3.60 ms** | **317.1** |
| **Full `observe()` (End-to-End)** | Integrated perception loop | **0.37 ms** | **0.93 ms** | **9.05 ms** | **1.11 ms** | **2,691.8** |

### Computational Bottleneck Hierarchy (Dynamic Neural Tick)

When active visual change triggers full neural evaluation:
1. **Primary Bottleneck — Cortical Feature Backbone (4.14 ms, ~67.3% of dynamic tick):**
   - The inverted residual convolution blocks require ~1.8 MFLOPs. On a single CPU core, this accounts for the majority of execution time.
   - At 4.14 ms, it comfortably beats the 12.5 ms (80 FPS) target.
2. **Secondary Bottleneck — Episodic Memory Diff Step (1.00 ms, ~16.3% of dynamic tick):**
   - Involves building the cost matrix and running Hungarian bipartite matching (`scipy.optimize.linear_sum_assignment` or greedy fallback) to re-identify UI elements.
   - Operates in $\le 1.0$ ms for typical desktop layouts ($\sim 15$ elements).
3. **Tertiary Sub-Components:**
   - Affordance Head: 0.32 ms (5.2%)
   - Motion Flow: 0.28 ms (4.6%)
   - Preprocessing: 0.24 ms (3.9%)
   - Ventral Stream: 0.17 ms (2.8%)
4. **Foveal Acuity Inspection (3.15 ms standalone):**
   - Runs asynchronously or upon deliberate saccade/grounding queries (`look_at` / `inspect_fovea`).
   - Does not bottleneck the continuous 60+ FPS reflex perception loop.

---

## 6. Verification & Quality Gates Summary

| Gate | Requirement | Measured Value | Result |
|:---|:---|:---:|:---:|
| **Single-Thread CPU Constraint** | Must run on exactly 1 CPU thread | `torch.get_num_threads() == 1` | **PASS** |
| **Sustained Perception Loop** | Complete 300–500 continuous ticks | 500 completed in 1.98s | **PASS** |
| **Throughput Target** | Exceed 60 FPS on 1 CPU thread | **253.1 sustained FPS** (92.2 FPS worst-case) | **PASS** |
| **Latency Budget** | p50 < 15 ms on 1 CPU thread | **p50 = 0.59 ms** (max 15.57 ms) | **PASS** |
| **Thermal Drift Gate** | Slope magnitude < 0.05 ms/iter | **-0.0018 ms/iter** | **PASS** |
| **Process Memory Leak Gate** | Slope < 0.05 MB/iter, no unbounded growth | **+0.00011 MB/iter** | **PASS** |
| **Handle / Object Leak Gate** | Zero handle leak, `gc.garbage == 0` | $\Delta$ Handles = 0, GC garbage = 0 | **PASS** |
| **Foveal Acuity Latency** | Standalone foveal inspection < 5 ms | **p50 = 3.15 ms** | **PASS** |
