# Department: Models (`avcs.models`)

## 1. Responsibilities & Boundaries
- Neural network architectures and forward policies for real-time desktop agency.
- Pure-RGB Vision-Language-Action (VLA) policy (`PureRGBVLAPolicy`, `VLACortexModel`).
- ShowUI Token Pruner (`ShowUITokenPruner`) for visual compression preserving cursor and interactive buttons.
- GUI Actor Patch Pointer (`GUIActorPatchPointer`) for coordinate prediction via cross-attention.
- Parallel Action Heads (`ParallelActionHeads`) for discrete action classes and continuous arguments.
- Flow Matching Drag Expert (`FlowMatchingDragExpert`) for continuous smooth dragging trajectories.
- Rolling KV Memory (`RollingKVMemory`) for temporal causal continuity.
- Stage 3 DAgger Recovery Policy (`DAggerRecoveryPolicy`) for hunting loop and modal popup mitigation.
- Multi-task loss functions (`AVCSPhase4MultiTaskLoss`).
- Local VAM client and Fast Brain grounding engines (`LocalVAMClient`, `LocalVAMEngine`).

## 2. Allowed Imports
- Standard Library (`math`, `typing`, `dataclasses`, `time`).
- `torch`, `torch.nn`, `torch.nn.functional`.
- `numpy`.
- `avcs.protocol` (for action and coordinate interpretations).

## 3. Forbidden Imports
- STRICT LINUX/KAGGLE SAFE: NEVER eagerly import Windows-only modules (`win32gui`, `ctypes.windll`, `avcs.os_io`).
- NEVER import training pipelines (`avcs.train`) or live runners (`avcs.runtime.runner`).

## 4. Sub-Agent Contract
- Exposes pure PyTorch `nn.Module` classes and forward callables.
- Modules must compile cleanly with TorchScript / CUDA Graphs and evaluate within the P95 < 350ms latency budget.
- Consumed by `avcs.train` during behavioral cloning / DAgger training and by `avcs.runtime` during inference.
