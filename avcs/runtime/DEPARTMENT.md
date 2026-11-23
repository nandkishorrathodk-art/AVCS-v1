# Department: Runtime (`avcs.runtime`)

## 1. Responsibilities & Boundaries
- Top-level autonomous agency execution engine (`AceAgent`, `AceRunner`, `CognitivePlanner`).
- Subgoal decomposition, English-only plan invariant enforcement, replanning, and step-budget tracking.
- Skill vault caching and anchor-verified deterministic replay (`AnchorVerifiedSkillCache`).
- Sensory feedback loops: Visual cortex stability gate integration, dual-stage verifiers, and loop deadlock mitigation.
- Frontier LLM and local model dispatch (`CustomLLMClient`, `LocalVAMClient`).

## 2. Allowed Imports
- Standard Library (`os`, `sys`, `time`, `re`, `logging`, `pathlib`, `typing`, `json`, `dataclasses`).
- `avcs.protocol` (for action schemas and coordinate transformations).
- `avcs.os_io` (for desktop observation and hardware actuation on Windows).
- `avcs.models` (for reflex model evaluation and local VAM fallback).
- `avcs.data` (for trajectory logging and episode recording).
- `PIL.Image`.

## 3. Forbidden Imports
- Training loops or dataset optimizers (`avcs.train`).
- Direct low-level Win32 ctypes calls (these belong exclusively in `avcs.os_io`).

## 4. Sub-Agent Contract
- Orchestrates end-to-end desktop automation sessions.
- Exposes clean entry points (`AceRunner.run()`, `AceAgent.step()`).
- Coordinates `avcs.os_io` for hardware interactions while keeping core decision logic platform-agnostic.
