"""AVCS Models Department.

Pure-RGB Vision-Language-Action (VLA) neural network architectures,
ShowUI token pruners, GUI actor patch pointers, parallel action heads,
flow matching drag experts, rolling KV memory, and DAgger recovery policies.
Strictly Linux/Kaggle import safe (never imports Windows OS or win32 modules).
"""
from __future__ import annotations

from avcs.models.local_model import AceClientError, EmbeddedVAMRuntime, LocalVAMClient
from avcs.models.vla_cortex import (
    ACTION_VOCAB_10,
    AVCSPhase4MultiTaskLoss,
    CUDAGraphRunner,
    DAggerFailureType,
    DAggerIntervention,
    DAggerRecoveryPolicy,
    FlowMatchingDragExpert,
    GUIActorPatchPointer,
    GUIPointerOutput,
    LiteTrain,
    ParallelActionHeads,
    ParallelActionOutput,
    PlaybackEpisode,
    PlaybackStep,
    PrunedTokens,
    PureRGBVLAPolicy,
    PureRGBVLAOutput,
    PureTensorServingModule,
    RollingKVMemory,
    RollingMemoryState,
    ShowUITokenPruner,
    UnifiedMultiTaskLoss,
    VLACortexConfig,
    VLACortexModel,
    VLACortexOutput,
    overfit_check,
)

LocalVAMEngine = LocalVAMClient

__all__ = [
    "ACTION_VOCAB_10",
    "AVCSPhase4MultiTaskLoss",
    "AceClientError",
    "CUDAGraphRunner",
    "DAggerFailureType",
    "DAggerIntervention",
    "DAggerRecoveryPolicy",
    "EmbeddedVAMRuntime",
    "FlowMatchingDragExpert",
    "GUIActorPatchPointer",
    "GUIPointerOutput",
    "LiteTrain",
    "LocalVAMClient",
    "LocalVAMEngine",
    "ParallelActionHeads",
    "ParallelActionOutput",
    "PlaybackEpisode",
    "PlaybackStep",
    "PrunedTokens",
    "PureRGBVLAPolicy",
    "PureRGBVLAOutput",
    "PureTensorServingModule",
    "RollingKVMemory",
    "RollingMemoryState",
    "ShowUITokenPruner",
    "UnifiedMultiTaskLoss",
    "VLACortexConfig",
    "VLACortexModel",
    "VLACortexOutput",
    "overfit_check",
]
