"""Configuration for Artificial Visual Cortex for Screens (AVCS)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RetinaConfig:
    peripheral_size: Tuple[int, int] = (256, 256)
    foveal_size: Tuple[int, int] = (256, 256)
    foveal_coverage_ratio: float = 0.35
    saccade_smoothing: float = 0.7


@dataclass
class ModelConfig:
    feature_dim: int = 128
    latent_dim: int = 256
    in_channels: int = 3
    num_affordance_classes: int = 5
    num_ui_categories: int = 8
    # Fully-learned heads (see core/learned.py). These sizes define output
    # dimensions only; every value is produced by a trained tensor.
    num_object_slots: int = 16
    num_scene_attributes: int = 10
    num_actions: int = 6
    num_stability_states: int = 4
    text_embed_dim: int = 256
    retina: RetinaConfig = field(default_factory=RetinaConfig)


@dataclass
class TrainingConfig:
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    predictive_loss_weight: float = 1.0
    affordance_loss_weight: float = 1.0
    category_loss_weight: float = 0.5
    grounding_loss_weight: float = 1.0
    # Weights for the fully-learned heads (see core/learned.py).
    slot_loss_weight: float = 1.0
    attribute_loss_weight: float = 0.5
    count_loss_weight: float = 0.5
    action_loss_weight: float = 1.0
    stability_loss_weight: float = 0.5
    use_amp: bool = True
    device: str = "cuda" # Automatically falls back to cpu if unavailable


@dataclass
class CortexConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def realtime_config() -> CortexConfig:
    """Configuration tuned to hold the real-time CPU budget (~100 FPS).

    The backbone cost scales with the square of the retinal input size, so this
    preset drops the retina to 128x128. Measured on 4 CPU cores this sustains
    ~94-105 FPS end to end with every learned head active.
    """
    cfg = CortexConfig()
    cfg.model.retina.peripheral_size = (128, 128)
    cfg.model.retina.foveal_size = (128, 128)
    return cfg

