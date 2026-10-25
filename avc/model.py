"""Artificial Visual Cortex (AVC) Unified Model.

Coordinates Retinal Foveation, Dual-path Backbone, Dorsal Stream (Affordances),
Ventral Stream (Semantics), Language Modulation, and Predictive Coding.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from avc.config import CortexConfig, ModelConfig
from avc.core.backbone import CorticalBackbone
from avc.core.dorsal import (
    AFFORDANCE_CLASSES,
    INTERACTIVE_CLASSES,
    DorsalAffordanceStream,
)
from avc.core.fovea import RetinalFoveation, RetinalView
from avc.core.learned import (
    ACTION_NAMES,
    SCENE_ATTRIBUTES,
    ActionPolicyHead,
    LearnedStabilityHead,
    LearnedTextEncoder,
    SceneUnderstandingHead,
    SlotDetector,
)
from avc.core.modulation import LanguageModulation
from avc.core.predictive import (
    STABILITY_STATES,
    PredictiveCodingEngine,
)
from avc.core.servoing import VisualMotorState, VisualServoingEngine
from avc.core.ventral import UI_CATEGORIES, VentralSemanticStream


@dataclass
class CortexState:
    """Rich, typed visual perception state emitted every tick."""
    visual_latent: np.ndarray             # (latent_dim,) compact semantic embedding
    primary_category: str                 # "browser_webpage", "editor_code", etc.
    category_confidence: float
    text_density: float                   # [0, 1] text saturation
    stability_state: str                  # "stable", "animating", "transitioning", "surprise_event"
    surprise_score: float                 # Temporal prediction error
    fixation: Tuple[float, float]         # Current gaze point (x, y) in [0, 1]
    motor_flow: Tuple[float, float] = (0.0, 0.0) # Closed-loop hand-eye steering vector (dx, dy)
    contact_acquired: bool = False        # True when hand/cursor aligns with gaze
    action_hotspots: List[Tuple[float, float, str, float]] = None # Interactive affordances
    affordance_map: Optional[np.ndarray] = None           # (H_grid, W_grid) dense discrete affordance mask
    query_target: Optional[Tuple[float, float]] = None    # Language grounded gaze target
    query_relevance_map: Optional[np.ndarray] = None
    objects: List[dict] = None                            # Detected object slots
    scene_attributes: Optional[np.ndarray] = None         # (num_attributes,) sigmoid probs
    object_count: float = 0.0                             # Regressed object count
    action: str = "noop"                                  # argmax over learned action logits
    action_target: Optional[Tuple[float, float]] = None   # Learned action coordinate
    action_confidence: float = 0.0
    inference_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_category": self.primary_category,
            "category_confidence": round(self.category_confidence, 3),
            "text_density": round(self.text_density, 3),
            "stability_state": self.stability_state,
            "surprise_score": round(self.surprise_score, 4),
            "fixation": (round(self.fixation[0], 3), round(self.fixation[1], 3)),
            "motor_flow": (round(self.motor_flow[0], 3), round(self.motor_flow[1], 3)),
            "contact_acquired": self.contact_acquired,
            "action_hotspots": [
                (round(x, 3), round(y, 3), act, round(c, 3))
                for x, y, act, c in (self.action_hotspots or [])
            ],
            "query_target": (
                (round(self.query_target[0], 3), round(self.query_target[1], 3))
                if self.query_target else None
            ),
            "objects": self.objects or [],
            "scene_attributes": (
                [round(float(v), 3) for v in self.scene_attributes]
                if self.scene_attributes is not None else None
            ),
            "object_count": round(float(self.object_count), 2),
            "action": self.action,
            "action_target": (
                (round(self.action_target[0], 3), round(self.action_target[1], 3))
                if self.action_target else None
            ),
            "action_confidence": round(float(self.action_confidence), 3),
            "inference_ms": round(self.inference_ms, 2),
        }


class ArtificialVisualCortex(nn.Module):
    """The Complete Artificial Visual Cortex (AVC) Neural Network."""

    def __init__(self, config: Optional[ModelConfig | CortexConfig] = None) -> None:
        super().__init__()
        if config is None:
            self.cfg = ModelConfig()
        elif isinstance(config, CortexConfig):
            self.cfg = config.model
        else:
            self.cfg = config

        # 1. Retinal Foveation (Sensor view extraction)
        self.retina = RetinalFoveation(
            peripheral_size=self.cfg.retina.peripheral_size,
            foveal_size=self.cfg.retina.foveal_size,
            foveal_coverage_ratio=self.cfg.retina.foveal_coverage_ratio,
            saccade_smoothing=self.cfg.retina.saccade_smoothing,
        )

        # 2. Cortical Neural Backbone
        self.backbone = CorticalBackbone(
            in_channels=self.cfg.in_channels,
            feature_dim=self.cfg.feature_dim,
        )

        # 3. Dorsal Stream ("WHERE/HOW" Affordances)
        self.dorsal = DorsalAffordanceStream(
            in_dim=self.cfg.feature_dim,
            num_classes=self.cfg.num_affordance_classes,
        )

        # 4. Ventral Stream ("WHAT" Semantics)
        self.ventral = VentralSemanticStream(
            in_dim=self.cfg.feature_dim,
            latent_dim=self.cfg.latent_dim,
        )

        # 5. Top-down Language Modulation
        self.language = LanguageModulation(
            visual_dim=self.cfg.feature_dim,
            text_dim=self.cfg.latent_dim,
        )

        # 6. Temporal Predictive Coding (World Model)
        self.predictive = PredictiveCodingEngine(
            latent_dim=self.cfg.latent_dim,
        )

        # 7. Biological Hand-Eye Servoing Engine
        self.servoing = VisualServoingEngine()

        # 8. Fully-learned heads
        self.text_encoder = LearnedTextEncoder(out_dim=self.cfg.latent_dim)
        self.slot_detector = SlotDetector(
            feature_dim=self.cfg.feature_dim,
            num_slots=self.cfg.num_object_slots,
            num_classes=self.cfg.num_affordance_classes,
        )
        self.scene_head = SceneUnderstandingHead(
            latent_dim=self.cfg.latent_dim,
            num_attributes=self.cfg.num_scene_attributes,
        )
        self.action_policy = ActionPolicyHead(
            latent_dim=self.cfg.latent_dim,
            text_dim=self.cfg.latent_dim,
            num_actions=self.cfg.num_actions,
        )
        self.stability_learned = LearnedStabilityHead(
            latent_dim=self.cfg.latent_dim,
            num_states=self.cfg.num_stability_states,
        )
        self._prev_state_idx: int = 0

    def reset_temporal_memory(self) -> None:
        """Reset internal temporal world model."""
        self.predictive.reset()

    def forward(
        self,
        peripheral: torch.Tensor,
        foveal: torch.Tensor,
        text_embedding: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Single-pass forward execution."""
        # 1. Visual Feature Extraction & Multi-scale Fusion
        fused_feat, feat_p, feat_f = self.backbone(peripheral, foveal)

        # 2. Dorsal Stream: Affordance Maps
        affordance_logits = self.dorsal(fused_feat)

        # 3. Ventral Stream: Category & Scene Latent
        ventral_out = self.ventral(fused_feat)
        scene_latent = ventral_out["latent"]

        # 4. Predictive Coding: Temporal Surprise & Stability
        updated_memory, surprise_tensor, stability_logits = (
            self.predictive(scene_latent)
        )

        # 5. Optional Language Modulation (Zero-generation grounding)
        rel_map, query_coords = None, None
        if text_embedding is not None:
            rel_map, query_coords = self.language(fused_feat, text_embedding)

        # 6. Fully-learned heads
        slots = self.slot_detector(fused_feat)
        scene_understanding = self.scene_head(scene_latent)

        policy_query = (
            text_embedding
            if text_embedding is not None
            else self.text_encoder([""] * scene_latent.shape[0], device=scene_latent.device)
        )
        action_out = self.action_policy(scene_latent, policy_query)

        prev_idx = torch.full(
            (scene_latent.shape[0],), self._prev_state_idx,
            dtype=torch.long, device=scene_latent.device,
        )
        stability_logits_learned = self.stability_learned(
            updated_memory, surprise_tensor, prev_idx
        )

        return {
            "fused_feat": fused_feat,
            "affordance_logits": affordance_logits,
            "category_logits": ventral_out["category_logits"],
            "text_density": ventral_out["text_density"],
            "scene_latent": scene_latent,
            "temporal_memory": updated_memory,
            "surprise_score": surprise_tensor,
            "stability_logits": stability_logits,
            "relevance_map": rel_map,
            "query_coords": query_coords,
            "slots": slots,
            "attribute_logits": scene_understanding["attribute_logits"],
            "object_count": scene_understanding["count"],
            "action_logits": action_out["action_logits"],
            "action_target": action_out["action_target"],
            "action_confidence": action_out["action_confidence"],
            "stability_logits_learned": stability_logits_learned,
        }

    def encode_query(self, query: str | list[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """Encodes a natural-language query with the learned text encoder."""
        return self.text_encoder(query, device=device)

    @torch.no_grad()
    def perceive(
        self,
        frame: np.ndarray | torch.Tensor,
        target_fixation: Optional[Tuple[float, float]] = None,
        cursor_position: Optional[Tuple[float, float]] = None,
        text_embedding: Optional[torch.Tensor] = None,
        device: Optional[torch.device | str] = None,
    ) -> CortexState:
        """High-level real-time perception interface."""
        t0 = time.perf_counter()

        if device is None:
            device = next(self.parameters()).device

        # 1. Retinal Foveation View Extraction
        retinal_view = self.retina.extract_views(frame, target_fixation=target_fixation)
        periph = retinal_view.peripheral.to(device)
        fov = retinal_view.foveal.to(device)

        if text_embedding is not None:
            text_embedding = text_embedding.to(device)

        # 2. Execute Neural Forward Pass
        out = self.forward(periph, fov, text_embedding=text_embedding)

        # 3. Extract High-Level Readouts
        cat_probs = torch.softmax(out["category_logits"], dim=-1)[0]
        cat_idx = int(torch.argmax(cat_probs).item())
        cat_name = UI_CATEGORIES[cat_idx] if cat_idx < len(UI_CATEGORIES) else "unknown"
        cat_conf = float(cat_probs[cat_idx].item())

        text_dense = float(out["text_density"][0].item())
        surprise = max(0.0, min(1.0, float(out["surprise_score"].mean().item())))

        # 4. Learned temporal stability
        stab_idx = int(torch.argmax(out["stability_logits_learned"][0]).item())
        stability = (
            STABILITY_STATES[stab_idx] if stab_idx < len(STABILITY_STATES) else "stable"
        )
        self._prev_state_idx = stab_idx
        self.predictive.current_state = stability

        # 5. Learned object slots
        slot_obj = torch.sigmoid(out["slots"]["objectness"][0])
        slot_boxes = out["slots"]["boxes"][0]
        slot_cls = torch.softmax(out["slots"]["class_logits"][0], dim=-1)
        order = torch.argsort(slot_obj, descending=True)

        objects: List[dict] = []
        hotspots: List[Tuple[float, float, str, float]] = []
        for i in order.tolist():
            cls_idx = int(torch.argmax(slot_cls[i]).item())
            cls_name = AFFORDANCE_CLASSES[cls_idx] if cls_idx < len(AFFORDANCE_CLASSES) else "static"
            conf = float(slot_obj[i].item())
            cx, cy = float(slot_boxes[i][0].item()), float(slot_boxes[i][1].item())
            objects.append({
                "box": [round(float(v), 3) for v in slot_boxes[i].tolist()],
                "objectness": round(conf, 3),
                "class": cls_name,
                "class_confidence": round(float(slot_cls[i][cls_idx].item()), 3),
            })
            if cls_name in INTERACTIVE_CLASSES:
                hotspots.append((cx, cy, cls_name, conf))
        hotspots = hotspots[: self.cfg.num_object_slots]

        # 6. Extract Language Grounding (if query passed)
        query_target = None
        rel_map_np = None
        if out["query_coords"] is not None:
            coords = out["query_coords"][0].cpu().numpy()
            query_target = (float(coords[0]), float(coords[1]))
            rel_map_np = out["relevance_map"][0, 0].cpu().numpy()

        # 7. Biological Hand-Eye Coordination & Visual Servoing
        motor_flow = (0.0, 0.0)
        contact_acquired = False
        target_gaze = query_target or retinal_view.fixation
        if cursor_position is not None:
            servo_state = self.servoing.compute_motor_flow(
                gaze_target=target_gaze,
                current_cursor=cursor_position,
                confidence=cat_conf,
            )
            motor_flow = servo_state.motor_velocity
            contact_acquired = servo_state.contact_acquired

        aff_map_np = torch.argmax(out["affordance_logits"][0], dim=0).cpu().numpy()
        scene_attributes = torch.sigmoid(out["attribute_logits"][0]).cpu().numpy()
        object_count = float(out["object_count"][0].item())

        action_logits = out["action_logits"][0]
        action_idx = int(torch.argmax(action_logits).item())
        action_name = ACTION_NAMES[action_idx] if action_idx < len(ACTION_NAMES) else "noop"
        action_target = tuple(float(v) for v in out["action_target"][0].cpu().tolist())
        action_conf = float(out["action_confidence"][0].item())

        dt = (time.perf_counter() - t0) * 1000.0

        return CortexState(
            visual_latent=out["scene_latent"][0].cpu().numpy(),
            primary_category=cat_name,
            category_confidence=cat_conf,
            text_density=text_dense,
            stability_state=stability,
            surprise_score=surprise,
            fixation=retinal_view.fixation,
            motor_flow=motor_flow,
            contact_acquired=contact_acquired,
            action_hotspots=hotspots,
            affordance_map=aff_map_np,
            query_target=query_target,
            query_relevance_map=rel_map_np,
            objects=objects,
            scene_attributes=scene_attributes,
            object_count=object_count,
            action=action_name,
            action_target=action_target,
            action_confidence=action_conf,
            inference_ms=dt,
        )

    def load_weights(
        self,
        weights_path: Union[str, Path],
        strict: bool = False,
        api_key: Optional[str] = None,
    ) -> None:
        """Loads trained weights checkpoint into the cortex modules.

        Supports both plaintext (.pt) checkpoints and encrypted binary vaults (.avcs).
        For .avcs vaults, weights are decrypted directly into RAM via AES-256-GCM
        without ever touching persistent storage.
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        # Check if file is an encrypted vault (.avcs or starting with AVCS magic)
        is_vault = weights_path.suffix.lower() == ".avcs"
        if not is_vault:
            with open(weights_path, "rb") as f:
                header = f.read(5)
            is_vault = (header == b"AVCS\x01")

        if is_vault:
            from avc.vault import InvalidKeyError, ModelVault
            key = (
                api_key
                or os.environ.get("AVC_API_KEY")
                or os.environ.get("AVCS_API_KEY")
                or os.environ.get("AVCS_KEY")
            )
            if not key:
                raise InvalidKeyError(
                    "Authentication Required: Encrypted model vault (.avcs) requires an API key. "
                    "Set the AVC_API_KEY environment variable or pass api_key='...' to load_weights()."
                )
            state = ModelVault.decrypt_to_ram(weights_path, api_key=key, map_location="cpu")
        else:
            state = torch.load(weights_path, map_location="cpu", weights_only=False)

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        self.load_state_dict(state, strict=strict)
