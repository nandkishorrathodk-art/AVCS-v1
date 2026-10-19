"""Multi-Task Loss Functions for AVCS Training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.vision.cortex.config import TrainingConfig


class AVCSMultiTaskLoss(nn.Module):
    """Combines self-supervised temporal prediction, spatial affordance,

    semantic UI category, and language-guided coordinate grounding.
    """

    def __init__(self, cfg: TrainingConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TrainingConfig()

        self.ce_loss = nn.CrossEntropyLoss()
        self.coord_loss = nn.SmoothL1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        # Set-matching detection loss: objectness is a focal-style BCE and the
        # box/class terms are applied only to the slot matched to each target.
        self.mse_loss = nn.MSELoss()

    def _slot_detection_loss(
        self,
        slots: dict[str, torch.Tensor],
        target_boxes: torch.Tensor,
        target_classes: torch.Tensor,
        target_present: torch.Tensor,
    ) -> torch.Tensor:
        """Learned set-prediction loss over object slots.

        target_boxes: (B, K, 4) normalised (cx, cy, w, h)
        target_classes: (B, K) class index
        target_present: (B, K) 1.0 for a real object, 0.0 for padding
        """
        obj_logits = slots["objectness"]           # (B, N)
        boxes = slots["boxes"]                     # (B, N, 4)
        cls_logits = slots["class_logits"]         # (B, N, C)
        B, N, _ = boxes.shape
        K = target_boxes.shape[1]

        # Objectness: every slot is a positive if matched to a real target.
        obj_target = torch.zeros_like(obj_logits)
        matched_boxes = torch.zeros_like(boxes)
        matched_cls = torch.zeros(B, N, dtype=torch.long, device=boxes.device)

        # Greedy matching by L1 box distance (cheap, deterministic, CPU-friendly).
        for b in range(B):
            if K == 0:
                continue
            cost = torch.cdist(boxes[b], target_boxes[b])  # (N, K)
            for k in range(K):
                if target_present[b, k] < 0.5:
                    continue
                # Only match slots not already claimed by an earlier target.
                avail = (obj_target[b] < 0.5).nonzero(as_tuple=False).flatten()
                if len(avail) == 0:
                    break
                sub = cost[avail, k]
                pick = avail[int(torch.argmin(sub).item())]
                obj_target[b, pick] = 1.0
                matched_boxes[b, pick] = target_boxes[b, k]
                matched_cls[b, pick] = target_classes[b, k].long()

        loss_obj = self.bce_loss(obj_logits, obj_target)
        pos = obj_target > 0.5
        if pos.any():
            loss_box = self.mse_loss(boxes[pos], matched_boxes[pos])
            loss_cls = self.ce_loss(cls_logits[pos], matched_cls[pos])
        else:
            loss_box = torch.zeros((), device=boxes.device)
            loss_cls = torch.zeros((), device=boxes.device)
        return loss_obj + loss_box + loss_cls

    def forward(
        self,
        model_outputs_t: dict[str, torch.Tensor],
        latent_next: torch.Tensor,
        affordance_targets: torch.Tensor,
        category_targets: torch.Tensor,
        coord_targets: torch.Tensor,
        object_boxes: torch.Tensor | None = None,
        object_classes: torch.Tensor | None = None,
        object_present: torch.Tensor | None = None,
        attribute_targets: torch.Tensor | None = None,
        count_targets: torch.Tensor | None = None,
        action_targets: torch.Tensor | None = None,
        action_coord_targets: torch.Tensor | None = None,
        stability_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all task losses.

        Parameters
        ----------
        model_outputs_t: dict from model.forward(...) on frame_t
        latent_next: (B, latent_dim) observed latent of frame_{t+1}
        affordance_targets: (B, H, W) ground-truth affordance class indices
        category_targets: (B,) ground-truth UI category class indices
        coord_targets: (B, 2) ground-truth target coordinates

        The remaining arguments drive the fully-learned heads. When they are
        omitted the corresponding loss terms are skipped, so existing callers
        keep working unchanged.
        """
        # 1. Self-Supervised Predictive World-Model Loss
        # Encourages temporal_memory to predict next frame's latent
        pred_memory = model_outputs_t["temporal_memory"]
        norm_pred = F.normalize(pred_memory, dim=-1)
        norm_next = F.normalize(latent_next.detach(), dim=-1)
        loss_predictive = (1.0 - (norm_pred * norm_next).sum(dim=-1)).mean()

        # 2. Dorsal Affordance Loss
        # Logits shape: (B, C, H, W), Targets: (B, H, W). The target grid is
        # resampled to the logits' spatial size so the loss is independent of the
        # input resolution the model was configured with.
        affordance_logits = model_outputs_t["affordance_logits"]
        _, _, Hl, Wl = affordance_logits.shape
        if affordance_targets.shape[-2:] != (Hl, Wl):
            tgt = affordance_targets.unsqueeze(1).float()
            tgt = F.interpolate(tgt, size=(Hl, Wl), mode="nearest").squeeze(1).long()
        else:
            tgt = affordance_targets
        loss_affordance = self.ce_loss(affordance_logits, tgt)

        # 3. Ventral UI Category Loss
        cat_logits = model_outputs_t["category_logits"]
        loss_category = self.ce_loss(cat_logits, category_targets)

        # 4. Language Grounding Coordinate Regression Loss
        query_coords = model_outputs_t["query_coords"]
        if query_coords is not None:
            loss_grounding = self.coord_loss(query_coords, coord_targets)
        else:
            loss_grounding = torch.tensor(0.0, device=affordance_logits.device)

        # 5. Fully-learned head losses (optional, added when targets are present)
        device = affordance_logits.device
        zero = torch.zeros((), device=device)

        if object_boxes is not None and object_present is not None:
            loss_slots = self._slot_detection_loss(
                model_outputs_t["slots"], object_boxes, object_classes, object_present
            )
        else:
            loss_slots = zero

        if attribute_targets is not None:
            loss_attributes = self.bce_loss(
                model_outputs_t["attribute_logits"], attribute_targets
            )
        else:
            loss_attributes = zero

        if count_targets is not None:
            loss_count = self.coord_loss(
                model_outputs_t["object_count"], count_targets.float()
            )
        else:
            loss_count = zero

        if action_targets is not None:
            loss_action = self.ce_loss(model_outputs_t["action_logits"], action_targets)
            if action_coord_targets is not None:
                loss_action = loss_action + self.coord_loss(
                    model_outputs_t["action_target"], action_coord_targets
                )
        else:
            loss_action = zero

        if stability_targets is not None:
            loss_stability = self.ce_loss(
                model_outputs_t["stability_logits_learned"], stability_targets
            )
        else:
            loss_stability = zero

        # Total Weighted Multi-Task Loss
        total_loss = (
            self.cfg.predictive_loss_weight * loss_predictive
            + self.cfg.affordance_loss_weight * loss_affordance
            + self.cfg.category_loss_weight * loss_category
            + self.cfg.grounding_loss_weight * loss_grounding
            + self.cfg.slot_loss_weight * loss_slots
            + self.cfg.attribute_loss_weight * loss_attributes
            + self.cfg.count_loss_weight * loss_count
            + self.cfg.action_loss_weight * loss_action
            + self.cfg.stability_loss_weight * loss_stability
        )

        return {
            "loss": total_loss,
            "loss_predictive": loss_predictive,
            "loss_affordance": loss_affordance,
            "loss_category": loss_category,
            "loss_grounding": loss_grounding,
            "loss_slots": loss_slots,
            "loss_attributes": loss_attributes,
            "loss_count": loss_count,
            "loss_action": loss_action,
            "loss_stability": loss_stability,
        }

