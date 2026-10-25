"""Fully-learned cortical heads: no hardcoded vocabularies, thresholds, or rules.

Every decision the cortex makes at inference time is produced by a trained
tensor in this module. The pieces here replace the fixed logic that previously
lived in the hot path:

- `LearnedTextEncoder` replaces the parameter-free hashed n-gram embedding, so
  the language->vision alignment is learned end to end.
- `SlotDetector` replaces greedy 2D non-maximum suppression over a dense
  affordance map. Object locations and counts come from a fixed set of learned
  query slots (DETR-style), so nothing is thresholded by hand.
- `SceneUnderstandingHead` produces a multi-label description of the scene and a
  regressed object count, so "what is on the screen" is read from learned
  logits rather than a closed category list alone.
- `ActionPolicyHead` produces a distribution over actions and an action target,
  so the model chooses what to do instead of the controller hardcoding it.

The classes are deliberately small: the whole point is to stay inside the
real-time CPU budget (see docs/REALITY_AUDIT.md). None of them add per-frame
Python branching; each is a pure tensor computation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Action space. The names are only used for reporting; the decision itself is
# the argmax over `ActionPolicyHead` logits, which is learned.
ACTION_NAMES = ["noop", "move", "click", "type", "scroll", "drag"]

# Scene attribute space (multi-label). Again, the values are learned logits;
# these strings only label the output dimensions.
SCENE_ATTRIBUTES = [
    "has_text",
    "has_toolbar",
    "has_modal",
    "has_input_field",
    "has_list",
    "has_image",
    "is_dark_theme",
    "is_dense",
    "has_cursor",
    "has_animation",
]


class LearnedTextEncoder(nn.Module):
    """Byte-level learned text encoder for language-conditioned grounding.

    A learned embedding over raw UTF-8 bytes followed by an attention pool and a
    projection. There is no vocabulary file and no hash function: the mapping
    from characters to the visual feature space is trained.
    """

    def __init__(self, out_dim: int = 256, embed_dim: int = 64, max_bytes: int = 48) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.max_bytes = max_bytes

        self.byte_embed = nn.Embedding(256, embed_dim)
        self.pos_embed = nn.Embedding(max_bytes, embed_dim)
        self.attn = nn.Linear(embed_dim, 1)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, out_dim),
        )

    def _encode_bytes(self, queries: list[str], device: torch.device) -> torch.Tensor:
        rows = []
        for q in queries:
            raw = (q or "").encode("utf-8")[: self.max_bytes]
            ids = list(raw) if raw else [32]
            rows.append(ids)
        # Pad to the longest byte sequence in the batch so the batch is a single
        # rectangular tensor; padding uses the space byte, matching empty input.
        width = max(len(r) for r in rows)
        padded = [r + [32] * (width - len(r)) for r in rows]
        idx = torch.tensor(padded, dtype=torch.long, device=device)  # (B, L)
        return self.byte_embed(idx)  # (B, L, embed_dim)

    def forward(self, query: str | list[str], device: torch.device | None = None) -> torch.Tensor:
        queries = [query] if isinstance(query, str) else list(query)
        dev = device or next(self.parameters()).device

        emb = self._encode_bytes(queries, dev)  # (B, L, E)
        L = emb.shape[1]
        pos = self.pos_embed(torch.arange(L, device=dev)).unsqueeze(0)
        emb = emb + pos

        # Attention pooling over the byte sequence.
        weights = torch.softmax(self.attn(emb).squeeze(-1), dim=-1)  # (B, L)
        pooled = (emb * weights.unsqueeze(-1)).sum(dim=1)  # (B, E)
        return F.normalize(self.proj(pooled), dim=-1)


class SlotDetector(nn.Module):
    """Learned object-slot detector (DETR-style), replacing greedy NMS.

    A fixed set of learned query slots cross-attends to the fused feature map and
    each slot predicts a normalised box (cx, cy, w, h), an objectness logit, and
    a class distribution. Absent objects are represented by a low objectness
    logit, so no manual threshold decides presence at inference.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        num_slots: int = 16,
        num_classes: int = 5,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_classes = num_classes

        self.slot_embed = nn.Embedding(num_slots, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.norm = nn.LayerNorm(feature_dim)

        self.box_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 4),
            nn.Sigmoid(),
        )
        self.objectness_head = nn.Linear(feature_dim, 1)
        self.class_head = nn.Linear(feature_dim, num_classes)

    def forward(self, fused_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        """fused_feat: (B, C, H, W). Returns per-slot boxes, objectness, class logits."""
        B, C, H, W = fused_feat.shape
        tokens = fused_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)

        q = self.query_proj(self.slot_embed.weight).unsqueeze(0).expand(B, -1, -1)
        k = self.key_proj(tokens)
        v = self.value_proj(tokens)

        # Single-head-per-group attention over the spatial tokens.
        Bq, Nq, _ = q.shape
        qh = q.view(Bq, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        kh = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        vh = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax(qh @ kh.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)
        attended = (attn @ vh).transpose(1, 2).reshape(B, Nq, C)
        slots = self.norm(q + attended)

        boxes = self.box_head(slots)                        # (B, N, 4)
        objectness = self.objectness_head(slots).squeeze(-1)  # (B, N)
        class_logits = self.class_head(slots)               # (B, N, num_classes)
        return {"boxes": boxes, "objectness": objectness, "class_logits": class_logits}


class SceneUnderstandingHead(nn.Module):
    """Learned multi-label scene description and object counting.

    Replaces the single hard category label with a learned attribute vector plus
    a regressed count, so the description of a scene is read off trained logits.
    """

    def __init__(self, latent_dim: int = 256, num_attributes: int = len(SCENE_ATTRIBUTES)) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.attribute_head = nn.Linear(latent_dim, num_attributes)
        self.count_head = nn.Linear(latent_dim, 1)

    def forward(self, scene_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(scene_latent)
        return {
            "attribute_logits": self.attribute_head(h),      # (B, num_attributes)
            "count": F.softplus(self.count_head(h)).squeeze(-1),  # (B,) >= 0
        }


class ActionPolicyHead(nn.Module):
    """Learned action selection and action target prediction.

    Given the scene latent and the (optional) language query embedding, the model
    emits a distribution over actions and a normalised action target. Nothing
    downstream hardcodes which action to take.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        text_dim: int = 256,
        num_actions: int = len(ACTION_NAMES),
    ) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(latent_dim + text_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.action_head = nn.Linear(latent_dim, num_actions)
        self.target_head = nn.Sequential(nn.Linear(latent_dim, 2), nn.Sigmoid())
        self.confidence_head = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())

    def forward(
        self,
        scene_latent: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h = self.fuse(torch.cat([scene_latent, query_embedding], dim=-1))
        return {
            "action_logits": self.action_head(h),
            "action_target": self.target_head(h),
            "action_confidence": self.confidence_head(h).squeeze(-1),
        }


class LearnedStabilityHead(nn.Module):
    """Learned temporal stability classifier with learned hysteresis.

    The previous implementation decoded stability with hand-tuned Schmitt-trigger
    thresholds (0.55 / 0.22 / 0.08). Here the transition behaviour is a learned
    function of the surprise signal and the previous state embedding, so the
    hysteresis curve is fit from data rather than written by hand.
    """

    def __init__(self, latent_dim: int = 256, num_states: int = 4) -> None:
        super().__init__()
        self.state_embed = nn.Embedding(num_states, 32)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1 + 32, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, num_states),
        )

    def forward(
        self,
        memory: torch.Tensor,
        surprise: torch.Tensor,
        prev_state_idx: torch.Tensor,
    ) -> torch.Tensor:
        prev_emb = self.state_embed(prev_state_idx)
        x = torch.cat([memory, surprise, prev_emb], dim=-1)
        return self.net(x)
