"""The shared identity space: pixels and words meet in the middle.

This is the part that makes the stream *joint vision and language* rather than a
pixel matcher with a lookup table bolted on.

Two branches, one space, one learned join:

    patch (pixels)  ->  PatchEncoder   --\
                                          >-- z in R^D, L2-normalised
    word  (language) ->  TextEncoder   --/

The vision branch never sees a character. The language branch never sees a
pixel. Neither branch is OCR: the vision branch produces a *shape identity*, and
the language branch produces the *identity of a spoken word*. Training pulls the
two embeddings of the same control together (symmetric InfoNCE), so the
correspondence between "what this patch looks like" and "what this control is
called" is learned from data, not hardcoded.

Because the space is shared and symmetric, both directions fall out of it:

    forward  : word "save"  -> nearest patch  -> where to click
    inverse  : patch        -> nearest word   -> what this control is

Nothing else in ``avc`` does the inverse direction, and nothing else has a
shared pixel/word space at all: ``ventral`` labels a whole scene's category,
``dorsal`` labels a region's affordance, and ``LearnedTextEncoder`` maps text to
an embedding for *conditioning* — it has no pixel side to meet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# The character table the *language* branch uses. It is only ever fed strings
# that came from speech or from the model, never from an image.
_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789_ -"
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(_CHARS)}   # 0 is reserved for unknown
NUM_CHARS = len(_CHARS) + 1

EMBED_DIM = 128
PATCH_H = 32
PATCH_W = 96


@dataclass
class Readout:
    """What the shared space says about one patch (or one word)."""

    label: str
    index: int
    confidence: float
    margin: float
    known: bool
    embedding: Tuple[float, ...] = ()

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "index": self.index,
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "known": self.known,
        }


class PatchEncoder(nn.Module):
    """Grayscale word patch -> identity embedding. Reads shapes, not letters."""

    def __init__(self, embed_dim: int = EMBED_DIM, width: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, stride=2, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1), nn.BatchNorm2d(width * 4), nn.ReLU(),
        )
        self.proj = nn.Linear(width * 4, embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """``patches``: (B, 1, H, W) or (B, H, W). Returns L2-normalised (B, D)."""
        if patches.dim() == 3:
            patches = patches.unsqueeze(1)
        feats = self.features(patches)
        pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        return F.normalize(self.proj(pooled), dim=-1)


class TextEncoder(nn.Module):
    """Word -> identity embedding. The word is language input, never an image."""

    def __init__(self, embed_dim: int = EMBED_DIM, char_dim: int = 32) -> None:
        super().__init__()
        self.char_embed = nn.Embedding(NUM_CHARS, char_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(char_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim),
        )

    @staticmethod
    def encode_word(word: str) -> torch.Tensor:
        ids = [CHAR_TO_ID.get(c, 0) for c in word.strip().lower()] or [0]
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, words) -> torch.Tensor:
        """``words``: list[str] or (B, L) long tensor. Returns L2-normalised (B, D)."""
        if isinstance(words, (list, tuple)):
            if not words:
                raise ValueError("TextEncoder needs at least one word")
            length = max(len(self.encode_word(w)) for w in words)
            batch = torch.zeros(len(words), length, dtype=torch.long)
            for i, w in enumerate(words):
                ids = self.encode_word(w)
                batch[i, :ids.numel()] = ids
            words = batch
        embedded = self.char_embed(words)
        pooled = embedded.mean(dim=1)
        return F.normalize(self.proj(pooled), dim=-1)


class LexiconEncoder(nn.Module):
    """Both branches plus the prototypes that define the closed vocabulary."""

    def __init__(self, labels, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        if not labels:
            raise ValueError("LexiconEncoder needs a non-empty label list")
        self.labels = list(labels)
        self.embed_dim = embed_dim
        self.patch = PatchEncoder(embed_dim)
        self.text = TextEncoder(embed_dim)
        # Prototypes are buffers: they must travel with the weights, like the
        # tokenizer does in the text side of this repo.
        self.register_buffer("prototypes", torch.zeros(len(self.labels), embed_dim))
        self.register_buffer("prototypes_ready", torch.zeros(1, dtype=torch.bool))

    # ------------------------------------------------------------------ #
    # Prototypes
    # ------------------------------------------------------------------ #
    def rebuild_prototypes(self) -> None:
        """Set each label's prototype to the mean of its text embedding.

        The text side is deterministic for a given vocabulary, so the prototypes
        can be derived rather than learned separately. This keeps one source of
        truth: the label list.
        """
        with torch.no_grad():
            emb = self.text(self.labels)
            self.prototypes.copy_(emb)
            self.prototypes_ready.fill_(True)

    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        return self.patch(patches)

    def encode_words(self, words) -> torch.Tensor:
        return self.text(words)

    # ------------------------------------------------------------------ #
    # The readout both directions share
    # ------------------------------------------------------------------ #
    def readout_from_embedding(self, embedding: torch.Tensor, floor: float = 0.55) -> Readout:
        """Nearest prototype, with a margin gate. Fail closed below the floor."""
        if not bool(self.prototypes_ready.item()):
            raise RuntimeError(
                "prototypes are not built; call rebuild_prototypes() or load a "
                "trained checkpoint before reading"
            )
        embedding = F.normalize(embedding.reshape(-1), dim=-1)
        sims = self.prototypes @ embedding
        top2 = torch.topk(sims, k=min(2, sims.numel()))
        index = int(top2.indices[0])
        confidence = float(top2.values[0])
        margin = confidence - float(top2.values[1]) if top2.values.numel() > 1 else confidence
        return Readout(
            label=self.labels[index],
            index=index,
            confidence=confidence,
            margin=margin,
            known=confidence >= floor,
            embedding=tuple(float(v) for v in embedding.tolist()),
        )

    def read_patch(self, patch: torch.Tensor, floor: float = 0.55) -> Readout:
        return self.readout_from_embedding(self.patch(patch), floor=floor)
