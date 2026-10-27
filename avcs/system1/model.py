"""The one model.

Design rules, held to strictly:

1. **One trunk.** Every task reads from the same feature map. No task has its own
   private front-end that re-derives what the trunk already knows.
2. **One latent.** ``z`` is the single internal state. It is produced once per
   frame and every head consumes it.
3. **No hand-written decisions.** Nothing in ``forward`` branches on content.
   The model returns distributions; the caller may read an argmax, but the model
   itself never says "if category == modal then ...".
4. **Text is an input, not a lookup.** The query is embedded by a small learned
   encoder and mixed into ``z`` by attention, so "click Save" and "click Stop"
   give different latents on the same frame.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Label vocabularies. These are the *things the model predicts*, i.e. the
# columns of its output distributions. Changing them means retraining.
# --------------------------------------------------------------------------- #
CATEGORY_NAMES = (
    "desktop_canvas",
    "editor_code",
    "browser_webpage",
    "terminal_shell",
    "modal_dialog",
    "form_input",
    "media_playback",
    "settings_panel",
    "unknown",
)
ELEMENT_KINDS = ("button", "input_field", "icon", "tab", "link", "text_label", "container")
WORD_NAMES = (
    "<none>", "Back", "Bookmarks", "Cancel", "Edit", "Exit", "File", "Menu",
    "Notes", "OK", "Pause", "Play", "Save", "Search", "Settings", "Start",
    "Stop", "Tools", "View",
)
ACTION_NAMES = ("click", "type", "scroll", "wait", "none")


@dataclasses.dataclass
class OneModelConfig:
    input_size: int = 128
    width: int = 64
    latent_dim: int = 128
    grid: int = 16
    query_vocab: int = 4096
    query_dim: int = 64
    query_len: int = 32


@dataclasses.dataclass
class OneModelOutput:
    """Everything the one model has to say about a frame, all from one latent."""

    latent: torch.Tensor                       # (B, latent_dim) the single state
    category_logits: torch.Tensor              # (B, n_categories)
    kind_logits: torch.Tensor                  # (B, grid*grid, n_kinds)
    objectness_logits: torch.Tensor            # (B, grid*grid)
    word_logits: torch.Tensor                  # (B, grid*grid, n_words)
    action_logits: torch.Tensor                # (B, n_actions)
    point_logits: torch.Tensor                 # (B, grid*grid) query-conditioned
    presence_logits: torch.Tensor              # (B, 1) query-conditioned presence

    def category_index(self) -> torch.Tensor:
        return self.category_logits.argmax(dim=-1)

    def category_name(self) -> list[str]:
        return [CATEGORY_NAMES[i] for i in self.category_index().tolist()]

    def point_norm(self) -> torch.Tensor:
        """Soft-argmax of the grounding heatmap as normalized (x, y) in [0, 1].

        The softmax is over the *flattened* grid, i.e. one distribution over
        cells. Normalizing along a single axis (as this did) leaves the other
        axis' marginal constant, so that coordinate collapses to 0.5 by
        construction -- a sharp head then reads as a dead one.
        """
        b, n = self.point_logits.shape
        g = int(round(n ** 0.5))
        p = F.softmax(self.point_logits, dim=-1).reshape(b, g, g)
        ys = torch.linspace(0.0, 1.0, g, device=p.device)
        x = (p.sum(dim=1) * ys).sum(dim=-1)
        y = (p.sum(dim=2) * ys).sum(dim=-1)
        return torch.stack([x, y], dim=-1)


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _Trunk(nn.Module):
    """The shared visual trunk. Its output is the only thing the heads see."""

    def __init__(self, width: int) -> None:
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 4
        self.stem = _ConvBlock(3, c1, stride=2)       # 128 -> 64
        self.stage1 = _ConvBlock(c1, c1)
        self.stage2 = _ConvBlock(c1, c2, stride=2)    # 64 -> 32
        self.stage3 = _ConvBlock(c2, c2)
        self.stage4 = _ConvBlock(c2, c3, stride=2)    # 32 -> 16
        self.stage5 = _ConvBlock(c3, c4)
        self.out_dim = c4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x


class _QueryEncoder(nn.Module):
    """Learned text encoder.

    Word -> id uses a *stable* hash. Python's built-in ``hash()`` on str is
    salted per process, so the id a word gets at training time would differ from
    the id it gets in a later inference process, and the grounding head would be
    reading a different token for the same word. A fixed FNV-1a keeps the id
    reproducible across processes, which is what lets a checkpoint transfer.
    """

    def __init__(self, vocab: int, dim: int, max_len: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.embed = nn.Embedding(vocab, dim, padding_idx=0)
        self.proj = nn.Linear(dim, dim)

    @staticmethod
    def word_id(word: str, vocab: int) -> int:
        h = 2166136261
        for ch in word.lower().encode("utf-8"):
            h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
        return (h % (vocab - 1)) + 1

    def tokenize(self, texts: list[str], device: torch.device) -> torch.Tensor:
        out = torch.zeros(len(texts), self.max_len, dtype=torch.long, device=device)
        for i, t in enumerate(texts):
            ids = [self.word_id(w, self.vocab) for w in t.split()][: self.max_len]
            if ids:
                out[i, : len(ids)] = torch.tensor(ids, device=device)
        return out

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        ids = self.tokenize(texts, device)
        mask = (ids != 0).unsqueeze(-1).float()
        emb = self.proj(self.embed(ids)) * mask
        summed = emb.sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom


class OneModel(nn.Module):
    """One trunk, one latent, every task."""

    def __init__(self, config: Optional[OneModelConfig] = None) -> None:
        super().__init__()
        self.config = config or OneModelConfig()
        c = self.config
        self.trunk = _Trunk(c.width)
        feat_dim = self.trunk.out_dim

        # ONE latent: the shared state of the model.
        self.latent = nn.Conv2d(feat_dim, c.latent_dim, 1)

        # Shared 1x1 heads: every task reads the same latent map.
        self.category_head = nn.Linear(c.latent_dim, len(CATEGORY_NAMES))
        self.kind_head = nn.Conv2d(c.latent_dim, len(ELEMENT_KINDS), 1)
        self.objectness_head = nn.Conv2d(c.latent_dim, 1, 1)
        self.word_head = nn.Conv2d(c.latent_dim, len(WORD_NAMES), 1)
        self.action_head = nn.Linear(c.latent_dim, len(ACTION_NAMES))

        # Query conditioning: text is mixed into the same latent map.
        self.query_encoder = _QueryEncoder(c.query_vocab, c.query_dim, c.query_len)
        self.query_to_latent = nn.Linear(c.query_dim, c.latent_dim)

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, images: torch.Tensor, queries: Optional[list[str]] = None) -> OneModelOutput:
        b = images.shape[0]
        feat = self.trunk(images)                     # (B, C, 16, 16)
        z_map = self.latent(feat)                     # (B, latent, 16, 16)  <-- THE latent
        z_vec = F.adaptive_avg_pool2d(z_map, 1).flatten(1)   # (B, latent)

        category_logits = self.category_head(z_vec)
        kind_logits = self.kind_head(z_map).flatten(2).transpose(1, 2)        # (B, HW, K)
        objectness_logits = self.objectness_head(z_map).flatten(2).squeeze(1)  # (B, HW)
        word_logits = self.word_head(z_map).flatten(2).transpose(1, 2)        # (B, HW, W)
        action_logits = self.action_head(z_vec)

        # Grounding: condition the same latent on the query, then score cells.
        if queries is None:
            queries = [""] * b
        q = self.query_to_latent(self.query_encoder(queries, images.device))   # (B, latent)
        conditioned = z_map * q.unsqueeze(-1).unsqueeze(-1)
        point_logits = conditioned.sum(dim=1).flatten(1)                       # (B, HW)
        presence_logits = (z_vec * q).sum(dim=-1, keepdim=True)                # (B, 1)

        return OneModelOutput(
            latent=z_vec,
            category_logits=category_logits,
            kind_logits=kind_logits,
            objectness_logits=objectness_logits,
            word_logits=word_logits,
            action_logits=action_logits,
            point_logits=point_logits,
            presence_logits=presence_logits,
        )
