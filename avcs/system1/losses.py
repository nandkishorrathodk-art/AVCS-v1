"""One loss for the one model.

All tasks are trained jointly from the single latent. There is no per-task
training loop and no task gets its own optimizer; the gradients meet in the
trunk, which is what makes the trunk learn a representation that serves all of
them at once.

Loss terms:

* category      - cross entropy over scene category
* objectness    - binary cross entropy, is there an element in this cell
* kind          - cross entropy over element kind, on occupied cells only
* word          - cross entropy over UI word, on occupied cells only
* action        - cross entropy over the action the scene affords
* point         - cross entropy over cells for the query's target
* presence      - binary cross entropy, does the query's target exist

`point` and `presence` are the query-conditioned terms; the rest are the
frame-level terms. Both come out of the same forward pass.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from avcs.system1.model import OneModelOutput


@dataclasses.dataclass
class LossBreakdown:
    total: float
    category: float
    objectness: float
    kind: float
    word: float
    action: float
    point: float
    presence: float

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


class OneModelLoss(nn.Module):
    def __init__(
        self,
        w_category: float = 1.0,
        w_objectness: float = 1.0,
        w_kind: float = 1.0,
        w_word: float = 1.0,
        w_action: float = 0.5,
        w_point: float = 2.0,
        w_presence: float = 1.0,
    ) -> None:
        super().__init__()
        self.w = dict(
            category=w_category, objectness=w_objectness, kind=w_kind, word=w_word,
            action=w_action, point=w_point, presence=w_presence,
        )

    def forward(self, out: OneModelOutput, batch: dict) -> tuple[torch.Tensor, LossBreakdown]:
        device = out.latent.device

        cat = F.cross_entropy(out.category_logits, batch["category"])
        obj = F.binary_cross_entropy_with_logits(out.objectness_logits, batch["objectness"])

        occupied = batch["objectness"] > 0.5
        if occupied.any():
            kind = F.cross_entropy(out.kind_logits[occupied], batch["kind"][occupied])
            word = F.cross_entropy(out.word_logits[occupied], batch["word"][occupied])
        else:
            kind = torch.zeros((), device=device)
            word = torch.zeros((), device=device)

        act = F.cross_entropy(out.action_logits, batch["action"])

        # Query-conditioned terms. Point loss is only defined where a target exists.
        present = batch["present"] > 0.5
        pres = F.binary_cross_entropy_with_logits(out.presence_logits.squeeze(-1), batch["present"])
        if present.any():
            point = F.cross_entropy(out.point_logits[present], batch["point_cell"][present])
        else:
            point = torch.zeros((), device=device)

        total = (
            self.w["category"] * cat
            + self.w["objectness"] * obj
            + self.w["kind"] * kind
            + self.w["word"] * word
            + self.w["action"] * act
            + self.w["point"] * point
            + self.w["presence"] * pres
        )
        breakdown = LossBreakdown(
            total=float(total.detach()),
            category=float(cat.detach()),
            objectness=float(obj.detach()),
            kind=float(kind.detach()),
            word=float(word.detach()),
            action=float(act.detach()),
            point=float(point.detach()),
            presence=float(pres.detach()),
        )
        return total, breakdown
