"""System 1 as a single model: one trunk, one latent, every task in one pass.

The point of this package is the property the repo keeps losing: **one
network**.  Not a button detector plus a word recogniser plus a stability head
wired together by hand-written rules, but one deep-learning model that builds a
single latent representation of what is on screen and reads every answer out of
that same latent.

Everything the model knows comes from one forward pass:

    pixels -> shared trunk -> ONE latent z
                                  |
             +--------------------+--------------------+------------------+
             |                    |                    |                  |
          category            elements             grounding           words
        "kaun sa app"      "kya kya hai"        "X kahan hai"      "kya likha"

There are no ``if/elif`` decision points anywhere in the inference path.  The
only vocabulary lists (category names, element kinds, UI words) are *labels the
model is trained to predict*, exactly like class names in any classifier --
they are not logic.  If you want to change what the model says, you change the
data and retrain; you do not edit a rule.
"""
from __future__ import annotations

from avcs.system1.model import (
    OneModel,
    OneModelOutput,
    OneModelConfig,
    CATEGORY_NAMES,
    ELEMENT_KINDS,
    WORD_NAMES,
    ACTION_NAMES,
)
from avcs.system1.losses import OneModelLoss, LossBreakdown
from avcs.system1.loop import OneModelLoop, LoopStep, VirtualHand, run_sequence

__all__ = [
    "OneModel",
    "OneModelOutput",
    "OneModelConfig",
    "OneModelLoss",
    "LossBreakdown",
    "OneModelLoop",
    "LoopStep",
    "VirtualHand",
    "run_sequence",
    "CATEGORY_NAMES",
    "ELEMENT_KINDS",
    "WORD_NAMES",
    "ACTION_NAMES",
]
