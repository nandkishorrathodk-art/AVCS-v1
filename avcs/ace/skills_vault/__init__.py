"""Legacy shim for avcs.ace.skills_vault.

Re-exports skill cache from canonical avcs.runtime.skills_vault package.
"""
from __future__ import annotations

import avcs.runtime.skills_vault as _skills_vault
from avcs.runtime.skills_vault.cache import (
    AnchorVerifiedSkillCache,
    SkillRecord,
    UIStateAnchor,
)

__all__ = [
    "AnchorVerifiedSkillCache",
    "SkillRecord",
    "UIStateAnchor",
]


def __getattr__(name: str):
    return getattr(_skills_vault, name)
