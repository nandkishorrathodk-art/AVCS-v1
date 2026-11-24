"""Legacy shim for avcs.ace.skills_vault.cache.

Re-exports AnchorVerifiedSkillCache and SkillCache from canonical avcs.runtime.skills_vault.cache package.
"""
from __future__ import annotations

import avcs.runtime.skills_vault.cache as _cache
from avcs.runtime.skills_vault.cache import *  # noqa: F401, F403
from avcs.runtime.skills_vault.cache import (
    AnchorVerifiedSkillCache,
    SkillCache,
    SkillCacheEntry,
    SkillLookupResult,
    SkillRecord,
    UIStateAnchor,
    compute_visual_hash,
    visual_hash_similarity,
)

__all__ = [
    "AnchorVerifiedSkillCache",
    "SkillCache",
    "SkillCacheEntry",
    "SkillLookupResult",
    "SkillRecord",
    "UIStateAnchor",
    "compute_visual_hash",
    "visual_hash_similarity",
]


def __getattr__(name: str):
    return getattr(_cache, name)
