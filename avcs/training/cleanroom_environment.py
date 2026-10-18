"""Small helpers for checking reproducible clean-room dependency environments."""
from __future__ import annotations
from __future__ import annotations

import re
from typing import Mapping


def compare_dependency_versions(
    expected: Mapping[str, str],
    observed: Mapping[str, str],
) -> dict[str, dict[str, str | None]]:
    """Return pinned dependency versions that are missing or differ."""
    def normalize(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    observed_normalized = {normalize(name): version for name, version in observed.items()}
    mismatches: dict[str, dict[str, str | None]] = {}
    for name, expected_version in expected.items():
        actual_version = observed_normalized.get(normalize(name))
        if actual_version != expected_version:
            mismatches[name] = {
                "expected": expected_version,
                "observed": actual_version,
            }
    return mismatches
