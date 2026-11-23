"""Legacy shim for benchmarks.adversarial_challenger_m2.

Re-exports adversarial challenger routines from canonical avcs.eval package.
"""
from __future__ import annotations

from avcs.eval.challenger import *  # noqa: F401, F403
from avcs.eval.challenger import (
    challenge_1_negative_screens,
    challenge_2_adversarial_queries,
    challenge_3_spatial_collapse_and_degeneracy,
    generate_200_adversarial_queries,
    main,
)

if __name__ == "__main__":
    main()
