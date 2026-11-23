"""AVCS Eval Department.

Showdown-Clicks benchmark runner, Wilson score intervals,
adversarial challenger suites, and trajectory evaluations.
"""
from __future__ import annotations

from avcs.eval.challenger import (
    challenge_1_negative_screens,
    challenge_2_adversarial_queries,
    challenge_3_spatial_collapse_and_degeneracy,
    generate_200_adversarial_queries,
)
from avcs.eval.showdown import (
    BenchmarkSample,
    evaluate_sample,
    generate_synthetic_showdown_suite,
    run_benchmark,
    wilson_score_interval,
)

__all__ = [
    "BenchmarkSample",
    "challenge_1_negative_screens",
    "challenge_2_adversarial_queries",
    "challenge_3_spatial_collapse_and_degeneracy",
    "evaluate_sample",
    "generate_200_adversarial_queries",
    "generate_synthetic_showdown_suite",
    "run_benchmark",
    "wilson_score_interval",
]
