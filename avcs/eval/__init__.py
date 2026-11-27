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
from avcs.eval.osworld_runner import (
    DeterministicPureRGBMockEnv,
    OSWorldObservationContract,
    OSWorldTask,
    generate_osworld_smoke_tasks,
    load_osworld_tasks,
    run_osworld_eval,
)
from avcs.eval.phase1_gate import run_phase1_gate, verify_mock_agent_lifecycle, verify_protocol_roundtrip
from avcs.eval.screenspot_pro import (
    ScreenSpotProSample,
    evaluate_screenspot_sample,
    generate_synthetic_screenspot_pro_suite,
    load_screenspot_pro,
    run_screenspot_pro,
)
from avcs.eval.showdown import (
    BenchmarkSample,
    bootstrap_confidence_interval,
    evaluate_sample,
    generate_synthetic_showdown_suite,
    load_showdown_clicks_dev,
    run_benchmark,
    wilson_score_interval,
)

__all__ = [
    "BenchmarkSample",
    "DeterministicPureRGBMockEnv",
    "OSWorldObservationContract",
    "OSWorldTask",
    "ScreenSpotProSample",
    "bootstrap_confidence_interval",
    "challenge_1_negative_screens",
    "challenge_2_adversarial_queries",
    "challenge_3_spatial_collapse_and_degeneracy",
    "evaluate_sample",
    "evaluate_screenspot_sample",
    "generate_200_adversarial_queries",
    "generate_osworld_smoke_tasks",
    "generate_synthetic_screenspot_pro_suite",
    "generate_synthetic_showdown_suite",
    "load_osworld_tasks",
    "load_screenspot_pro",
    "load_showdown_clicks_dev",
    "run_benchmark",
    "run_osworld_eval",
    "run_phase1_gate",
    "run_screenspot_pro",
    "verify_mock_agent_lifecycle",
    "verify_protocol_roundtrip",
    "wilson_score_interval",
]
