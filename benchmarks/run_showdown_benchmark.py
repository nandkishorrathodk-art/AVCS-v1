"""Legacy shim for benchmarks.run_showdown_benchmark.

Re-exports showdown benchmark runner from canonical avcs.eval package.
"""
from __future__ import annotations

from avcs.eval.showdown import *  # noqa: F401, F403
from avcs.eval.showdown import (
    BenchmarkSample,
    evaluate_sample,
    generate_synthetic_showdown_suite,
    main,
    run_benchmark,
    wilson_score_interval,
)

if __name__ == "__main__":
    main()
