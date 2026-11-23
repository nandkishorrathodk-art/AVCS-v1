# Department: Eval (`avcs.eval`)

## 1. Responsibilities & Boundaries
- Rigorous benchmarking, offline trajectory evaluation, and statistical validation.
- Showdown-Clicks benchmark runner (`run_benchmark`, `evaluate_sample`, `generate_synthetic_showdown_suite`).
- Wilson score interval calculation (exact 95% confidence intervals).
- Adversarial challenger suites (`AdversarialChallengerM2`) and latency benchmarking.

## 2. Allowed Imports
- Standard Library (`math`, `statistics`, `dataclasses`, `typing`, `time`, `json`, `pathlib`).
- `avcs.protocol` (for action and coordinate evaluations).
- `avcs.models`, `avcs.data`, `avcs.runtime` (for evaluating model accuracy and agent execution).

## 3. Forbidden Imports
- Unsandboxed physical OS clicks (all evaluation actuation must be mocked or executed in dry-run mode).
- Training loop logic (`avcs.train`).

## 4. Sub-Agent Contract
- Provides deterministic grading suites to evaluate agent accuracy, Wilson confidence bounds, and P95 latency compliance against the system SLA.
