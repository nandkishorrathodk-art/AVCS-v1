"""Generate a clean-room System 1 corpus at a chosen scale.

Wraps the repo's own generator so a large corpus can be produced reproducibly
for a GPU training run, without hand-editing the generator's defaults.

    python3 -m avcs.system1.make_corpus --out /tmp/corpus --train 12000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--train", type=int, default=12000)
    ap.add_argument("--validation", type=int, default=600)
    ap.add_argument("--heldout", type=int, default=600)
    ap.add_argument("--negative", type=int, default=600)
    a = ap.parse_args()

    from avcs.dataset.cleanroom_v1.generator import generate_dataset

    counts = {
        "train": a.train,
        "validation": a.validation,
        "test_heldout": a.heldout,
        "test_negative": a.negative,
    }
    t0 = time.perf_counter()
    report = generate_dataset(output_dir=Path(a.out), seed=a.seed, counts=counts)
    dt = time.perf_counter() - t0
    total = sum(counts.values())
    print(f"generated {total} frames in {dt:.1f}s ({total / dt:.0f} frames/s)")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, (list, dict))}, indent=2))


if __name__ == "__main__":
    main()
