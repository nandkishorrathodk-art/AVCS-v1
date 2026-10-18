"""Evaluate StreamingVisualCortexV1 on source-held-out sequence recordings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avcs.streaming_v1.training import evaluate_streaming_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--change-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_streaming_checkpoint(
        args.checkpoint,
        args.dataset,
        split=args.split,
        change_threshold=args.change_threshold,
        report_path=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
