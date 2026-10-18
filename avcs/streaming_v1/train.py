"""Train a manifest-approved StreamingVisualCortexV1 candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avcs.streaming_v1.training import train_streaming_v1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--trust-config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--change-positive-weight", type=float, default=1.0)
    parser.add_argument(
        "--temporal-mode",
        choices=("spatial", "convgru", "convgru_delta"),
        default="convgru",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train_streaming_v1(
        args.dataset,
        args.output,
        approval_path=args.approval,
        trust_config_path=args.trust_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        patience=args.patience,
        change_positive_weight=args.change_positive_weight,
        model_config={"temporal_mode": args.temporal_mode},
        resume=args.resume,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
