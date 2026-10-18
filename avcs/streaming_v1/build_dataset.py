"""Create the small deterministic synthetic sequence corpus for pipeline tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avcs.streaming_v1.data import generate_procedural_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--train-recordings", type=int, default=12)
    parser.add_argument("--validation-recordings", type=int, default=4)
    parser.add_argument("--test-recordings", type=int, default=4)
    parser.add_argument("--frames-per-recording", type=int, default=24)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    result = generate_procedural_sequences(
        args.output,
        seed=args.seed,
        train_recordings=args.train_recordings,
        validation_recordings=args.validation_recordings,
        test_recordings=args.test_recordings,
        frames_per_recording=args.frames_per_recording,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
