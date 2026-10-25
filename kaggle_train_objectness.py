"""Self-Contained Kaggle GPU Training Script for AVCS UI Objectness Model.

Run on Kaggle GPU:
  python kaggle_train_objectness.py --epochs 25 --batch-size 32 --device cuda
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from avcs.training.train_objectness import train_objectness

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle GPU UI Objectness Training")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    results = train_objectness(
        override_epochs=args.epochs,
        override_batch_size=args.batch_size,
        override_lr=args.lr,
        device_str=args.device,
        seed=args.seed,
    )
    print("[Kaggle] Training complete:", results["metrics"])
