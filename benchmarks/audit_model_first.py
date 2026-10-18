import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
from avcs.human_eye import HumanEye
from avcs.goal_grounder import GoalConditionedGrounder
from benchmarks.run_ui_grounding_benchmark import UIGroundingBenchmarkRunner

print("==================================================================")
print("PRIORITY 1: AUDIT OF EXISTING MODEL ARCHITECTURE AND CHECKPOINTS")
print("==================================================================")

ckpt_path = Path("runs/weights/human_eye_weights.pt")
print(f"human_eye_weights.pt exists: {ckpt_path.exists()} ({ckpt_path.stat().st_size:,} bytes)")

gg_path = Path("runs/weights/goal_grounder_best.pt")
print(f"goal_grounder_best.pt exists: {gg_path.exists()} ({gg_path.stat().st_size:,} bytes)")

# Instantiate HumanEye and inspect loaded model
eye = HumanEye(load_weights=True)
print(f"\nHumanEye loaded:")
print(f"  weights_loaded: {eye.weights_loaded}")
print(f"  grounder present: {hasattr(eye, 'grounder')}")
print(f"  grounder type: {type(eye.grounder).__name__}")
print(f"  grounder.is_trained: {getattr(eye.grounder, 'is_trained', False)}")
if hasattr(eye.grounder, "_checkpoint_metadata"):
    print(f"  grounder._checkpoint_metadata: {eye.grounder._checkpoint_metadata}")

total_params = sum(p.numel() for p in eye.parameters())
grounder_params = sum(p.numel() for p in eye.grounder.parameters())
print(f"\nParameter Count:")
print(f"  Total HumanEye params: {total_params:,}")
print(f"  Grounder params: {grounder_params:,}")

# Run evaluation on test_held_out split
print("\nRunning Evaluation on test_held_out split...")
runner = UIGroundingBenchmarkRunner(device="cpu")
results = runner.evaluate_model(eye, split="test_held_out")
print("\n" + results.summary())

print("\nRunning Evaluation on negative screens (test_negative)...")
# Check how runner evaluates negative screens
neg_screens = [s for s in runner.manifest.get("screens", []) if s.get("split") == "test_negative"]
print(f"Found {len(neg_screens)} negative screens in manifest.")

print("==================================================================")

