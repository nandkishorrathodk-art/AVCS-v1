import sys
from pathlib import Path

# Ensure repo root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import avcs
import avcs.perception.human_eye
from avcs.human_eye import HumanEye

eye = HumanEye(load_weights=False)
print("eye class:", eye.__class__.__module__, eye.__class__.__name__)
print("eye.grounder class:", eye.grounder.__class__.__module__, eye.grounder.__class__.__name__)
print("hasattr predict_point:", hasattr(eye, "predict_point"))
print("hasattr locate:", hasattr(eye, "locate"))

