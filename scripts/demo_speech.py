import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from avcs.observation import VisualObservationEngine

engine = VisualObservationEngine()

# Scenario 1: Code Editor Screen
class EditorState:
    timestamp = 100.0
    frame_index = 1
    primary_category = 'code_editor'
    stability = 'stable'
    surprise_score = 0.02
    has_active_modal = False
    affordance_hotspots = [
        (0.85, 0.05, 'run_button', 0.99),
        (0.10, 0.40, 'file_tab', 0.90),
        (0.50, 0.05, 'search_bar', 0.95),
    ]

f1 = np.full((128, 128, 3), 120, dtype=np.uint8)
f1[10:30, 10:30] = 255
obs1 = engine.generate(EditorState(), frame=f1)
print("=== SCENARIO 1 (Code Editor Khula Hai) ===")
print("Model ki boli:")
print(f"  \"{obs1.scene_description}\"")
print()

# Scenario 2: Save Dialog Pop-up (Surprise & Modal)
class ModalState:
    timestamp = 105.0
    frame_index = 2
    primary_category = 'confirmation_dialog'
    stability = 'modal_active'
    surprise_score = 0.85
    has_active_modal = True
    affordance_hotspots = [
        (0.50, 0.50, 'dialog_box', 0.98),
        (0.45, 0.60, 'save_button', 0.95),
        (0.55, 0.60, 'cancel_button', 0.95),
    ]

f2 = np.full((128, 128, 3), 180, dtype=np.uint8)
f2[40:80, 40:80] = 20
obs2 = engine.generate(ModalState(), frame=f2)
print("=== SCENARIO 2 (Save Pop-up Aane Par) ===")
print("Model ki boli:")
print(f"  \"{obs2.scene_description}\"")
print()

# Scenario 3: Blank Screen (Missing Target)
class BlankState:
    timestamp = 110.0
    frame_index = 3
    primary_category = 'blank_canvas'
    stability = 'stable'
    surprise_score = 0.0
    has_active_modal = False
    affordance_hotspots = []

f3 = np.full((128, 128, 3), 40, dtype=np.uint8)
obs3 = engine.generate(BlankState(), frame=f3)
print("=== SCENARIO 3 (Jab Screen Blank / Target Missing Ho) ===")
print("Model ki boli:")
print(f"  \"{obs3.scene_description}\"")
