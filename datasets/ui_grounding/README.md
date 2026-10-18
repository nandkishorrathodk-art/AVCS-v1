# AVCS UI Grounding & Affordance Dataset

- **Version:** 1.0.0
- **Created At:** 2026-10-15T17:23:09.046943+00:00
- **Provenance:** avcs_procedural_synthetic_desktop_v1
- **Total Frames:** 270
- **Resolution:** 256x256 RGB PNG
- **Affordance Grid:** 16x16 int64 NumPy masks

## Partition Summary
- `train`: 160 frames (8 sessions, balanced across 8 categories)
- `val`: 40 frames (4 sessions, distinct seeds)
- `test_held_out`: 40 frames (4 sessions, strictly disjoint from train/val)
- `test_negative`: 30 frames (10 solid blank, 10 smooth gradients, 10 procedural wallpapers, 0 interactive points)

## Verification
To verify dataset manifest schema and split isolation:
```bash
python3 -m pytest tests/test_ui_grounding_dataset.py -v
```
