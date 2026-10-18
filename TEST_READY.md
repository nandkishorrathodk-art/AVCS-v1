# E2E Test Suite Ready

## Test Runner
- Command: `python3 -m pytest tests/e2e/ -v`
- Full regression command: `python3 -m pytest tests/ -q`
- Expected: all 70 E2E tests pass with exit code 0; full regression suite (737 tests) passes with 0 failures, 1 skipped.

## Coverage Summary
| Tier | Count | Description |
|------|------:|-------------|
| 1. Feature Coverage | 34 | Features 1-20 core contracts & public API coverage |
| 2. Boundary & Corner | 23 | Resolution extremes (320x240 to 4K), blank/wallpaper negatives (zero FP), contrast thresholds, blink frequencies |
| 3. Cross-Feature | 8 | Perception+motor coupling, grounding+fovea, focus+typing, broker+verifier, 60 FPS telemetry |
| 4. Real-World Application | 5 | Desktop exploration, safe form-fill, unvalidated lockout, negative rejection, observe-replan loop |
| **Total** | **70** | Full 4-tier opaque-box E2E coverage |

## Feature Checklist
| Feature | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---------|:------:|:------:|:------:|:------:|
| 1. Baseline Test & Checkpoint Integrity | ✓ | ✓ | - | - |
| 2. Synthetic & Held-Out UI Dataset | ✓ | ✓ | - | - |
| 3. Remove SHA-256 Text Embeddings | ✓ | - | ✓ | - |
| 4. Genuine Cross-Modal Grounding Head | ✓ | - | ✓ | ✓ |
| 5. Eliminate DSNT Center-Collapse | ✓ | - | - | - |
| 6. Grounding Accuracy & Localization | ✓ | - | ✓ | ✓ |
| 7. Weight Packaging & Auto-Discovery | ✓ | - | - | - |
| 8. Foveal Perception & ROI Inspection | ✓ | ✓ | ✓ | ✓ |
| 9. Zero-Hallucination Observation | ✓ | ✓ | - | ✓ |
| 10. 60 FPS Continuous Telemetry Bus | ✓ | - | ✓ | - |
| 11. 5th-Order Minimum-Jerk Kinematics | ✓ | - | ✓ | - |
| 12. Configurable 10.5 Hz Micro-Tremor | ✓ | - | - | - |
| 13. Fail-Closed Typing Safety | ✓ | ✓ | ✓ | ✓ |
| 14. Fresh Post-Action Frame Verification | ✓ | - | ✓ | ✓ |
| 15. Central AVCSController Orchestrator | ✓ | - | - | ✓ |
| 16. Coordinate Encapsulation & Intent APIs | ✓ | ✓ | - | ✓ |
| 17. Dynamic Grounding Validation Gate | ✓ | - | - | ✓ |
| 18. Observe-to-Replan Execution Loop | ✓ | - | - | ✓ |
| 19. Single-Thread CPU Latency Benchmark | ✓ | - | - | - |
| 20. Full Test Suite & Release Verification | ✓ | ✓ | ✓ | ✓ |

