from __future__ import annotations

import numpy as np

from avcs.cortex_2m import HumanEye2M


def test_perceive_uses_supplied_frame_without_capturing_source(tmp_path):
    class NoCaptureSource:
        def grab(self):
            raise AssertionError("perceive must use the supplied frame")

    eye = HumanEye2M(
        source=NoCaptureSource(),
        weights_path=str(tmp_path / "missing-cortex-2m.pt"),
    )
    seen_frames = []
    eye.motion_engine.process = lambda frame: (
        seen_frames.append(frame) or (None, 0.0, [], 0.0)
    )
    eye.frame_count = 1
    eye._cached_affordances = [(0.5, 0.5, "button_primary", 1.0)]
    frame = np.full((48, 64, 3), 90, dtype=np.uint8)

    result = eye.perceive(frame=frame)

    assert seen_frames == [frame]
    assert result["is_static"] is True
    eye.close()


def test_perceive_rejects_invalid_supplied_frame_before_capture():
    class NoCaptureSource:
        def grab(self):
            raise AssertionError("invalid frame must fail before source capture")

    eye = HumanEye2M.__new__(HumanEye2M)
    eye.source = NoCaptureSource()

    try:
        eye.perceive(frame=np.zeros((8, 8, 4), dtype=np.uint8))
    except ValueError as exc:
        assert "uint8 RGB array" in str(exc)
    else:
        raise AssertionError("invalid supplied frame must be rejected")
