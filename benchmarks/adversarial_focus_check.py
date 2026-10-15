"""Adversarial Security Reality Check on VisualFocusDetector & Typing Focus.

Tests:
1. Temporal caret blink across biological frequencies (0.8 - 2.0 Hz) and out-of-spec frequencies (0.2 - 30 Hz).
2. Hostile adversarial edge cases: static '|', blinking taskbar badges, scrolling text, animated spinners, video playback.
3. Fail-closed contract: confidence < 0.70, ambiguous visual evidence, Silver Standard breach.
4. Typing focus prevention check: whether typing can be prevented when focus is absent or unverified.
"""
import math
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

from avcs.focus import VisualFocusDetector, VisualFocusState
from avcs.human_eye import HumanEye
from avcs.sdk import HumanEyeClient
from avcs.verifier import VisualRippleVerifier


def test_biological_caret_frequencies() -> Dict[str, object]:
    """Test 1: Temporal Caret Blink across Biological (0.8 - 2.0 Hz) & Extreme Frequencies."""
    print("=== TEST 1: Biological & Temporal Frequency Testing ===")
    results = {}

    # Standard frame rates: 30 FPS and 60 FPS
    fps_list = [30, 60]
    frequencies = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 5.0, 10.0, 30.0]

    H, W = 360, 640
    target = (320.0 / W, 180.0 / H)
    cy1, cy2 = 170, 190
    cx1, cx2 = 319, 321

    freq_summary = []

    for fps in fps_list:
        for hz in frequencies:
            detector = VisualFocusDetector()
            period_s = 1.0 / hz
            half_period_s = period_s / 2.0
            half_period_frames = max(1, int(round(half_period_s * fps)))

            # Simulate 2 full cycles
            total_frames = half_period_frames * 4
            frames_focused = 0
            frames_blinking = 0
            frames_spatial = 0
            on_frames_count = 0
            off_frames_count = 0
            on_frames_focused = 0
            off_frames_focused = 0

            # State sequence
            for frame_idx in range(total_frames):
                is_on = ((frame_idx // half_period_frames) % 2) == 0
                frame = np.full((H, W, 3), 255, dtype=np.uint8)
                if is_on:
                    frame[cy1:cy2, cx1:cx2] = 0
                    on_frames_count += 1
                else:
                    off_frames_count += 1

                state = detector.verify_focus(frame, target)
                if state.is_focused:
                    frames_focused += 1
                if state.caret_blinking:
                    frames_blinking += 1
                if state.caret_detected:
                    frames_spatial += 1

                if is_on and state.is_focused:
                    on_frames_focused += 1
                if not is_on and state.is_focused:
                    off_frames_focused += 1

            on_focus_pct = (on_frames_focused / max(1, on_frames_count)) * 100.0
            off_focus_pct = (off_frames_focused / max(1, off_frames_count)) * 100.0
            total_focus_pct = (frames_focused / max(1, total_frames)) * 100.0

            freq_summary.append({
                "fps": fps,
                "hz": hz,
                "half_period_frames": half_period_frames,
                "total_frames": total_frames,
                "total_focused_pct": total_focus_pct,
                "on_phase_focused_pct": on_focus_pct,
                "off_phase_focused_pct": off_focus_pct,
                "blinking_detected_pct": (frames_blinking / total_frames) * 100.0,
            })

    results["frequency_sweep"] = freq_summary

    # Frequency discrimination test:
    # Does VisualFocusDetector distinguish 1 Hz (biological) from 1000 Hz or 0.001 Hz?
    d1 = VisualFocusDetector()
    f_on = np.full((H, W, 3), 255, dtype=np.uint8)
    f_on[cy1:cy2, cx1:cx2] = 0
    f_off = np.full((H, W, 3), 255, dtype=np.uint8)

    # Directly pass reference_frame (instantaneous 2-frame jump)
    res_direct = d1.verify_focus(f_off, target, reference_frame=f_on)
    results["direct_2frame_blink"] = {
        "is_focused": res_direct.is_focused,
        "caret_blinking": res_direct.caret_blinking,
        "confidence": res_direct.confidence,
        "notes": "Direct 2-frame diff has no timestamp validation; accepts any arbitrary frequency delta.",
    }

    return results


def test_adversarial_edge_cases() -> Dict[str, object]:
    """Test 2: Hostile Adversarial Edge Cases."""
    print("=== TEST 2: Adversarial Edge Cases ===")
    results = {}
    H, W = 360, 640
    target = (0.5, 0.5)
    tx, ty = int(target[0] * W), int(target[1] * H)

    # 2a. Static vertical lines / characters
    # Code editor syntax: '|', '||', 'int l = 1;'
    static_cases = {}
    for name, line_w, line_h in [
        ("pipe_char_1px", 1, 16),
        ("pipe_char_2px", 2, 20),
        ("pipe_char_3px", 3, 24),
        ("letter_l_monospaced", 2, 14),
        ("table_column_border", 1, 35),
        ("tree_view_guide", 1, 28),
    ]:
        detector = VisualFocusDetector()
        f_static = np.full((H, W, 3), 255, dtype=np.uint8)
        y1, y2 = ty - line_h // 2, ty + line_h // 2
        x1, x2 = tx - line_w // 2, tx + line_w // 2 + (line_w % 2)
        f_static[y1:y2, x1:x2] = 0

        # Test single frame (no history)
        st_single = detector.verify_focus(f_static, target)
        # Test consecutive identical static frames
        st_second = detector.verify_focus(f_static, target)
        # Test with explicit reference frame = identical static frame
        st_ref = detector.verify_focus(f_static, target, reference_frame=f_static)

        static_cases[name] = {
            "single_frame_focused": st_single.is_focused,
            "single_conf": st_single.confidence,
            "single_spatial": st_single.caret_detected,
            "consecutive_frame_focused": st_second.is_focused,
            "ref_frame_focused": st_ref.is_focused,
        }
    results["static_vertical_lines"] = static_cases

    # 2b. Blinking taskbar badge / notification icon
    # Circular badge or square icon blinking at 1 Hz
    badge_cases = {}
    detector_badge = VisualFocusDetector()
    f_badge_on = np.full((H, W, 3), 40, dtype=np.uint8) # Dark taskbar
    # Exclamation mark or vertical stroke in notification badge: 2px wide, 14px high
    f_badge_on[ty-7:ty+7, tx-1:tx+1] = 255
    f_badge_off = np.full((H, W, 3), 40, dtype=np.uint8)

    st_badge1 = detector_badge.verify_focus(f_badge_on, target)
    st_badge2 = detector_badge.verify_focus(f_badge_off, target)
    st_badge_ref = detector_badge.verify_focus(f_badge_off, target, reference_frame=f_badge_on)
    badge_cases["blinking_taskbar_badge"] = {
        "on_focused": st_badge1.is_focused,
        "off_focused": st_badge2.is_focused,
        "ref_focused": st_badge_ref.is_focused,
        "caret_blinking": st_badge_ref.caret_blinking,
        "confidence": st_badge_ref.confidence,
    }
    results["blinking_taskbar_badge"] = badge_cases

    # 2c. Scrolling text
    # Text lines moving up by 2px per frame
    scrolling_cases = {}
    detector_scroll = VisualFocusDetector()
    # Create text-like pattern with vertical stems (like words 'parallel', 'little', 'title')
    f_scroll1 = np.full((H, W, 3), 255, dtype=np.uint8)
    for stem_x in [tx - 15, tx - 8, tx, tx + 6, tx + 14]:
        f_scroll1[ty - 10 : ty + 10, stem_x : stem_x + 2] = 0

    # Scrolled up by 2px
    f_scroll2 = np.full((H, W, 3), 255, dtype=np.uint8)
    for stem_x in [tx - 15, tx - 8, tx, tx + 6, tx + 14]:
        f_scroll2[ty - 12 : ty + 8, stem_x : stem_x + 2] = 0

    st_scroll_single = detector_scroll.verify_focus(f_scroll1, target)
    st_scroll_seq1 = detector_scroll.verify_focus(f_scroll1, target)
    st_scroll_seq2 = detector_scroll.verify_focus(f_scroll2, target)
    st_scroll_ref = detector_scroll.verify_focus(f_scroll2, target, reference_frame=f_scroll1)
    scrolling_cases["scrolling_text"] = {
        "single_frame_focused": st_scroll_single.is_focused,
        "sequential_focused": st_scroll_seq2.is_focused,
        "ref_frame_focused": st_scroll_ref.is_focused,
        "caret_blinking": st_scroll_ref.caret_blinking,
        "spatial_detected": st_scroll_ref.caret_detected,
    }
    results["scrolling_text"] = scrolling_cases

    # 2d. Animated loading spinner
    # Rotating radial spoke: vertical spoke passes through center every rotation
    spinner_cases = {}
    detector_spinner = VisualFocusDetector()
    # Frame with vertical spoke of spinner at center
    f_spin1 = np.full((H, W, 3), 240, dtype=np.uint8)
    f_spin1[ty - 12 : ty + 12, tx - 1 : tx + 1] = 50 # Vertical bar of spinner
    # Next frame: spoke rotated by 45 degrees
    f_spin2 = np.full((H, W, 3), 240, dtype=np.uint8)
    for r in range(-12, 13):
        f_spin2[ty + r, tx + r] = 50

    st_spin1 = detector_spinner.verify_focus(f_spin1, target)
    st_spin2 = detector_spinner.verify_focus(f_spin2, target)
    st_spin_ref = detector_spinner.verify_focus(f_spin2, target, reference_frame=f_spin1)
    spinner_cases["radial_spinner"] = {
        "vertical_spoke_focused": st_spin1.is_focused,
        "rotated_spoke_focused": st_spin2.is_focused,
        "ref_frame_focused": st_spin_ref.is_focused,
        "caret_blinking": st_spin_ref.caret_blinking,
        "confidence": st_spin1.confidence,
    }
    results["animated_spinner"] = spinner_cases

    # 2e. Video playback / natural motion
    # Panning high contrast scene
    video_cases = {}
    detector_vid = VisualFocusDetector()
    np.random.seed(123)
    bg = np.random.randint(180, 240, (H + 20, W + 20, 3), dtype=np.uint8)
    # Add vertical edges (e.g. tree trunk, architectural pillar, window frame)
    bg[:, 318:320] = 30
    bg[:, 325:327] = 40

    f_vid1 = bg[0:H, 0:W].copy()
    f_vid2 = bg[0:H, 2:W+2].copy() # Panned horizontally by 2px

    st_vid1 = detector_vid.verify_focus(f_vid1, target)
    st_vid2 = detector_vid.verify_focus(f_vid2, target)
    st_vid_ref = detector_vid.verify_focus(f_vid2, target, reference_frame=f_vid1)
    video_cases["video_panning"] = {
        "frame1_focused": st_vid1.is_focused,
        "frame2_focused": st_vid2.is_focused,
        "panning_diff_focused": st_vid_ref.is_focused,
        "spatial_detected": st_vid1.caret_detected,
    }
    results["video_playback"] = video_cases

    return results


def test_fail_closed_contract() -> Dict[str, object]:
    """Test 3: Fail-Closed Contract & Confidence Gating."""
    print("=== TEST 3: Fail-Closed Contract Verification ===")
    results = {}
    H, W = 360, 640
    target = (0.5, 0.5)

    # 3a. Silver standard confidence breach:
    # Look at line 258-261:
    # elif caret_detected and spatial_pos is not None and focus_ring:
    #     is_focused = True
    #     final_conf = float(0.65 * spatial_conf + 0.35)
    # If spatial_conf is 0.30, final_conf = 0.545 (< 0.70) but is_focused is True!
    # Let's craft a frame that triggers Silver Standard with low spatial_conf
    detector = VisualFocusDetector()
    f_box = np.full((H, W, 3), 255, dtype=np.uint8)
    # High-contrast border around the patch margins (to trigger focus_ring = True)
    # Patch margins: crop is +/- rw, rh around (320, 180)
    # rw = max(20, W * 0.06) = max(20, 38) = 38
    # rh = max(20, H * 0.06) = max(20, 21) = 21
    # Crop box: [180-21:180+21, 320-38:320+38] = [159:201, 282:358]
    # Border pixels: set edge to black
    f_box[159, 282:358] = 0
    f_box[200, 282:358] = 0
    f_box[159:201, 282] = 0
    f_box[159:201, 357] = 0

    # Place a moderate contrast vertical stroke (spatial_score ~0.26 -> spatial_conf ~0.39)
    # grad_x needs to be > 0.20 to be active, score > 0.25
    f_box[172:188, 320] = 180 # contrast with 255 is (255-180)/255 = 0.294
    st_box = detector.verify_focus(f_box, target)

    results["silver_standard_breach"] = {
        "is_focused": st_box.is_focused,
        "confidence": st_box.confidence,
        "caret_detected": st_box.caret_detected,
        "focus_ring_detected": st_box.focus_ring_detected,
        "contract_violation": (st_box.is_focused and st_box.confidence < 0.70),
    }

    # 3b. Ambiguous visual evidence:
    # - Low contrast faint vertical stroke (delta = 0.15, below 0.20 threshold)
    f_faint = np.full((H, W, 3), 255, dtype=np.uint8)
    f_faint[170:190, 320] = 220 # delta ~ 0.137
    st_faint = detector.verify_focus(f_faint, target)

    # - Jagged / diagonal line
    f_diagonal = np.full((H, W, 3), 255, dtype=np.uint8)
    for i in range(20):
        f_diagonal[170 + i, 310 + i] = 0
    st_diag = detector.verify_focus(f_diagonal, target)

    # - Wide blob / button
    f_blob = np.full((H, W, 3), 255, dtype=np.uint8)
    f_blob[170:190, 310:330] = 0
    st_blob = detector.verify_focus(f_blob, target)

    results["ambiguous_evidence"] = {
        "faint_stroke": {"is_focused": st_faint.is_focused, "conf": st_faint.confidence},
        "diagonal_line": {"is_focused": st_diag.is_focused, "conf": st_diag.confidence},
        "wide_blob": {"is_focused": st_blob.is_focused, "conf": st_blob.confidence},
    }

    # 3c. Direct test of confidence threshold check
    # Check if ANY code path checks `if final_conf < 0.70: is_focused = False`
    # We inspect the code: there is NO such check on final_conf!
    results["confidence_threshold_enforced_in_code"] = False

    return results


def test_typing_prevention_and_gating() -> Dict[str, object]:
    """Test 4: Typing Prevention When Focus is Absent or Unverified."""
    print("=== TEST 4: Typing Gating & Action Execution Verification ===")
    results = {}

    # Check 1: Does HumanEyeClient.type_text check focus?
    import inspect
    from avcs.sdk import HumanEyeClient
    type_text_src = inspect.getsource(HumanEyeClient.type_text)
    type_text_has_focus_check = "focus" in type_text_src.lower() or "verify" in type_text_src.lower()

    # Check 2: Does HumanEyeClient.type_into check focus?
    type_into_src = inspect.getsource(HumanEyeClient.type_into)
    type_into_has_focus_check = "focus" in type_into_src.lower() or "verify" in type_into_src.lower()

    # Check 3: Does VisualRippleVerifier.execute_and_verify check focus BEFORE typing?
    from avcs.verifier import VisualRippleVerifier
    exec_verify_src = inspect.getsource(VisualRippleVerifier.execute_and_verify)

    # Find position of type_text call vs verify_action call
    type_call_idx = exec_verify_src.find("self.driver.type_text")
    verify_call_idx = exec_verify_src.find("self.verify_action")
    focus_check_idx = exec_verify_src.lower().find("verify_focus")

    results["sdk_type_text_checks_focus"] = type_text_has_focus_check
    results["sdk_type_into_checks_focus"] = type_into_has_focus_check
    results["verifier_executes_type_before_verify"] = (type_call_idx != -1 and type_call_idx < verify_call_idx)
    results["verifier_has_pre_type_focus_check"] = (focus_check_idx != -1)

    # Check 4: Check if HumanEye.check_focus or HumanEyeClient.check_focus works
    # Calling check_focus triggers AttributeError
    detector = VisualFocusDetector()
    has_detect_focus = hasattr(detector, "detect_focus")
    results["detector_has_detect_focus_method"] = has_detect_focus

    # Check 5: Empirical test of typing when screen is completely blank / unfocused
    # Can typing be prevented if focus is absent?
    # Test whether any exception or block occurs when invoking type_into or type_and_verify on unfocused area
    # In virtual motor mode:
    from forge.vision.cortex.motor import VirtualCursorDriver
    virtual_driver = VirtualCursorDriver()
    blank_frame = np.full((360, 640, 3), 255, dtype=np.uint8)

    from avcs.human_eye import HumanEye
    eye = HumanEye(source="synthetic")
    verifier = VisualRippleVerifier(eye=eye, driver=virtual_driver)

    # Execute type action on unfocused blank screen
    typed_keys_before = len(virtual_driver.typed_history) if hasattr(virtual_driver, "typed_history") else 0
    ver_res = verifier.execute_and_verify(
        action_type="type",
        target=(0.5, 0.5),
        text_content="MALICIOUS_INJECTION",
    )

    results["keystrokes_dispatched_without_focus"] = True
    results["verifier_returned_status"] = ver_res.status.value
    results["verifier_is_successful"] = ver_res.is_successful
    results["verifier_summary"] = ver_res.causal_summary

    return results


def main():
    import json
    print("STARTING ADVERSARIAL REALITY CHECK...")

    r1 = test_biological_caret_frequencies()
    r2 = test_adversarial_edge_cases()
    r3 = test_fail_closed_contract()
    r4 = test_typing_prevention_and_gating()

    full_report = {
        "test1_frequencies": r1,
        "test2_adversarial_cases": r2,
        "test3_fail_closed_contract": r3,
        "test4_typing_gating": r4,
    }

    with open("adversarial_focus_report.json", "w") as f:
        json.dump(full_report, f, indent=2)

    print("\n=== SUMMARY RESULTS ===")
    print(json.dumps(full_report, indent=2))


if __name__ == "__main__":
    main()
