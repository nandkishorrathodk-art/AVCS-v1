# MILESTONE 3: Physical OS Motor Driver (Closed-Loop Hand-Eye Servoing)

**Status:** ⚪ **PENDING** *(Locked until Milestone 2 completion)*  
**Repository:** [https://github.com/Praxis-Neural/AVCS](https://github.com/Praxis-Neural/AVCS)  
**Target Delivery:** Active operating system mouse cursor driver steered continuously via biological Fitts' Law velocity vectors.

---

## 1. Objective

Connect AVCS's closed-loop visual servoing engine directly to the operating system's physical mouse cursor. Instead of abrupt, unnatural coordinate jumps (`x=542, y=318`), AVCS steers the cursor along continuous biological motor flow vectors ($\vec{\Delta} = \text{Gaze} - \text{Cursor}$) with natural acceleration and deceleration, clicking only when foveal alignment contact ($r \le 0.025$) is acquired.

---

## 2. Deliverables & Task Checklist

| Task ID | Item Description | Status | Evidence / Artifact |
| :--- | :--- | :--- | :--- |
| **M3.1** | **Visual Servoing Engine Core (`VisualServoingEngine`)** | ✅ **COMPLETED** | `forge/vision/cortex/core/servoing.py` |
| **M3.2** | **Simulated Servoing Convergence Test (100% Pass)** | ✅ **COMPLETED** | `tests/test_cortex.py::test_visual_servoing` |
| **M3.3** | **Windows Win32 OS Cursor Driver (`SendInput` / API)** | ⚪ **PENDING** | `forge/vision/cortex/motor/win32.py` |
| **M3.4** | **Cross-Platform Motor Fallback (PyAutoGUI / X11)** | ⚪ **PENDING** | `forge/vision/cortex/motor/driver.py` |
| **M3.5** | **Biological Jitter & Natural Saccadic Acceleration Curve** | ⚪ **PENDING** | Fitts' Law damping curve validation |
| **M3.6** | **Emergency Fail-Safe Mechanism (Corner Breakout)** | ⚪ **PENDING** | Safety cursor abort mechanism |
| **M3.7** | **Live Physical Cursor Verification on Real Screen** | ⚪ **PENDING** | Video / Telemetry log of real steering |
| **M3.8** | **Milestone 3 Final Verification & Sign-Off** | ⚪ **PENDING** | Complete before opening Milestone 4 |

---

## 3. Acceptance Criteria
1. The cursor moves smoothly across the screen toward designated gaze targets with zero instantaneous coordinate jumping.
2. Servoing achieves target contact ($r \le 0.025$) within $\le 350\text{ ms}$ on standard desktop resolutions.
3. User can abort agent cursor control at any time by moving the mouse to a designated corner of the screen.

