# Department: OS I/O (`avcs.os_io`)

## 1. Responsibilities & Boundaries
- Hardware, display, and operating system integration layer for Windows desktop actuation.
- Encapsulates Win32 desktop captures, per-monitor DPI awareness calibration, mouse/keyboard input injection (`SendInput`, `mouse_event`), and transparent layered HUD overlays (Dynamic Island HUD).
- Isolates all platform-specific C/Win32 ctypes code behind clean Python adapter interfaces (`WindowsComputerAdapter`, `AVCSVisualHUD`).

## 2. Allowed Imports
- `ctypes`, `ctypes.windll`, `ctypes.wintypes` (guarded or Windows-only).
- `win32gui`, `win32con`, `win32api`, `win32process` (Windows-only).
- `PIL.Image`, `PIL.ImageDraw`.
- `avcs.protocol` (for action and coordinate interpretations).

## 3. Forbidden Imports
- Direct reliance on training pipelines (`avcs.train`).
- Direct dependency on deep learning vision architectures (`avcs.models.vla_cortex`).
- Orchestration planning logic (`avcs.runtime.planner`).

## 4. Sub-Agent Contract
- OS I/O is strictly isolated to prevent breaking non-Windows environments (such as Linux CI or Kaggle GPU runners).
- `avcs.data`, `avcs.models`, and `avcs.train` MUST NEVER eagerly import `avcs.os_io`.
- All higher-level agents in `avcs.runtime` interact with OS actuation exclusively through the `WindowsComputer` / `WindowsComputerAdapter` interfaces.
