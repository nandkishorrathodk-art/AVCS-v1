# AVCS local demonstration data

`avcs-record-demo` is an opt-in offline data-collection utility: demonstration
mode writes labeled frame images to disk. It does not choose targets, move the
cursor, type, or claim task completion. Do not use that mode for the requested
no-image live-stream workflow. The current offline trainer consumes saved
demonstration frames; online learning directly from a live stream is not
implemented, so no model training should be claimed under a no-file constraint.

## Record a disposable window

1. Open a disposable test application window with a unique title and no private
   content.
2. From an interactive PowerShell terminal, run:

   ```powershell
   avcs-record-demo `
     --window-title "AVCS_DEMO_WINDOW" `
     --task "Complete the harmless test form" `
     --output-dir "runs/demonstrations/form-01" `
     --confirm-isolated-window
   ```

3. Enter a short description of the next control in the terminal, focus the
   named test window, and click that control. Repeat for each demonstrated
   click. Enter `:stop` in the terminal to finish.

For a guided run that does not require entering a label between clicks, pass
each expected target label up front. The recorder shows the next label in its
console and stops after every supplied target has been clicked:

```powershell
python -m avcs.training.demo_recorder `
  --window-title "Untitled - Notepad" `
  --task "Ground visible controls in a blank Notepad window" `
  --output-dir "runs/demonstrations/notepad-session-01" `
  --target-label "File menu" `
  --target-label "Edit menu" `
  --target-label "View menu" `
  --target-label "document editing area" `
  --confirm-isolated-window
```

## Check the live cursor-to-pixel stream (no screenshots saved)

Calibration continuously passes each fresh cropped RGB frame through
`HumanEye2M.perceive(frame=...)` directly from memory, alongside the physical
cursor coordinate. On the GDI backend, it captures the selected window's
physical-pixel rectangle directly instead of copying the whole desktop and
cropping afterward. It prints recent measured capture rate, capture and eye-pass
latency, the eye's static/dynamic path and motion signal, frame dimensions,
checkpoint availability, and window-relative cursor coordinates.
It saves no frames, overlays, labels, manifests, or training samples. Focus the
selected test window and move the pointer over several visible locations
without clicking. Press Ctrl+C in PowerShell to stop; when the window is not
foreground, the stream pauses. Closing the selected window stops calibration.

```powershell
python -m avcs.training.demo_recorder `
  --window-title "Untitled - Notepad" `
  --task "Calibrate cursor-to-frame coordinates" `
  --calibrate-only `
  --confirm-isolated-window
```

This exercises the live capture-to-eye input path and checks cursor-coordinate
alignment; it does not train the model or validate desktop understanding.
`weights_loaded` only reports whether a checkpoint is present; the checked-in
checkpoint was trained on procedural colored blocks and is not evidence of UI
understanding. Calibration mode rejects `--output-dir` and target labels to
prevent accidental recording. The existing offline grounder trainer still
requires saved frames, so it is not part of this no-image live-stream path.

The recorder requires an exact, unique, visible window title and a new output
directory. It records only while that window (or an owned modal) is foreground,
stores only cropped window frames and normalized left-click points, and never
records keyboard input or uploads data. Task descriptions and target labels are
saved with the examples; do not include secrets or real user content.
Window bounds come from DWM physical-pixel geometry and click points from
`GetPhysicalCursorPos`, avoiding DPI-virtualized `GetWindowRect` coordinates.
If Windows cannot provide physical bounds for the selected window, recording
fails rather than guessing a scale. Review each saved frame and its click point
before accepting the label. When the operator is following instructions
remotely, record one target label per run to avoid label/click order drift.

The output contains `manifest.json`, `samples.jsonl`, `frames/*.png`, and
review-only `audit/*.png` previews with a red marker at each saved click point.
The audit preview is a separate copy; only the unmarked `frames/*.png` image is
used for training. Samples are raw human demonstrations, not model predictions
or validation results. The checked-in Cortex-2M checkpoint was trained on procedural colored blocks.
Its recorded synthetic validation category accuracy is 10.83%; this is not a
real-application grounding score. The synthetic trainer now requires the
explicit `--synthetic-data-only` flag and writes to
`runs/synthetic_cortex_2m` by default. These demonstrations still need to be
reviewed, split by task/window, and used to train and independently evaluate a
click-grounding model. Until then, AVCS must not claim general GUI task
capability.

## Goal-conditioned pixel grounder

`GoalConditionedGrounder` (`avcs/goal_grounder.py`) is the new System-1
grounding architecture. It consumes an RGB frame and the planner's target
phrase, then produces a dense spatial click-target heatmap. Its learned byte
encoder consumes only the supplied phrase; UI pixels are not OCR'd or replaced
with accessibility/DOM state. It does not plan multi-step tasks, type text,
dispatch input, or verify that a click succeeded.

`GroundingStream` accepts fresh RGB frames one at a time and returns a
per-frame candidate plus measured inference latency and input cadence. It does
not capture the screen itself, keep temporal state, drop frames, or authorize
mouse actions. This API is not a real-time performance claim; measure it on
target hardware with trained weights. A temporal model needs sequence
demonstrations and separate validation.

The architecture is implemented but **has not been trained on real UI
demonstrations**. `predict()` intentionally refuses inference from a randomly
initialized model. Train only after collecting at least five separate
recording sessions; the trainer reserves complete sessions for validation and
held-out test, so frames from one session cannot leak between splits and at
least three sessions remain for training:

```powershell
avcs-train-grounder `
  --data-dir "runs/demonstrations" `
  --output-dir "runs/real_ui_grounder"
```

The same trainer is available without the installed console entry point as
`python -m avcs.training.train_grounder --data-dir "runs/demonstrations"`.

The trainer uses validation click-point distance to choose a checkpoint and
does not evaluate the held-out test split. It fingerprints all split data into
the checkpoint so evaluation can reject changed data. After training, run the
held-out evaluator separately:

```powershell
python -m avcs.training.evaluate_grounder `
  --data-dir "runs/demonstrations" `
  --checkpoint "runs/real_ui_grounder/goal_grounder_best.pt" `
  --output "runs/real_ui_grounder/held_out_report.json"
```

Its labels are click points, not target boxes, so it reports normalized/pixel
point distances and point-distance thresholds, not target-box hit rate. The
held-out results are only useful if the sessions and labels are representative;
they are not end-to-end task success. Once trained, a checkpoint can be loaded
with `load_grounding_checkpoint`; its output score is an uncalibrated peak
probability, not a confidence guarantee. Do not connect candidate coordinates
to live input until separate safety, calibration, and isolated acceptance work
establishes that behavior.
