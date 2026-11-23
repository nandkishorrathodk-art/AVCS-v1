"""AVCS Step 5.5: Live Windows 11 Closed-Loop Client.

Captures active desktop via Win32FastCapture, sends frame to the remote VLA server,
parses the predicted physical action coordinate, and optionally executes actuation via WindowsComputer.
"""
__test__ = False  # Prevent pytest from treating this live CLI script as a test

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path
import time
import urllib.request
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forge.vision.capture import Win32FastCapture
from avcs.os_io.computer import WindowsComputer
from avcs.protocol.actions import Action, ActionKind

SERVER_URL = "https://affairs-counting-massive-feeds.trycloudflare.com"

def test_live_step(
    instruction: str = "Perform a left click on the taskbar search bar at the bottom",
    actuate: bool = False
):
    print("=" * 70)
    print("AVCS LIVE WINDOWS 11 CLOSED-LOOP TEST")
    print(f"Target Server: {SERVER_URL}")
    print(f"Instruction:   '{instruction}'")
    print(f"Hardware Actuation: {'ENABLED (Physical Mouse/Keyboard)' if actuate else 'DISABLED (Dry-Run)'}")
    print("=" * 70)

    # 1. Capture screen via Win32FastCapture (native Windows GDI)
    t_cap_start = time.time()
    capture = Win32FastCapture(target_size=(1280, 800))
    raw_frame = capture.grab() # shape: (800, 1280, 3)
    t_cap_ms = (time.time() - t_cap_start) * 1000.0
    print(f"[1/4] Native Win32 Desktop Frame Captured in {t_cap_ms:.2f} ms | Shape: {raw_frame.shape}")

    # 2. Encode to JPEG Base64
    t_enc_start = time.time()
    img = Image.fromarray(raw_frame)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    t_enc_ms = (time.time() - t_enc_start) * 1000.0
    print(f"[2/4] JPEG Compression ({len(b64_str) // 1024} KB) in {t_enc_ms:.2f} ms")

    # 3. Dispatch to Remote Cloud Model
    t_net_start = time.time()
    payload = json.dumps({
        "instruction": instruction,
        "image_base64": b64_str,
        "width": 1280,
        "height": 800,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{SERVER_URL}/act",
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    t_roundtrip_ms = (time.time() - t_net_start) * 1000.0

    raw_resp = result.get("raw_response", "")
    coord = result.get("coordinate")
    action_type = result.get("action", "left_click")

    print(f"[3/4] Model Server Responded in {t_roundtrip_ms:.2f} ms (Server GPU Latency: {result.get('latency_ms')} ms)")
    print(f"      Raw Generation: {raw_resp}")
    print(f"      Parsed Action:  {action_type} at {coord}")

    # 4. Optional Hardware Actuation
    if actuate and coord:
        print("[4/4] Executing Physical Win32 Hardware Actuation...")
        computer = WindowsComputer()
        # Ensure coordinates are integers
        target_pt = (int(round(coord[0])), int(round(coord[1])))
        action = Action(
            kind=ActionKind(action_type),
            coordinate=target_pt,
            metadata={"user_confirmed": True}
        )
        computer.execute(action, user_confirmed=True)
        print(f"      Successfully executed {action_type} at native pixel {target_pt}!")
    else:
        print("[4/4] Physical Actuation Skipped (Dry-Run mode). Pass --actuate to move cursor.")

    # 5. Summary Telemetry
    print("\n" + "=" * 70)
    print("END-TO-END TELEMETRY SUMMARY")
    print("=" * 70)
    print(f"Win32 Capture:          {t_cap_ms:.1f} ms")
    print(f"JPEG Compression:       {t_enc_ms:.1f} ms")
    print(f"Cloud VLA Forward Pass: {result.get('latency_ms')} ms")
    print(f"Total Roundtrip:        {t_roundtrip_ms:.1f} ms")
    print(f"Predicted Action:       {raw_resp}")
    print("=" * 70)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live AVCS Closed-Loop Client")
    parser.add_argument("--instruction", type=str, default="Perform a left click on the taskbar search bar at the bottom")
    parser.add_argument("--actuate", action="store_true", help="Execute physical Win32 mouse/keyboard actions")
    args = parser.parse_args()

    test_live_step(instruction=args.instruction, actuate=args.actuate)
