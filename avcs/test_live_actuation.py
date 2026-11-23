"""AVCS Step 6.1: Live Windows 11 Closed-Loop Client (WebSocket + Turbo JPEG).

Captures active desktop via Win32FastCapture, encodes with low-latency Turbo JPEG (<15ms, <35KB),
dispatches frame to the remote VLA server via persistent WebSocket (/ws/act) with Assistant Prefill,
parses the predicted physical action coordinate, and executes actuation via WindowsComputer.
"""
__test__ = False  # Prevent pytest from treating this live CLI script as a test

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forge.vision.capture import Win32FastCapture
from avcs.os_io.computer import WindowsComputer
from avcs.protocol.actions import Action, ActionKind
from avcs.runtime.turbo_jpeg import encode_turbo_jpeg
from avcs.runtime.vla_client import VLAClient

SERVER_URL = "https://affairs-counting-massive-feeds.trycloudflare.com"


def test_live_step(
    instruction: str = "Perform a left click on the taskbar search bar at the bottom",
    actuate: bool = False,
    server_url: str = SERVER_URL,
    use_websocket: bool = True,
    client: Optional[VLAClient] = None,
) -> Dict[str, Any]:
    protocol = "Persistent WebSocket (/ws/act)" if use_websocket else "HTTP POST (/act)"
    print("=" * 70)
    print("AVCS LIVE WINDOWS 11 CLOSED-LOOP TEST (STEP 6.1)")
    print(f"Target Server:       {server_url}")
    print(f"Transport Protocol:  {protocol}")
    print(f"Instruction:         '{instruction}'")
    print(f"Assistant Prefill:   " + '{"action": "left_click", "coordinate": [')
    print(f"Hardware Actuation:  {'ENABLED (Physical Mouse/Keyboard)' if actuate else 'DISABLED (Dry-Run)'}")
    print("=" * 70)

    # 1. Capture screen via Win32FastCapture (native Windows GDI)
    t_cap_start = time.time()
    capture = Win32FastCapture(target_size=(1280, 800))
    raw_frame = capture.grab()  # shape: (800, 1280, 3)
    t_cap_ms = (time.time() - t_cap_start) * 1000.0
    print(f"[1/4] Native Win32 Desktop Frame Captured in {t_cap_ms:.2f} ms | Shape: {raw_frame.shape}")

    # 2 & 3. Turbo JPEG Encode and Dispatch via Client
    own_client = False
    if client is None:
        client = VLAClient(server_url=server_url, use_websocket=use_websocket, jpeg_quality=75)
        own_client = True

    try:
        result = client.act(raw_frame, instruction=instruction, width=1280, height=800)
    finally:
        if own_client:
            client.close()

    t_enc_ms = result.get("compression_ms", 0.0)
    t_roundtrip_ms = result.get("roundtrip_ms", 0.0)
    payload_kb = result.get("payload_bytes", 0) // 1024
    print(f"[2/4] Turbo JPEG Compression ({payload_kb} KB, <35KB target) in {t_enc_ms:.2f} ms (<15ms target)")

    raw_resp = result.get("raw_response", "")
    coord = result.get("coordinate")
    action_type = result.get("action", "left_click")

    print(f"[3/4] Model Server Responded in {t_roundtrip_ms:.2f} ms (Server Latency: {result.get('latency_ms')} ms)")
    print(f"      Raw Generation: {raw_resp}")
    print(f"      Parsed Action:  {action_type} at {coord}")

    # 4. Optional Hardware Actuation
    if actuate and coord:
        print("[4/4] Executing Physical Win32 Hardware Actuation...")
        try:
            target_pt = (int(round(coord[0])), int(round(coord[1])))
            valid_kinds = {k.value: k for k in ActionKind}
            kind = valid_kinds.get(action_type, ActionKind.LEFT_CLICK)
            computer = WindowsComputer()
            action = Action(
                kind=kind,
                coordinate=target_pt,
                metadata={"user_confirmed": True},
            )
            computer.execute(action, user_confirmed=True)
            print(f"      Successfully executed {kind.value} at native pixel {target_pt}!")
        except Exception as exc:
            print(f"      Hardware actuation encountered error: {exc}")
    else:
        print("[4/4] Physical Actuation Skipped (Dry-Run mode). Pass --actuate to move cursor.")

    # 5. Summary Telemetry
    print("\n" + "=" * 70)
    print("END-TO-END TELEMETRY SUMMARY")
    print("=" * 70)
    print(f"Win32 Capture:          {t_cap_ms:.1f} ms")
    print(f"Turbo JPEG Compression: {t_enc_ms:.1f} ms")
    print(f"Payload Size:           {payload_kb} KB")
    print(f"Cloud VLA Latency:      {result.get('latency_ms')} ms")
    print(f"Total Roundtrip:        {t_roundtrip_ms:.1f} ms")
    print(f"Predicted Action:       {raw_resp}")
    print("=" * 70)

    result["capture_ms"] = round(t_cap_ms, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Live AVCS Closed-Loop Client (WebSocket + Turbo JPEG)")
    parser.add_argument("--instruction", type=str, default="Perform a left click on the taskbar search bar at the bottom")
    parser.add_argument("--actuate", action="store_true", help="Execute physical Win32 mouse/keyboard actions")
    parser.add_argument("--server-url", type=str, default=SERVER_URL, help="VLA model server URL")
    parser.add_argument("--http", action="store_true", help="Force legacy HTTP POST /act instead of persistent WebSocket")
    parser.add_argument("--steps", type=int, default=1, help="Number of closed-loop actuation steps")
    args = parser.parse_args()

    use_websocket = not args.http

    if args.steps <= 1:
        test_live_step(
            instruction=args.instruction,
            actuate=args.actuate,
            server_url=args.server_url,
            use_websocket=use_websocket,
        )
    else:
        print(f"Starting multi-step closed-loop session ({args.steps} steps) over persistent connection...")
        with VLAClient(server_url=args.server_url, use_websocket=use_websocket) as client:
            for step_idx in range(1, args.steps + 1):
                print(f"\n--- [STEP {step_idx}/{args.steps}] ---")
                test_live_step(
                    instruction=args.instruction,
                    actuate=args.actuate,
                    server_url=args.server_url,
                    use_websocket=use_websocket,
                    client=client,
                )
                time.sleep(0.5)


if __name__ == "__main__":
    main()
