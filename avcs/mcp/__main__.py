"""Command-line executable entrypoint for AVCS MCP server.

Run as:
    python -m avcs.mcp
    python -m avcs.mcp --synthetic --virtual-motor
"""
from __future__ import annotations

import argparse
import sys

from avcs.mcp.server import AVCSMCPServer
from forge.vision.cortex.sdk import ScreenEye
from forge.vision.capture import SyntheticScreen


def main() -> None:
    parser = argparse.ArgumentParser(description="AVCS Model Context Protocol (MCP) Server")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic screen source instead of physical display")
    parser.add_argument("--virtual-motor", action="store_true", help="Use in-memory virtual cursor & keyboard driver")
    parser.add_argument("--device", default="cpu", help="PyTorch compute device (default: cpu)")
    args = parser.parse_args()

    source = SyntheticScreen() if args.synthetic else None
    eye = ScreenEye(source=source, device=args.device)
    server = AVCSMCPServer(eye=eye, force_virtual_motor=args.virtual_motor)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()

