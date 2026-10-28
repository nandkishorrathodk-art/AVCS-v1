"""Launch the AVCS GUI model.

    python3 -m avcs.gui.run --source desktop --port 12000
    python3 -m avcs.gui.run --source screen --port 12000      # real desktop
    python3 -m avcs.gui.run --live                            # real actuation

The default is the synthetic desktop and a virtual hand: the page is useful
immediately, and nothing on it can touch the machine until `--live` is passed
and the machine actually has a hand (`xdotool` on X11, or the Windows motor).
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(prog="avcs.gui.run")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=12000)
    ap.add_argument("--source", default="desktop",
                    choices=["desktop", "screen", "synthetic"],
                    help="desktop = synthetic controls; screen = the real screen")
    ap.add_argument("--grant", nargs="*", default=["click"],
                    help="capabilities granted to the kernel (default: click)")
    ap.add_argument("--live", action="store_true",
                    help="authorize REAL mouse/keyboard actuation (default: virtual)")
    ap.add_argument("--checkpoint", default=None,
                    help="lexicon checkpoint (default runs/weights/avc_lexicon.pt)")
    args = ap.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: python3 -m pip install uvicorn", file=sys.stderr)
        return 1

    from avcs.gui.server import build_app

    app = build_app(
        source=args.source,
        grants=args.grant,
        checkpoint=args.checkpoint,
        live=args.live,
    )
    gui = app.state.gui
    caps = gui.capabilities()
    print("AVCS GUI model")
    print(f"  source     : {args.source}")
    print(f"  brain      : {caps.get('brain')} (key present: {caps.get('brain_has_key')})")
    print(f"  grants     : {args.grant}")
    print(f"  actuation  : {caps.get('actuation')}")
    print(f"  http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
