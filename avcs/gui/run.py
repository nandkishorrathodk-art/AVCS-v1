"""Launch the AVCS GUI assistant.

    python3 -m avcs.gui.run                        # assistant on a scripted screen
    python3 -m avcs.gui.run --source screen        # the real screen
    python3 -m avcs.gui.run --eyes lexicon --grant click   # opt in to control reading

The default is an **assistant**: it looks at the screen (pure pixels, no OCR),
talks Hinglish through the LLM brain, and listens through the browser mic. It has
no mutating capability by default, so it cannot click anything. The `lexicon`
eyes and a `click` grant are an explicit opt-in for the control-grounding path.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(prog="avcs.gui.run")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=12000)
    ap.add_argument("--source", default="demo",
                    choices=["demo", "desktop", "synthetic", "screen"],
                    help="demo = a scripted screen with layout and motion; "
                         "screen = the real screen")
    ap.add_argument("--eyes", default="scene", choices=["scene", "lexicon"],
                    help="scene = describe the screen (the assistant default); "
                         "lexicon = also read control labels, which is what can "
                         "ground a click")
    ap.add_argument("--scene", default="desktop",
                    choices=["desktop", "terminal", "media", "dialog", "scroll"],
                    help="initial scripted scene for --source demo")
    ap.add_argument("--grant", nargs="*", default=None,
                    help="capabilities granted to the kernel; defaults to none "
                         "(the assistant looks and talks) and to ['click'] only "
                         "with --eyes lexicon")
    ap.add_argument("--live", action="store_true",
                    help="authorize REAL mouse/keyboard actuation (needs a grant)")
    ap.add_argument("--checkpoint", default=None,
                    help="lexicon checkpoint for --eyes lexicon")
    ap.add_argument("--mouth", default="browser", choices=["browser", "local", "silent"],
                    help="where the assistant speaks: the browser (default), the "
                         "server's own audio device, or nowhere (tests)")
    ap.add_argument("--no-ears", action="store_true",
                    help="disable server-side speech recognition (the mic button "
                         "then reports that ears are off)")
    ap.add_argument("--memory", default=None,
                    help="path to the durable memory file (default ~/.avcs/memory.json)")
    ap.add_argument("--no-memory", action="store_true",
                    help="run without durable memory (nothing is written to disk)")
    ap.add_argument("--trajectory", default=None,
                    help="path to the JSONL trajectory log (default "
                         "~/.avcs/trajectories.jsonl)")
    ap.add_argument("--no-trajectory", action="store_true",
                    help="do not log trajectories")
    ap.add_argument("--path-roots", nargs="*", default=None,
                    help="roots the file ops may read (needed for read_file/list_dir)")
    ap.add_argument("--allowed-apps", nargs="*", default=None,
                    help="app binaries launch_app may start (needed for launch_app)")
    ap.add_argument("--servo", action="store_true",
                    help="move the pointer along a human neuromuscular trajectory "
                         "(minimum-jerk + Fitts' Law + micro-tremor) instead of "
                         "teleporting. Adds ~0.3-0.9 s of travel per click, which is "
                         "the point: it looks like a hand, not a jump.")
    ap.add_argument("--proactive", action="store_true",
                    help="watch the screen and speak first when something is worth "
                         "mentioning (a dialog, a view switch, a stuck user). The "
                         "watcher has no hand, so it can never actuate.")
    args = ap.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: python3 -m pip install uvicorn", file=sys.stderr)
        return 1

    from avcs.gui.server import build_app

    app = build_app(
        source=args.source,
        category=args.scene,
        grants=args.grant,
        checkpoint=args.checkpoint,
        live=args.live,
        mouth=args.mouth,
        ears=not args.no_ears,
        eyes=args.eyes,
        memory_path=None if args.no_memory else args.memory,
        trajectory_path=None if args.no_trajectory else args.trajectory,
        path_roots=args.path_roots,
        allowed_apps=args.allowed_apps,
        proactive=args.proactive,
        servo=args.servo,
    )
    gui = app.state.gui
    caps = gui.capabilities()
    print("AVCS assistant")
    print(f"  source     : {args.source}" + (f" ({args.scene})" if args.source == "demo" else ""))
    print(f"  eyes       : {caps.get('eyes')}")
    print(f"  brain      : {caps.get('brain')} (key present: {caps.get('brain_has_key')})")
    print(f"  mouth      : {caps.get('mouth')}")
    print(f"  ears       : {caps.get('ears')}")
    print(f"  can act    : {caps.get('can_act')} (grants: {gui.grants or 'none'})")
    print(f"  actuation  : {caps.get('actuation')}"
          + (f" ({caps.get('hand')})" if caps.get("hand") else ""))
    if args.live and caps.get("actuation") == "live-unavailable":
        print("  WARNING    : --live was requested but no hand is available "
              "(need xdotool + $DISPLAY on Linux, or Windows user32); "
              "every actuation will be refused.")
    print(f"  proactive  : {caps.get('proactive')}")
    print(f"  motion     : {'neuromuscular (human)' if args.servo else 'teleport'}")
    print(f"  http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

