"""Launch the AVCS virtual GUI assistant.

    python3 -m avcs.gui.run                        # assistant on a scripted screen
    python3 -m avcs.gui.run --virtual              # text GUI, virtual hand, no audio
    python3 -m avcs.gui.run --source screen        # the real screen
    python3 -m avcs.gui.run --eyes lexicon --grant click   # opt in to control reading

The default is an **assistant**: it looks at the screen (pure pixels, no OCR),
talks Hinglish through the LLM brain, and listens through the browser mic. It has
no mutating capability by default, so it cannot click anything. The `lexicon`
eyes and a `click` grant are an explicit opt-in for the control-grounding path.
"""
from __future__ import annotations

import argparse
import os
import sys


def _validate_llm_endpoint(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    clean = endpoint.strip()
    if not clean:
        return None
    from avcs.voice.brain import is_loopback_endpoint

    if not is_loopback_endpoint(clean):
        raise ValueError(
            f"--llm-endpoint {clean!r} rejected: only loopback endpoints "
            "(127.0.0.1, localhost, ::1) are permitted."
        )
    return clean


def main() -> int:
    ap = argparse.ArgumentParser(prog="avcs.gui.run")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=12000)
    ap.add_argument("--source", default="demo",
                    choices=["demo", "desktop", "synthetic", "screen"],
                    help="demo = a scripted screen with layout and motion; "
                         "screen = the real screen")
    ap.add_argument("--eyes", default="scene", choices=["scene", "lexicon", "vla"],
                    help="scene = describe the screen (the assistant default); "
                         "lexicon = also read control labels, which is what can "
                         "ground a click; vla = local FastVLAInferenceEngine perception")
    ap.add_argument("--scene", default="desktop",
                    choices=["desktop", "terminal", "media", "dialog", "scroll"],
                    help="initial scripted scene for --source demo")
    ap.add_argument("--grant", nargs="*", default=None,
                    help="capabilities granted to the kernel; defaults to none "
                         "(fail-closed) and to ['click'] only with --eyes lexicon. "
                         "Pass '--grant all' to unlock all MUTATING_OPS.")
    ap.add_argument("--live", action="store_true",
                    help="authorize REAL mouse/keyboard actuation (needs a grant)")
    ap.add_argument("--virtual", action="store_true",
                    help="run the GUI in virtual mode: no microphone, no server TTS, "
                         "and no physical mouse/keyboard actuation")
    ap.add_argument("--llm-endpoint", default=None,
                    help="Local loopback OpenAI-compatible LLM endpoint (127.0.0.1, "
                         "localhost, ::1) for the GUI bridge")
    ap.add_argument("--llm-model", default=None,
                    help="LLM model name for the GUI bridge "
                         "(or set AVCS_LLM_MODEL)")
    ap.add_argument("--language", choices=["hinglish", "english", "auto"],
                    default="hinglish", help="assistant reply language")
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
        llm_endpoint = _validate_llm_endpoint(args.llm_endpoint)
    except ValueError as exc:
        ap.error(str(exc))

    if args.virtual and args.live:
        ap.error("--virtual cannot be combined with --live")

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: python3 -m pip install uvicorn", file=sys.stderr)
        return 1

    from avcs.gui.server import build_app

    brain = None
    if llm_endpoint or args.llm_model or args.language != "hinglish":
        from avcs.reasoning.provider import LLMReasoningProvider
        from avcs.voice.brain import ConversationBrain

        if llm_endpoint:
            provider = LLMReasoningProvider(
                endpoint=llm_endpoint,
                api_key=(os.environ.get("AVCS_LLM_API_KEY") or "").strip(),
                model=args.llm_model,
            )
            brain = ConversationBrain(provider=provider, language=args.language)
        else:
            brain = ConversationBrain(provider=None, language=args.language)
            if args.llm_model and getattr(brain, "provider", None) is not None:
                brain.provider.model = args.llm_model

    mouth = "silent" if args.virtual else args.mouth
    ears = False if args.virtual else not args.no_ears

    app = build_app(
        source=args.source,
        category=args.scene,
        grants=args.grant,
        checkpoint=args.checkpoint,
        live=False if args.virtual else args.live,
        mouth=mouth,
        ears=ears,
        eyes=args.eyes,
        brain=brain,
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
    print(f"  mode       : {'virtual' if args.virtual or not args.live else 'live'}")
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

