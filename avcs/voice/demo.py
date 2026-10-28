"""Voice assistant demo: talk to AVCS, hear it answer, watch it act.

Examples::

    # One text turn (no mic/speaker needed) -- proves the brain path
    python3 -m avcs.voice.demo --text "mujhe X kaam karna hai"

    # Speak a phrase out loud and print how long it took
    python3 -m avcs.voice.demo --say "Hello sir, what work would you like me to do?"

    # Full voice loop: listen on the mic, answer out loud, act through the gate
    python3 -m avcs.voice.demo --listen --turns 5 --grant click

    # Same loop from recorded WAV files, on a machine with no microphone
    python3 -m avcs.voice.demo --audio turn1.wav,turn2.wav --turns 2 --grant click

    # Does the cloud brain actually work? Exit 0 only if the key/endpoint do
    python3 -m avcs.voice.demo --check-llm

    # The same clarify-then-act flow with no microphone: scripted utterances
    python3 -m avcs.voice.demo --script "mujhe X kaam karna hai" "Save button daba do"

    # Real screen perception with the trained grounder, so the gate can pass
    python3 -m avcs.voice.demo --script "Save button daba do" --eyes --ground --grant click

    # Same, with real actuation (opt-in, dangerous)
    python3 -m avcs.voice.demo --listen --eyes --live --grant click
"""
from __future__ import annotations

import argparse
import json
import sys

from avcs.control.capability import CapabilityKernel
from avcs.voice.assistant import Assistant, StaticPerception
from avcs.voice.brain import ConversationBrain
from avcs.voice.stt import NullEngine, default_stt_engine
from avcs.voice.tts import EspeakNgEngine, NullEngine as NullTtsEngine, Speaker, VoiceProfile


def _voice_profile(args) -> VoiceProfile:
    """Pick the mouth's voice. Hinglish is the default, since the user speaks it."""
    if getattr(args, "voice_language", "hinglish") == "english":
        return VoiceProfile(language="en-us")
    return VoiceProfile.hinglish()


def _wav_turn_reader(spec: str):
    """Return a ``record_audio`` callable that yields one WAV file per turn.

    ``spec`` is a comma-separated list of WAV paths, or a single directory
    (consumed in sorted order). Each call returns the next clip, so a multi-turn
    session replays a real audio conversation without a microphone. Running out
    of clips returns silence, which the assistant reports honestly as "no speech
    detected" rather than looping on the last utterance forever.
    """
    import os

    import numpy as np

    from avcs.voice.stt import AudioClip

    if os.path.isdir(spec):
        paths = sorted(
            os.path.join(spec, n) for n in os.listdir(spec) if n.lower().endswith(".wav")
        )
    else:
        paths = [p.strip() for p in spec.split(",") if p.strip()]
    if not paths:
        raise SystemExit(f"--audio found no WAV files at {spec!r}")

    state = {"i": 0}

    def read_next(_seconds: float) -> AudioClip:
        i = state["i"]
        state["i"] = i + 1
        if i >= len(paths):
            return AudioClip(samples=np.zeros(0, dtype="float32"), sample_rate=16000)
        return AudioClip.read_wav(paths[i])

    return read_next


def build_assistant(args) -> Assistant:
    from avcs.reasoning.provider import LLMReasoningProvider

    provider = LLMReasoningProvider(
        endpoint=args.llm_endpoint or None,
        model=args.llm_model or None,
    )
    brain = ConversationBrain(
        provider=provider,
        language=getattr(args, "voice_language", "hinglish"),
    )

    if args.silent:
        speaker = Speaker(engine=NullTtsEngine(), profile=_voice_profile(args))
    elif args.no_audio:
        speaker = Speaker(engine=NullTtsEngine(), profile=_voice_profile(args))
    else:
        speaker = Speaker(engine=EspeakNgEngine(), profile=_voice_profile(args))

    stt = (
        default_stt_engine(model_path=args.vosk_model, language=args.stt_language)
        if (args.listen or args.loop or args.audio)
        else NullEngine()
    )

    record_audio = None
    if args.audio:
        record_audio = _wav_turn_reader(args.audio)

    perception = None
    if args.lexicon_eyes:
        # Labelled-control grounding via the AVC lexicon. This is the path that
        # actually authorizes a click: it returns a real candidate point from a
        # recognised control label, where the goal grounder's gate collapses.
        try:
            from avcs.voice.lexicon_perception import LexiconPerception

            if args.desktop_eyes:
                # A synthetic desktop whose controls are real vocabulary words,
                # so the reader has something to read without a display.
                from avcs.gui.desktop import DesktopSource

                perception = LexiconPerception(frame_source=DesktopSource())
            else:
                perception = LexiconPerception(
                    source="screen" if not args.synthetic_eyes else "synthetic"
                )
        except Exception as exc:
            print(f"warning: lexicon eyes unavailable ({exc}); running without perception.",
                  file=sys.stderr)
            perception = None
    elif args.eyes:
        try:
            if args.ground:
                from avcs.voice.perception import GroundingPerception

                perception = GroundingPerception(
                    source="screen" if not args.synthetic_eyes else "synthetic"
                )
            else:
                from avcs.voice.perception import EyePerception

                perception = EyePerception(
                    source="screen" if not args.synthetic_eyes else "synthetic"
                )
        except Exception as exc:
            # No checkpoint or no display: run without eyes rather than crash.
            # The gate is fail-closed, so actions are refused with a clear reason
            # instead of being performed blind.
            print(f"warning: eyes unavailable ({exc}); running without perception.",
                  file=sys.stderr)
            perception = None

    # The kernel's default floor of 0.75 is calibrated for the goal grounder's
    # peak, which sits above 0.75 on a hit. The lexicon reader's confidence is a
    # different scale (cosine margin against 61 prototypes), where a correct
    # control reads 0.5-0.7, so 0.75 would refuse every real control. The floor
    # is therefore a per-perception choice, not a global constant, and the
    # default here is the lexicon's.
    floor = args.peak_floor
    if floor is None:
        floor = 0.45 if args.lexicon_eyes else 0.75
    kernel = CapabilityKernel(granted_ops=args.grant or (), min_peak_probability=floor)
    hand = None
    if args.live:
        # Real actuation: use the platform's real hand (Windows HandMotor, or
        # xdotool on Linux/X11). Fail closed when neither is present rather than
        # silently falling back to a virtual hand the caller did not ask for.
        from avcs.voice.actuators import default_hand

        hand = default_hand()
        if hand is None:
            print(
                "warning: --live requested but no real hand is available "
                "(install xdotool and set DISPLAY, or run on Windows); "
                "actions will be refused.",
                file=sys.stderr,
            )
    elif args.lexicon_eyes:
        # A granted click with no hand attached is reported as allowed but not
        # performed, which reads as a gap. A virtual hand records the coordinate
        # the click resolved to and moves no cursor, so the loop is visibly
        # complete without touching the machine.
        from avcs.system1.loop import VirtualHand

        hand = VirtualHand(width=900, height=520)

    return Assistant(
        brain=brain,
        speaker=speaker,
        stt=stt,
        perception=perception,
        kernel=kernel,
        hand=hand,
        listen_seconds=args.seconds,
        record_audio=record_audio,
        actuation="live" if args.live else "virtual",
        stream=not args.no_stream,
        max_task_steps=args.max_steps,
    )


def _check_llm(assistant, args) -> None:
    """Probe the LLM endpoint once and report honestly. Exit 0 only on success.

    Mirrors the ``--check`` honesty rule in ``avcs_sdk``: an unconfigured or
    unreachable endpoint must exit non-zero, so a caller cannot mistake
    "the process ran" for "the brain works".
    """
    provider = assistant.brain.provider
    if not getattr(provider, "api_key", ""):
        print("[LLM CHECK] no API key configured.")
        print("  set AVCS_LLM_API_KEY (and optionally AVCS_LLM_ENDPOINT / AVCS_LLM_MODEL),")
        print("  or pass --llm-endpoint / --llm-model. Without a key the brain falls back")
        print("  to the local RuleBrain and cannot hold a real conversation.")
        raise SystemExit(2)
    if not hasattr(provider, "verify_live_connection"):
        print(f"[LLM CHECK] provider {type(provider).__name__} cannot be probed.")
        raise SystemExit(2)
    print(f"[LLM CHECK] endpoint={provider.endpoint} model={provider.model} "
          f"key={provider.masked_api_key}")
    try:
        report = provider.verify_live_connection()
    except Exception as exc:
        print(f"[LLM CHECK] FAILED: {exc}")
        raise SystemExit(2) from exc
    print(f"[LLM CHECK] OK  status={report['status']}  latency={report['latency_ms']} ms")
    print(f"  probe_output: {report['probe_output'][:200]}")
    if report.get("last_usage"):
        print(f"  usage: {report['last_usage']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="AVCS real-time voice assistant demo")
    ap.add_argument("--text", type=str, default=None, help="one user utterance, no microphone")
    ap.add_argument("--task", type=str, default=None,
                    help="run this request autonomously to completion (multi-step), no microphone")
    ap.add_argument("--script", nargs="+", default=None,
                    help="a scripted multi-turn conversation, no microphone "
                         '(e.g. --script "mujhe X kaam karna hai" "Save button daba do")')
    ap.add_argument("--say", type=str, default=None, help="speak this phrase and exit")
    ap.add_argument("--listen", action="store_true", help="record from the microphone each turn")
    ap.add_argument("--audio", type=str, default=None,
                    help="read the user turn from a WAV file instead of a microphone "
                         "(one file per turn, in order; lets the full ears->brain->mouth "
                         "loop run on a machine with no sound card)")
    ap.add_argument("--loop", action="store_true",
                    help="continuous real-time session: listen -> think -> speak -> act, until "
                         "a stop phrase ('stop'/'band karo') or Ctrl-C")
    ap.add_argument("--turns", type=int, default=3, help="max conversational turns")
    ap.add_argument("--no-stream", action="store_true",
                    help="disable streaming speech (wait for the whole reply, then speak)")
    ap.add_argument("--max-steps", type=int, default=8,
                    help="max autonomous action steps per spoken task")
    ap.add_argument("--seconds", type=float, default=4.0, help="seconds to record per turn")
    ap.add_argument("--vosk-model", type=str, default=None, help="path to a Vosk model directory")
    ap.add_argument("--stt-language", type=str, default=None,
                    help="force a Whisper language (default: auto-detect, best for Hinglish)")
    ap.add_argument("--eyes", action="store_true", help="use real pure-pixel screen perception")
    ap.add_argument("--lexicon-eyes", action="store_true",
                    help="ground labelled controls via the AVC lexicon (authorizes clicks)")
    ap.add_argument("--synthetic-eyes", action="store_true", help="force the headless frame buffer")
    ap.add_argument("--desktop-eyes", action="store_true",
                    help="with --lexicon-eyes, read a synthetic desktop with real controls")
    ap.add_argument("--ground", action="store_true",
                    help="with --eyes, also run the trained grounder so the gate can pass")
    ap.add_argument("--llm-endpoint", type=str, default=None,
                    help="override the LLM endpoint (default: env or local Ollama)")
    ap.add_argument("--llm-model", type=str, default=None, help="override the LLM model name")
    ap.add_argument("--voice-language", choices=["hinglish", "english"], default="hinglish",
                    help="mouth voice; default hinglish since the user speaks it")
    ap.add_argument("--grant", nargs="*", default=["click"],
                    help="mutating ops allowed; pass no values for a fully fail-closed run")
    ap.add_argument("--peak-floor", type=float, default=None,
                    help="kernel grounding floor (default: 0.45 for --lexicon-eyes, else 0.75)")
    ap.add_argument("--live", action="store_true", help="authorize REAL mouse/keyboard actuation")
    ap.add_argument("--silent", action="store_true", help="never touch the audio device")
    ap.add_argument("--no-audio", action="store_true", help="alias for --silent")
    ap.add_argument("--json", action="store_true", help="emit the turn records as JSON")
    ap.add_argument("--check-llm", action="store_true",
                    help="probe the configured LLM endpoint with one minimal request and "
                         "report honestly whether the key, endpoint and model work, then exit")
    args = ap.parse_args()

    assistant = build_assistant(args)

    if args.check_llm:
        _check_llm(assistant, args)
        return

    if args.say:
        result = assistant.speaker.speak(args.say)
        print(f"engine={result.engine} spoken={result.spoken} duration={result.duration_s:.2f}s")
        if result.error:
            print(f"note: {result.error}")
        return

    caps = assistant.capabilities()
    print("[AVCS voice assistant]")
    for k, v in caps.items():
        print(f"  {k:<16}: {v}")
    print()

    if args.text:
        rec = assistant.step(args.text, turn=1)
        _print_record(rec, args.json)
        return

    if args.task:
        records = assistant.run_task(args.task, max_steps=args.max_steps)
        for rec in records:
            _print_record(rec, args.json)
        if args.json:
            print(json.dumps([r.to_dict() for r in records], indent=2))
        else:
            acted = sum(1 for r in records if r.performed)
            verified = sum(1 for r in records if r.verified)
            print(f"\n[summary] steps={len(records)} actions={acted} verified={verified}")
        return

    if args.loop or args.audio:
        # Continuous real-time session. --turns 0 means run until stopped.
        # --audio drives the same loop from WAV files, so the ears path is
        # exercised even where no microphone exists.
        report = assistant.loop(
            max_turns=args.turns if args.turns else 0,
            on_record=lambda rec: _print_record(rec, args.json),
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"\n[summary] turns={report.total_turns} actions={report.actions_taken} "
                  f"verified={report.actions_verified} duration={report.duration_s:.1f}s")
        return

    if not args.listen and not args.script:
        print("Nothing to do: pass --text, --task, --say, --script, or --listen.")
        print("Try: python3 -m avcs.voice.demo --task \"Save button daba do\" --eyes --grant click")
        return

    report = assistant.converse(max_turns=args.turns, utterances=args.script)
    for rec in report.turns:
        _print_record(rec, args.json)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"\n[summary] turns={report.total_turns} actions={report.actions_taken} "
              f"verified={report.actions_verified} duration={report.duration_s:.1f}s")


def _print_record(rec, as_json: bool) -> None:
    if as_json:
        print(json.dumps(rec.to_dict(), indent=2))
        return
    if rec.kind == "silent":
        print(f"[turn {rec.turn}] (heard nothing: {rec.listen_error})")
        return
    print(f"[turn {rec.turn}]")
    print(f"  you   : {rec.heard}")
    print(f"  avcs  : {rec.said}   (spoken={rec.spoken}, brain={rec.source})")
    if rec.action:
        print(f"  action: {rec.action} target={rec.target!r}")
        print(f"  gate  : {'ALLOW' if rec.gate_allowed else 'DENY'} ({rec.gate_reason})")
        if rec.performed:
            extra = f" ({rec.detail})" if rec.detail else ""
            where = f" at {rec.clicked}" if rec.clicked else ""
            print(f"  hand  : {rec.actuation} {rec.action}{extra}{where}")
        if rec.verified is not None:
            print(f"  verify: {'changed' if rec.verified else 'no change'} "
                  f"(delta={rec.visual_delta})")
    if rec.speak_error:
        print(f"  note  : speech not played ({rec.speak_error})")
    if rec.brain_ms or rec.act_ms or rec.listen_ms:
        print(f"  timing: listen={rec.listen_ms:.0f}ms brain={rec.brain_ms:.0f}ms "
              f"act={rec.act_ms:.0f}ms")


if __name__ == "__main__":
    sys.exit(main())
