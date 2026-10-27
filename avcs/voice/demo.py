"""Voice assistant demo: talk to AVCS, hear it answer, watch it act.

Examples::

    # One text turn (no mic/speaker needed) -- proves the brain path
    python3 -m avcs.voice.demo --text "mujhe X kaam karna hai"

    # Speak a phrase out loud and print how long it took
    python3 -m avcs.voice.demo --say "Hello sir, what work would you like me to do?"

    # Full voice loop: listen on the mic, answer out loud, act through the gate
    python3 -m avcs.voice.demo --listen --turns 5 --grant click

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


def build_assistant(args) -> Assistant:
    from avcs.reasoning.provider import LLMReasoningProvider

    provider = LLMReasoningProvider(
        endpoint=args.llm_endpoint or None,
        model=args.llm_model or None,
    )
    brain = ConversationBrain(provider=provider)

    if args.silent:
        speaker = Speaker(engine=NullTtsEngine(), profile=_voice_profile(args))
    elif args.no_audio:
        speaker = Speaker(engine=NullTtsEngine(), profile=_voice_profile(args))
    else:
        speaker = Speaker(engine=EspeakNgEngine(), profile=_voice_profile(args))

    stt = (
        default_stt_engine(model_path=args.vosk_model, language=args.stt_language)
        if args.listen
        else NullEngine()
    )

    perception = None
    if args.eyes:
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

    kernel = CapabilityKernel(granted_ops=args.grant or ())
    hand = None
    if args.live:
        from avcs_sdk.hand_motor import HandMotor

        hand = HandMotor()

    return Assistant(
        brain=brain,
        speaker=speaker,
        stt=stt,
        perception=perception,
        kernel=kernel,
        hand=hand,
        listen_seconds=args.seconds,
        actuation="live" if args.live else "virtual",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="AVCS real-time voice assistant demo")
    ap.add_argument("--text", type=str, default=None, help="one user utterance, no microphone")
    ap.add_argument("--script", nargs="+", default=None,
                    help="a scripted multi-turn conversation, no microphone "
                         '(e.g. --script "mujhe X kaam karna hai" "Save button daba do")')
    ap.add_argument("--say", type=str, default=None, help="speak this phrase and exit")
    ap.add_argument("--listen", action="store_true", help="record from the microphone each turn")
    ap.add_argument("--turns", type=int, default=3, help="max conversational turns")
    ap.add_argument("--seconds", type=float, default=4.0, help="seconds to record per turn")
    ap.add_argument("--vosk-model", type=str, default=None, help="path to a Vosk model directory")
    ap.add_argument("--stt-language", type=str, default=None,
                    help="force a Whisper language (default: auto-detect, best for Hinglish)")
    ap.add_argument("--eyes", action="store_true", help="use real pure-pixel screen perception")
    ap.add_argument("--synthetic-eyes", action="store_true", help="force the headless frame buffer")
    ap.add_argument("--ground", action="store_true",
                    help="with --eyes, also run the trained grounder so the gate can pass")
    ap.add_argument("--llm-endpoint", type=str, default=None,
                    help="override the LLM endpoint (default: env or local Ollama)")
    ap.add_argument("--llm-model", type=str, default=None, help="override the LLM model name")
    ap.add_argument("--voice-language", choices=["hinglish", "english"], default="hinglish",
                    help="mouth voice; default hinglish since the user speaks it")
    ap.add_argument("--grant", nargs="*", default=["click"],
                    help="mutating ops allowed; pass no values for a fully fail-closed run")
    ap.add_argument("--live", action="store_true", help="authorize REAL mouse/keyboard actuation")
    ap.add_argument("--silent", action="store_true", help="never touch the audio device")
    ap.add_argument("--no-audio", action="store_true", help="alias for --silent")
    ap.add_argument("--json", action="store_true", help="emit the turn records as JSON")
    args = ap.parse_args()

    assistant = build_assistant(args)

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

    if not args.listen and not args.script:
        print("Nothing to do: pass --text, --say, --script, or --listen.")
        print("Try: python3 -m avcs.voice.demo --text \"mujhe X kaam karna hai\"")
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
        if rec.clicked:
            print(f"  hand  : {rec.actuation} click at {rec.clicked}")
    if rec.speak_error:
        print(f"  note  : speech not played ({rec.speak_error})")


if __name__ == "__main__":
    sys.exit(main())
