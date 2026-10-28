# PENDING — Real-time Hinglish voice assistant (AVCS-v1 + cloud LLM)

Goal: extend AVCS-v1 into a real-time conversational assistant.
- Brain = cloud LLM (OpenAI-compatible), Hinglish, streamed in real time.
- Screen perception = pure pixel based (NO OCR).
- No cloud GPU for training; local CPU only.

User decision: **Option 3 — cloud LLM**. Wire `ConversationBrain` to a real
OpenAI-compatible endpoint so the assistant actually speaks Hinglish.

## Status

### DONE
- T1 audit of AVCS-v1 end-to-end (report delivered).
- T2 completeness verdict: NOT complete (RuleBrain turns every utterance into a
  "click"; no trained text-generation model wired to voice; ForgeLM not wired).
- Fixed blocking bug: `EyeStream(api_key=...)` -> `EyeStream()` at 3 call sites
  in `avcs_sdk/run_live_agent.py`. `--check` now honest, exits 2 without weights.
- Added `tests/test_sdk_live_agent.py` (3 tests pass).
- Updated AGENTS.md test counts to measured reality.
- Committed: "Fix live-agent entrypoint and wire presence_floor to the kernel".

### DONE — cloud LLM path (T3, Option C)
- `tests/test_cloud_llm_path.py`: real local HTTP server speaking OpenAI
  `/chat/completions` (stream + non-stream) drives the REAL
  `LLMReasoningProvider` + `ConversationBrain` (existing voice tests only use
  fake providers, so the real HTTP path was untested). Covers non-stream parse,
  SSE fragments, Hinglish end-to-end with sentinel + action recovery, retry on
  429/5xx, exhaustion, Bearer auth, token telemetry, and (new) wire-level
  assertions that the language instruction reaches the actual HTTP body.
- **Bug fixed: `CONVERSATION_SYSTEM_PROMPT` was dead code.** `plan_step` had no
  `system_prompt` parameter (only `stream_step` did), so the non-streaming voice
  path silently answered with the *silent System-2 planner* prompt. `plan_step`
  now takes `system_prompt` on the ABC, `MockReasoningProvider` and
  `LLMReasoningProvider`, and the brain passes the conversation prompt.

### DONE — Hinglish is now enforced, not merely permitted
- `ConversationBrain(language=...)`: `hinglish` (default) / `english` / `auto`.
  The policy is appended to whichever prompt is in use, so the two prompts stay
  one source of truth. Unknown values raise instead of silently defaulting.
- `avcs/voice/demo.py --voice-language` now drives BOTH the mouth and the brain
  (it previously only changed the espeak voice).

### DONE — ears without a sound card (`--audio`)
- `avcs/voice/demo.py --audio FILE[,FILE...]|<dir>` feeds one WAV per turn into
  the same `record_audio` hook the microphone uses, so the full
  ears -> brain -> mouth loop runs on a machine with no audio device. The
  reader returns silence when the list is exhausted rather than repeating the
  last utterance forever.
- `--check-llm`: probes the configured endpoint once and exits **0 only on
  success**, **2** with no key or an unreachable endpoint (same honesty rule as
  `avcs_sdk --check`). Measured: real mock endpoint -> exit 0, "connected";
  no key -> exit 2; connection refused -> exit 2.

### Measured end-to-end (mock cloud endpoint, no cloud key needed)
Real WAV -> Vosk STT -> real streaming HTTP LLM -> Hinglish reply streamed
sentence-by-sentence -> action parsed -> kernel DENY (no eyes attached):
```
brain_has_key: True | ears: vosk | mouth: espeak-ng
you   : click a button
avcs  : Ji sir, main save button daba raha hoon.   (brain=llm)
action: click target='save button'
gate  : DENY (grounder produced no candidate point)
timing: listen=888ms brain=28ms
```
`brain=llm`, not `rule_fallback`: the cloud path is live, not a fallback.

### BLOCKED (needs the user, not code)
- **A real cloud LLM key.** No `AVCS_LLM_API_KEY` / `OPENAI_API_KEY` exists in
  this environment. The platform's managed proxy
  (`OH_LLM_API_KEY_REFRESH_URL`) returns **401 Invalid session API key**, so it
  cannot be used either. Everything up to the credential is verified against a
  real OpenAI-protocol server; only the real key is missing.
  Run `python3 -m avcs.voice.demo --check-llm` once a key is set.
- **Screen perception with real eyes** needs `runs/weights/*.pt` (gitignored)
  and, for real actuation, `xdotool` + `DISPLAY`. The gate is fail-closed, so
  actions are refused with a clear reason instead of acting blind.

### TODO (ordered)
1. ~~Finish `tests/test_cloud_llm_path.py`~~ (done).
3. ~~Verify Hinglish streaming conversation end-to-end~~ — verified against a
   real OpenAI-protocol server; **needs a real key** to verify against a live
   vendor (blocked above).
4. ~~Wire voice I/O (STT + TTS)~~ (done). Vosk model installed, espeak-ng
   installed, `--audio` path added for machines without a microphone.
5. Document the runnable demo + env vars (done, see "Run" below).

## Run

```bash
# Does the brain actually work? Exit 0 only if the key/endpoint/model do.
export AVCS_LLM_API_KEY=...            # and AVCS_LLM_ENDPOINT / AVCS_LLM_MODEL
python3 -m avcs.voice.demo --check-llm

# Full voice loop from recorded WAV files (no microphone needed)
export AVCS_VOSK_MODEL=/path/to/vosk-model-small-en-us-0.15
python3 -m avcs.voice.demo --audio turn1.wav,turn2.wav --turns 2 --grant click

# Live microphone conversation, Hinglish, streaming speech
python3 -m avcs.voice.demo --listen --loop --grant click

# Same with real screen perception and real actuation (opt-in, dangerous)
python3 -m avcs.voice.demo --listen --loop --eyes --ground --live --grant click
```

## Env / deps
- Repo: `/workspace/project/AVCS-v1`; Python: `/usr/local/bin/python`.
- Runtime deps: torch numpy psutil cryptography fastapi httpx requests Pillow.
- Dev: pytest pytest-asyncio build wheel.
- STT: Vosk model installed at `/tmp/vosk/vosk-model-small-en-us-0.15`
  (`AVCS_VOSK_MODEL`). TTS: espeak-ng at `/usr/bin/espeak-ng`. No audio device,
  so `--audio` is the way to exercise the loop here.
- LLM key: **not set**. `AVCS_LLM_API_KEY` (+ `AVCS_LLM_ENDPOINT`,
  `AVCS_LLM_MODEL`) is what the provider reads.

## Key files
- `avcs/reasoning/provider.py` — `LLMReasoningProvider` (OpenAI-compatible,
  `stream_step` SSE + non-stream fallback, retries on 429/5xx). `plan_step` and
  `stream_step` both take `system_prompt`.
- `avcs/voice/brain.py` — `ConversationBrain(provider=..., language=...)`.
- `avcs/voice/demo.py` — CLI: `--check-llm`, `--audio`, `--listen`, `--loop`,
  `--voice-language`, `--grant`, `--live`.
- `tests/test_cloud_llm_path.py` — real-HTTP cloud path tests.
- `avcs_sdk/run_live_agent.py` — `run_agent(goal, max_turns, dry_run, ...)`.
