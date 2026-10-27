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
- Test suite: 109 passed, 2 skipped.
- Updated AGENTS.md test counts to measured reality.
- Committed: "Fix live-agent entrypoint and wire presence_floor to the kernel".

### IN PROGRESS
- T3 Option C: wire cloud LLM into `ConversationBrain` end-to-end.

### DONE
- Finished `tests/test_cloud_llm_path.py`: real local HTTP server speaking
  OpenAI `/chat/completions` (stream + non-stream) drives the REAL
  `LLMReasoningProvider` + `ConversationBrain` (existing voice tests only use
  fake providers, so the real HTTP path was untested).
  - Added `_stream()` SSE chunk emitter + `openai_server` pytest fixture
    (background thread on a free port).
  - Tests: non-stream `plan_step` parse; SSE `stream_step` fragments; Hinglish
    `ConversationBrain.respond_stream` end-to-end (sentinel + action recovery);
    429/5xx retry+backoff then success; retry exhaustion raises; Bearer auth
    header; token usage telemetry accumulation. 11 tests pass.
- Full suite: 120 passed, 2 skipped.

### TODO (ordered)
1. ~~Finish `tests/test_cloud_llm_path.py`~~ (done).
3. Verify reliable Hinglish streaming conversation end-to-end with a real key
   (`OPENAI_API_KEY` / endpoint / model via env).
4. Wire voice I/O: STT (Vosk model dir missing -> `AVCS_VOSK_MODEL`) and TTS
   (espeak-ng present at /usr/bin/espeak-ng). No soundcard/mic, no DISPLAY.
5. Document the runnable demo + env vars.

## Env / deps
- Repo: `/workspace/project/AVCS-v1`; Python: `/usr/local/bin/python`.
- Runtime deps: torch numpy psutil cryptography fastapi httpx requests Pillow.
- Dev: pytest pytest-asyncio build wheel.
- STT: Vosk (model dir MISSING). TTS: espeak-ng present. No audio device.
- No LLM API key currently set in env.

## Key files
- `avcs/reasoning/provider.py` — `LLMReasoningProvider` (OpenAI-compatible,
  `stream_step` SSE + non-stream fallback, retries on 429/5xx).
- `avcs/voice/brain.py` — `ConversationBrain(provider=...)`.
- `avcs_sdk/run_live_agent.py` — `run_agent(goal, max_turns, dry_run, ...)`.
