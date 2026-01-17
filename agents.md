# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-17 (Asia/Seoul)
**Commit:** n/a
**Branch:** n/a

## OVERVIEW
Voice-based Korean assistant. Pipeline: STT (faster-whisper, PyAudio) → LLM (openai-agents) → TTS (OpenAI gpt-4o-mini-tts). Config-driven via YAML.

## STRUCTURE
```
./
├── ria/            # runtime code (main loop, stt/tts/llm)
├── config/         # stt/tts/llm configs, device indices
├── .env.example    # API key example
├── pyproject.toml  # deps (faster-whisper, openai-agents, sounddevice, pyaudio)
├── unity/          # VRM/3D assets (non-code)
└── .venv/          # local env (ignore)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Run loop | ria/main.py | Blocking while-loop STT→LLM→TTS |
| STT impl | ria/stt/stt.py | VAD-threshold, max 30s capture, whisper decode |
| Audio I/O | ria/stt/audio.py | PyAudio stream, ring buffer |
| LLM agent | ria/llm/llm.py | openai-agents Agent/Runner, no tools/history |
| TTS synth | ria/tts/tts.py | OpenAI TTS PCM, multi-device playback |
| Configs | config/stt.yaml, config/tts.yaml | Device indices, sample rates, model names |
| Device listing | ria/test/find_audio.py | Lists PyAudio devices |
| Manual STT tests | ria/test/stt_test.py, audio_test.py | Ad-hoc scripts, not pytest |

## CODE MAP (key symbols)
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| main() | func | ria/main.py | Loop: STT→LLM→TTS |
| listen_and_transcribe | func | ria/stt/stt.py | VAD capture + whisper decode |
| transcribe | func | ria/stt/stt.py | Faster-whisper inference |
| AudioStreamer | class | ria/stt/audio.py | PyAudio stream lifecycle |
| chat | func | ria/llm/llm.py | openai-agents Runner call |
| synthesize | func | ria/tts/tts.py | OpenAI TTS PCM fetch |
| play | func | ria/tts/tts.py | Multi-device playback with resample |

## CONVENTIONS (project-specific)
- Config-first: stt/tts parameters from YAML; input/output device indices in config.
- Language: Korean logs/prompts; LLM runs Korean instructions.
- Sync pipeline: no async, no queues.
- .env expected under repo root (but llm.py loads ria/.env relative to module).

## ANTI-PATTERNS / MISSING
- No wake-word handling (ria/stt/kws.py empty).
- No conversation history; every LLM call stateless.
- No agent tools; only plain chat.
- No CI, no formal tests; only manual scripts.
- README empty; agents.md holds context.

## UNIQUE STYLES
- VAD uses mean amplitude threshold (500) and silence duration (1.5s).
- Audio buffer (2s) maintained but not used for wake word yet.
- TTS can target multiple devices concurrently; resamples per device SR.

## COMMANDS
```bash
uv sync
uv run python -m ria.main
# Device listing
uv run python -m ria.test.find_audio
```

## NOTES
- Dev environment: WSL/Ubuntu; runtime target: Windows. Adjust device indices accordingly.
- Faster-whisper configured for CUDA, float16, model "base" (config/stt.yaml).
- Output devices default to indices [14,15] (config/tts.yaml); adjust per machine.
- Unity assets are unrelated to runtime; keep excluded from doc tooling.
- Avoid committing .env/.venv; configs hold device IDs.
