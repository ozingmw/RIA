"""
OpenAI TTS 및 Supertone supertonic-2 백엔드를 지원하는 음성 합성/재생 유틸리티.
config/tts.yaml 의 backend 설정으로 사용 백엔드를 선택합니다.
- backend: openai | supertonic2
"""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING, cast

import numpy as np
import sounddevice as sd
import yaml
from dotenv import load_dotenv
from scipy import signal

# .env 파일 로드 (OpenAI 키 등)
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    load_dotenv(env_path)

# 구성 로드
cfg_path = Path(__file__).resolve().parents[2] / "config" / "tts.yaml"
with cfg_path.open("r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f) or {}

BACKEND = str(_cfg.get("backend", "openai"))
OUTPUT_DEVICE_NAME = _cfg.get("output_device_name")
OUTPUT_CHANNELS = int(_cfg.get("output_channels", 2))

_openai_cfg = _cfg.get("openai", {})
OPENAI_MODEL_NAME = str(_openai_cfg.get("model_name", "gpt-4o-mini-tts"))
OPENAI_SAMPLE_RATE = int(_openai_cfg.get("output_sample_rate", 24000))
OPENAI_VOICE_DEFAULT = str(_openai_cfg.get("voice", {}).get("play_default", "shimmer"))

_super_cfg = _cfg.get("supertonic2", {})
SUPER_STYLE_DEFAULT = str(_super_cfg.get("voice_style", "F1"))
SUPER_LANG_DEFAULT = str(_super_cfg.get("lang", "ko"))
SUPER_TOTAL_STEP = int(_super_cfg.get("total_step", 5))
SUPER_SPEED = float(_super_cfg.get("speed", 1.05))
SUPER_SAMPLE_RATE = int(_super_cfg.get("sample_rate", 44100))
SUPER_MAX_CHUNK = int(_super_cfg.get("max_chunk_length", 120))
SUPER_SILENCE = float(_super_cfg.get("silence_duration", 0.3))
SUPER_AUTO_DOWNLOAD = bool(_super_cfg.get("auto_download", True))


@lru_cache(maxsize=1)
def _get_openai_client():
    from openai import OpenAI

    return OpenAI()


def _synthesize_openai(text: str, voice: str | None) -> tuple[np.ndarray, int]:
    """OpenAI TTS로 합성하여 float32 mono 오디오와 샘플레이트를 반환합니다."""
    client = _get_openai_client()
    response = client.audio.speech.create(
        model=OPENAI_MODEL_NAME,
        voice=voice or OPENAI_VOICE_DEFAULT,
        input=text,
        response_format="pcm",
    )
    audio_int16 = np.frombuffer(response.content, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    return audio_float, OPENAI_SAMPLE_RATE


@lru_cache(maxsize=1)
def _get_supertonic_client():
    try:
        from supertonic import TTS as SuperTonicTTS
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "supertonic 백엔드 사용 시 `pip install supertonic` 이 필요합니다."
        ) from exc

    return SuperTonicTTS(auto_download=SUPER_AUTO_DOWNLOAD)


@lru_cache(maxsize=32)
def _get_supertonic_voice_style(name: str):
    client = _get_supertonic_client()
    return client.get_voice_style(voice_name=name)


def _synthesize_supertonic(
    text: str, voice_style: str | None
) -> tuple[np.ndarray, int]:
    """Supertone supertonic-2로 합성하여 float32 mono 오디오와 샘플레이트를 반환합니다."""
    client = _get_supertonic_client()
    style_name = voice_style or SUPER_STYLE_DEFAULT
    style = _get_supertonic_voice_style(style_name)
    wav, _duration = client.synthesize(
        text,
        voice_style=style,
        lang=SUPER_LANG_DEFAULT,
        total_steps=SUPER_TOTAL_STEP,
        speed=SUPER_SPEED,
        max_chunk_length=SUPER_MAX_CHUNK,
        silence_duration=SUPER_SILENCE,
    )
    audio = np.asarray(wav).reshape(-1).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    sample_rate = getattr(client, "sample_rate", SUPER_SAMPLE_RATE)
    return audio, sample_rate


def synthesize(text: str, voice: str | None = None) -> bytes:
    """
    텍스트를 PCM 음성 데이터로 변환합니다.

    Args:
        text: 변환할 텍스트
        voice: OpenAI 백엔드에서는 음성 이름, supertonic-2 백엔드에서는 voice_style 이름

    Returns:
        PCM 오디오 데이터 (bytes)
    """
    if BACKEND == "openai":
        audio_float, sample_rate = _synthesize_openai(text, voice)
    elif BACKEND == "supertonic2":
        audio_float, sample_rate = _synthesize_supertonic(text, voice)
    else:  # pragma: no cover - config error
        raise ValueError(f"지원하지 않는 TTS backend: {BACKEND}")

    # bytes로 반환 (int16 PCM)
    audio_int16 = np.clip(audio_float * 32768.0, -32768, 32767).astype(np.int16)
    return audio_int16.tobytes()


def _synthesize_float(text: str, voice: str | None = None) -> tuple[np.ndarray, int]:
    """float32 오디오와 샘플레이트를 반환 (재생용 내부 헬퍼)."""
    if BACKEND == "openai":
        return _synthesize_openai(text, voice)
    if BACKEND == "supertonic2":
        return _synthesize_supertonic(text, voice)
    raise ValueError(f"지원하지 않는 TTS backend: {BACKEND}")


def play(
    text: str,
    voice: str | None = None,
    device_name: str
    | int
    | tuple[str, int]
    | Sequence[str | int | tuple[str, int]]
    | None = OUTPUT_DEVICE_NAME,
):
    """
    텍스트를 음성으로 변환하여 바로 재생합니다.

    Args:
        text: 변환할 텍스트
        voice: OpenAI 백엔드에서는 음성 이름, supertonic-2 백엔드에서는 voice_style 이름
        device_name: 출력 장치 이름/인덱스/(이름, 샘플레이트) 또는 그 목록
    """
    if device_name is None:
        # (input, output) 구조에서 output만 사용
        default_output = sd.default.device[1] if sd.default.device else None
        target_names = [] if default_output is None else [default_output]
    elif isinstance(device_name, (str, int, tuple)):
        target_names = [device_name]
    else:
        target_names = list(device_name)
    if not target_names:
        raise RuntimeError("최소 1개의 출력 장치를 지정해야 합니다.")

    base_audio, base_rate = _synthesize_float(text, voice)

    # 각 장치로 동시에 출력
    threads: list[threading.Thread] = []

    for selector in target_names:
        device_info: dict[str, Any] = sd.query_devices(selector)  # type: ignore[assignment]
        device_sample_rate = int(
            cast(dict[str, Any], device_info)["default_samplerate"]
        )

        audio_for_device = base_audio
        if device_sample_rate != base_rate:
            num_samples = int(len(base_audio) * device_sample_rate / base_rate)
            audio_for_device = np.asarray(
                signal.resample(base_audio, num_samples), dtype=np.float32
            )

        # 모노 신호를 원하는 채널 수로 복제
        audio_float = np.repeat(audio_for_device[:, None], OUTPUT_CHANNELS, axis=1)

        def _play_target(play_data: np.ndarray, rate: int, target_device):
            with sd.OutputStream(
                device=target_device,
                samplerate=rate,
                channels=OUTPUT_CHANNELS,
                dtype="float32",
            ) as stream:
                stream.write(play_data)

        thread = threading.Thread(
            target=_play_target,
            args=(audio_float, device_sample_rate, selector),
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    # 모든 출력 완료 대기
    for thread in threads:
        thread.join()
