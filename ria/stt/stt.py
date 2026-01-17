"""
STT (Speech-to-Text) 모듈

Faster-Whisper를 사용한 음성 인식
"""

import time
from pathlib import Path

import numpy as np
import torch
import yaml
from faster_whisper import WhisperModel

from .audio import (
    CHUNK_SIZE,
    SAMPLE_RATE,
    AudioStreamer,
    add_to_buffer,
    clear_buffer,
    get_buffer_audio,
)

cfg_path = Path(__file__).resolve().parents[2] / "config" / "stt.yaml"
with cfg_path.open("r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f) or {}

model_cfg = _cfg["model"]
vad_cfg = _cfg.get("vad", {})
recording_cfg = _cfg.get("recording", {})


# Whisper 모델 설정 (config 기반)
MODEL_SIZE = str(model_cfg["size"])  # tiny, base, small, medium, large-v2, large-v3
DEVICE = str(model_cfg["device"])  # cpu 또는 cuda
COMPUTE_TYPE = str(model_cfg["compute_type"])  # float16, int8, int8_float16

VAD_TYPE = str(vad_cfg.get("type", "energy"))
VAD_SILERO_CFG = vad_cfg.get("silero", {})
VAD_ENERGY_CFG = vad_cfg.get("energy", {})


# 전역 모델 인스턴스 (한 번만 로드)
_model = None
_vad_model = None
_vad_device = "cpu"


def get_model():
    """Whisper 모델 인스턴스 반환"""
    global _model
    if _model is None:
        print(f"[Whisper 모델 로딩] {MODEL_SIZE} ({DEVICE}, {COMPUTE_TYPE})...")
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("[Whisper 모델 로딩 완료]")
    return _model


def _load_silero_model(device: str):
    global _vad_model, _vad_device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if _vad_model is None:
        torch.set_num_threads(1)
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
        )
        _vad_model = model
    if _vad_device != device:
        _vad_model.to(device)
        _vad_device = device
    _vad_model.reset_states()
    return _vad_model, device


def _silero_expected_samples():
    if SAMPLE_RATE == 16000:
        return 512
    if SAMPLE_RATE == 8000:
        return 256
    raise ValueError("Silero VAD requires 8000 or 16000 sample rate")


def _silero_prob(chunk: np.ndarray, device: str) -> float:
    if _vad_model is None:
        raise RuntimeError("Silero VAD model is not loaded")
    if len(chunk) != _silero_expected_samples():
        raise ValueError("Silero VAD chunk size mismatch")
    audio_float = chunk.astype(np.float32) / 32768.0
    audio_tensor = torch.from_numpy(audio_float).to(device)
    with torch.no_grad():
        return float(_vad_model(audio_tensor, SAMPLE_RATE).item())


def _frames_from_ms(duration_ms: int, samples_per_frame: int) -> int:
    if duration_ms <= 0:
        return 1
    frames = int((duration_ms / 1000.0) * SAMPLE_RATE / max(1, samples_per_frame))
    return max(1, frames)


def transcribe(audio_data: np.ndarray) -> str:
    """
    오디오 데이터를 텍스트로 변환

    Args:
        audio_data: int16 numpy 배열 (16kHz 모노)

    Returns:
        인식된 텍스트
    """
    if len(audio_data) == 0:
        return ""

    model = get_model()

    # int16 -> float32로 변환 (-1.0 ~ 1.0 범위)
    audio_float = audio_data.astype(np.float32) / 32768.0

    # Whisper로 변환
    segments, info = model.transcribe(
        audio_float,
        language="ko",  # 한국어
        beam_size=5,  # 빠른 처리를 위해 1로 설정
        vad_filter=True,  # VAD 필터 활성화 ( 무음 구간 제거 )
    )

    # 세그먼트 텍스트 합치기
    text = "".join(segment.text for segment in segments).strip()

    return text


def listen_and_transcribe(
    vad_threshold: float = 500,
    silence_duration: float = 1.5,
    max_record_seconds: float = 30,
):
    """마이크로부터 음성을 듣고 VAD로 구간을 잘라 Whisper로 텍스트 변환"""

    clear_buffer()

    streamer = AudioStreamer()
    if not streamer.start():
        return None

    print("\n[대기 모드] 듣고 있습니다... (말씀해 주세요)")

    recorded_chunks: list[np.ndarray] = []
    start_time = time.time()
    silence_start_time = None
    is_speech_detected = False
    use_silero = VAD_TYPE == "silero"
    speech_frames = 0
    silence_frames = 0

    silero_device = str(VAD_SILERO_CFG.get("device", "cpu"))
    start_threshold = float(VAD_SILERO_CFG.get("start_threshold", 0.6))
    end_threshold = float(VAD_SILERO_CFG.get("end_threshold", 0.35))
    start_trigger_frames = int(VAD_SILERO_CFG.get("start_trigger_frames", 3))
    end_trigger_frames = int(VAD_SILERO_CFG.get("end_trigger_frames", 12))
    min_speech_ms = int(VAD_SILERO_CFG.get("min_speech_ms", 300))
    max_utterance_s = float(VAD_SILERO_CFG.get("max_utterance_s", 15))
    pre_roll_ms = int(VAD_SILERO_CFG.get("pre_roll_ms", 500))

    if use_silero:
        try:
            _load_silero_model(silero_device)
        except Exception as exc:
            print(f"\n[Silero VAD 로딩 실패] {exc}")
            use_silero = False

    if VAD_ENERGY_CFG:
        vad_threshold = float(VAD_ENERGY_CFG.get("vad_threshold", vad_threshold))
        silence_duration = float(
            VAD_ENERGY_CFG.get("silence_duration", silence_duration)
        )

    initial_chunk = streamer.read_chunk()
    if initial_chunk is None:
        samples_per_frame = CHUNK_SIZE
    else:
        samples_per_frame = len(initial_chunk)

    if use_silero:
        vad_chunk_samples = _silero_expected_samples()
        if vad_chunk_samples != samples_per_frame:
            print(
                "\n[Silero VAD 경고] chunk_size가 512/256과 다릅니다. energy VAD로 전환합니다."
            )
            use_silero = False

    min_speech_frames = _frames_from_ms(min_speech_ms, samples_per_frame)
    min_speech_samples = int((min_speech_ms / 1000.0) * SAMPLE_RATE)

    try:
        while True:
            chunk = streamer.read_chunk()
            if chunk is None:
                break

            recorded_chunks.append(chunk)
            add_to_buffer(chunk)

            amplitude = np.abs(chunk).mean()
            level = int(amplitude / 500)
            bar = "█" * min(level, 50)
            status = "듣는 중..." if is_speech_detected else "대기 중..."
            print(f"\r[{status}] {bar:<50} {amplitude:>6.0f}", end="", flush=True)

            current_time = time.time()

            if current_time - start_time > (
                max_utterance_s if use_silero else max_record_seconds
            ):
                print("\n[최대 시간 초과]")
                break

            if use_silero:
                try:
                    prob = _silero_prob(chunk, silero_device)
                except Exception as exc:
                    print(f"\n[Silero VAD 오류] {exc}")
                    use_silero = False
                    continue

                if prob >= start_threshold:
                    speech_frames += 1
                    silence_frames = 0
                elif prob <= end_threshold:
                    silence_frames += 1
                    speech_frames = 0
                else:
                    speech_frames = 0
                    silence_frames = 0

                if not is_speech_detected and speech_frames >= start_trigger_frames:
                    is_speech_detected = True
                    speech_frames = 0
                    silence_frames = 0
                    print("\n[음성 감지됨] 녹음 시작...")
                    pre_roll_samples = int((pre_roll_ms / 1000.0) * SAMPLE_RATE)
                    pre_roll_audio = get_buffer_audio()
                    if pre_roll_samples > 0 and len(pre_roll_audio) > 0:
                        recorded_chunks = [pre_roll_audio[-pre_roll_samples:]]
                elif is_speech_detected and silence_frames >= end_trigger_frames:
                    if np.concatenate(recorded_chunks).size >= min_speech_samples:
                        print("\n[발화 종료 감지]")
                        break
            else:
                if amplitude > vad_threshold:
                    if not is_speech_detected:
                        is_speech_detected = True
                        print("\n[음성 감지됨] 녹음 시작...")
                    silence_start_time = None
                else:
                    if is_speech_detected:
                        if silence_start_time is None:
                            silence_start_time = current_time
                        elif current_time - silence_start_time > silence_duration:
                            print("\n[발화 종료 감지]")
                            break

    except KeyboardInterrupt:
        print("\n[중단됨]")
    finally:
        streamer.stop()

    if not recorded_chunks or not is_speech_detected:
        return None

    audio_data = np.concatenate(recorded_chunks)
    duration = len(audio_data) / SAMPLE_RATE
    print(f"\n[녹음 완료] {duration:.2f}초")

    print("[STT 변환 중...]")
    text = transcribe(audio_data)

    if text:
        print(f"[인식 결과] {text}")
    else:
        print("[인식 결과] (인식된 텍스트 없음)")

    return text
