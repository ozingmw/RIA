"""
STT (Speech-to-Text) 모듈

Faster-Whisper를 사용한 음성 인식
"""

import time

import numpy as np
import pyaudio
from faster_whisper import WhisperModel
from ria.stt.audio import AudioStreamer, add_to_buffer, clear_buffer, SAMPLE_RATE


# Whisper 모델 설정
MODEL_SIZE = "base"  # tiny, base, small, medium, large-v2, large-v3
DEVICE = "cuda"  # cpu 또는 cuda
COMPUTE_TYPE = "float16"  # float16, int8, int8_float16
# DEVICE = "cpu"
# COMPUTE_TYPE = "int8"

# 전역 모델 인스턴스 (한 번만 로드)
_model = None


def get_model():
    """Whisper 모델 인스턴스 반환 (싱글톤)"""
    global _model
    if _model is None:
        print(f"[Whisper 모델 로딩] {MODEL_SIZE} ({DEVICE}, {COMPUTE_TYPE})...")
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("[Whisper 모델 로딩 완료]")
    return _model


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

            if current_time - start_time > max_record_seconds:
                print("\n[최대 시간 초과]")
                break

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
