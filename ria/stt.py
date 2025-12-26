"""
STT (Speech-to-Text) 모듈

Faster-Whisper를 사용한 음성 인식
"""

import numpy as np
from audio import SAMPLE_RATE
from faster_whisper import WhisperModel

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
