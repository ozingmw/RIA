"""
OpenAI TTS 기반 음성 합성 및 재생 유틸리티
"""

import threading
from pathlib import Path
from typing import Sequence

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from scipy import signal

# .env 파일 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# OpenAI 클라이언트 생성
client = OpenAI()

# 오디오 출력 설정
MODEL_NAME = "gpt-4o-mini-tts"  # OpenAI TTS 음성 합성 모델
OUTPUT_SAMPLE_RATE = 24000
OUTPUT_CHANNELS = 2
OUTPUT_DEVICE_NAME = [14, 15]


def synthesize(text: str, voice: str = "nova") -> bytes:
    """
    텍스트를 PCM 음성 데이터로 변환합니다.

    Args:
        text: 변환할 텍스트
        voice: 음성 종류 (alloy, echo, fable, onyx, nova, shimmer)

    Returns:
        PCM 오디오 데이터 (bytes)
    """
    response = client.audio.speech.create(
        model=MODEL_NAME,
        voice=voice,
        input=text,
        response_format="pcm",  # PCM 형식으로 받기
    )
    return response.content


def play(
    text: str,
    voice: str = "shimmer",
    device_name: str
    | int
    | tuple[str, int]
    | Sequence[str | int | tuple[str, int]] = OUTPUT_DEVICE_NAME,
):
    """
    텍스트를 음성으로 변환하여 바로 재생합니다.

    Args:
        text: 변환할 텍스트
        voice: 음성 종류 (alloy, echo, fable, onyx, nova, shimmer)
        device_name: 출력 장치 이름/인덱스/(이름, 샘플레이트) 또는 그 목록
    """
    # device_name을 리스트로 표준화해 여러 장치를 동시에 지원
    if device_name is None:
        # (input, output) 구조에서 output만 사용
        default_output = sd.default.device[1] if sd.default.device else None
        target_names: list[str | int | tuple[str, int]] = [default_output]
    elif isinstance(device_name, (str, int, tuple)):
        target_names = [device_name]
    else:
        target_names = list(device_name)
    if not target_names or target_names == [None]:
        raise RuntimeError("최소 1개의 출력 장치를 지정해야 합니다.")

    tts_pcm = synthesize(text, voice=voice)
    base_audio = np.frombuffer(tts_pcm, dtype=np.int16)

    # 각 장치로 동시에 출력
    threads: list[threading.Thread] = []

    for selector in target_names:
        device_info = sd.query_devices(selector)
        device_sample_rate = int(device_info["default_samplerate"])

        # 리샘플링 (장치 샘플레이트와 다르면 변환)
        audio_for_device = base_audio
        if device_sample_rate != OUTPUT_SAMPLE_RATE:
            num_samples = int(len(base_audio) * device_sample_rate / OUTPUT_SAMPLE_RATE)
            audio_for_device = signal.resample(base_audio, num_samples).astype(np.int16)

        # sounddevice는 float32, shape (frames, channels) 필요
        audio_float_mono = audio_for_device.astype(np.float32) / 32768.0
        # 모노 신호를 원하는 채널 수로 복제
        audio_float = np.repeat(audio_float_mono[:, None], OUTPUT_CHANNELS, axis=1)

        def _play_target(play_data: np.ndarray, rate: int, target_device):
            # Bind data/rate/device per thread to avoid late-binding bugs
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


# def list_output_devices():
#     """사용 가능한 출력 장치 목록을 출력합니다."""
#     print("사용 가능한 출력 장치:")
#     print("-" * 60)
#     for i, dev in enumerate(sd.query_devices()):
#         if dev["max_output_channels"] > 0:
#             name = dev["name"]
#             channels = dev["max_output_channels"]
#             rate = int(dev["default_samplerate"])
#             print(f"  [{i:2d}] {name} (ch:{channels}, rate:{rate})")
#     print("-" * 60)


# if __name__ == "__main__":
#     # 테스트
#     print("TTS 테스트")
#     print("-" * 40)

#     # 출력 장치 목록 표시
#     list_output_devices()
#     print()

#     test_text = "안녕하세요! 저는 리아입니다. 만나서 반갑습니다."
#     print(f"텍스트: {test_text}")
#     print("[재생 중...]")

#     play(test_text)
#     print("[재생 완료]")
