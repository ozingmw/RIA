import threading
from collections import deque

import numpy as np
import pyaudio

# 오디오 설정 상수
SAMPLE_RATE = 16000  # 16kHz (음성 인식에 최적)
CHUNK_SIZE = 512  # 한 번에 읽어올 샘플 수 (약 32ms)
CHANNELS = 1  # 모노
FORMAT = pyaudio.paInt16  # 16비트 정수
BUFFER_SECONDS = 2  # 순환 버퍼 크기 (초)
INPUT_DEVICE_INDEX = 1  # 마이크(Realtek High Definition Audio) 장치 인덱스

# 순환 버퍼: 최근 2초간의 오디오 데이터 유지 (호출어 인식 시점 보정용)
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_SECONDS / CHUNK_SIZE)
audio_buffer = deque(maxlen=BUFFER_SIZE)

# 스트리밍 상태 관리
stream_lock = threading.Lock()


class AudioStreamer:
    """실시간 오디오 스트리밍 클래스"""

    def __init__(self):
        self.p = None
        self.stream = None
        self.is_running = False

    def start(self):
        """오디오 스트림 시작"""
        self.p = pyaudio.PyAudio()

        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=INPUT_DEVICE_INDEX,
                frames_per_buffer=CHUNK_SIZE,
            )
            self.is_running = True
            print(
                f"[오디오 스트림 시작] 장치: {INPUT_DEVICE_INDEX}, 샘플레이트: {SAMPLE_RATE}Hz"
            )
            return True
        except Exception as e:
            print(f"[오류] 오디오 스트림 시작 실패: {e}")
            return False

    def read_chunk(self):
        """오디오 청크 하나 읽기"""
        if not self.is_running or self.stream is None:
            return None

        try:
            data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
            # bytes를 numpy 배열로 변환 (int16)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            return audio_chunk
        except Exception as e:
            print(f"[오류] 오디오 읽기 실패: {e}")
            return None

    def stop(self):
        """오디오 스트림 종료"""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        print("[오디오 스트림 종료]")


def add_to_buffer(chunk):
    """순환 버퍼에 오디오 청크 추가"""
    with stream_lock:
        audio_buffer.append(chunk)


def get_buffer_audio():
    """순환 버퍼에 저장된 오디오 데이터를 하나의 배열로 반환"""
    with stream_lock:
        if len(audio_buffer) == 0:
            return np.array([], dtype=np.int16)
        return np.concatenate(list(audio_buffer))


def clear_buffer():
    """순환 버퍼 초기화"""
    with stream_lock:
        audio_buffer.clear()