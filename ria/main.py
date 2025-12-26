import time

import numpy as np
from audio import (
    SAMPLE_RATE,
    AudioStreamer,
    add_to_buffer,
    clear_buffer,
    get_buffer_audio,
)
from llm import chat
from stt import transcribe
from tts import play


def STT():
    """
    음성을 듣고 텍스트로 변환합니다 (STT).
    VAD(Voice Activity Detection)를 사용하여 발화가 끝날 때까지 녹음합니다.
    """
    clear_buffer()  # 버퍼 초기화

    streamer = AudioStreamer()
    if not streamer.start():
        return None

    print("\n[대기 모드] 듣고 있습니다... (말씀해 주세요)")

    # VAD 설정
    VAD_THRESHOLD = 500  # 음성 감지 임계값 (환경에 따라 조절 필요)
    SILENCE_DURATION = 1.5  # 이 시간만큼 침묵이 지속되면 녹음 종료
    MAX_RECORD_SECONDS = 30  # 최대 녹음 시간

    recorded_chunks = []
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

            # 현재 오디오 레벨 계산
            amplitude = np.abs(chunk).mean()

            # 시각적 피드백 (레벨 미터)
            level = int(amplitude / 500)
            bar = "█" * min(level, 50)
            status = "듣는 중..." if is_speech_detected else "대기 중..."
            print(f"\r[{status}] {bar:<50} {amplitude:>6.0f}", end="", flush=True)

            current_time = time.time()

            # 최대 녹음 시간 초과 체크
            if current_time - start_time > MAX_RECORD_SECONDS:
                print("\n[최대 시간 초과]")
                break

            # 음성 감지 로직
            if amplitude > VAD_THRESHOLD:
                if not is_speech_detected:
                    is_speech_detected = True
                    print("\n[음성 감지됨] 녹음 시작...")
                silence_start_time = None  # 말하는 중이므로 침묵 타이머 리셋
            else:
                # 침묵 상태
                if is_speech_detected:
                    if silence_start_time is None:
                        silence_start_time = current_time
                    elif current_time - silence_start_time > SILENCE_DURATION:
                        print("\n[발화 종료 감지]")
                        break

    except KeyboardInterrupt:
        print("\n[중단됨]")
    finally:
        streamer.stop()

    # 녹음된 오디오가 없거나 너무 짧은 경우 처리
    if not recorded_chunks or not is_speech_detected:
        # print("\n[알림] 감지된 음성이 없습니다.")
        return None

    audio_data = np.concatenate(recorded_chunks)
    duration = len(audio_data) / SAMPLE_RATE
    print(f"\n[녹음 완료] {duration:.2f}초")

    # STT 변환
    print("[STT 변환 중...]")
    text = transcribe(audio_data)

    if text:
        print(f"[인식 결과] {text}")
    else:
        print("[인식 결과] (인식된 텍스트 없음)")

    return text


def LLM(text):
    """
    LLM을 통해 응답을 생성합니다.
    """
    if not text:
        return None

    print(f"[LLM 처리 중...]")
    response = chat(text)
    print(f"[LLM 응답] {response}")
    return response


def TTS(text):
    """
    텍스트를 음성으로 변환하여 재생합니다.
    """
    if not text:
        return None

    print("[TTS 재생 중...]")
    play(text)
    print("[TTS 완료]")


def main():
    print("음성 인식 에이전트 시작")

    # 대화 루프: STT -> LLM -> TTS 순환
    try:
        while True:
            text = STT()

            if not text:
                print("[알림] 음성을 인식하지 못했습니다. 다시 시도합니다.")
                continue

            response = LLM(text)

            if response:
                TTS(response)
            else:
                print("[알림] LLM 응답이 없습니다. 다시 시도합니다.")

    except KeyboardInterrupt:
        print("\n[종료] 사용자 중단으로 프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
