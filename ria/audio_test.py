import time
import numpy as np
import matplotlib.pyplot as plt

from audio import (
    AudioStreamer,
    add_to_buffer,
    get_buffer_audio,
    clear_buffer,
)

def main():
    # 1. 스트림 시작
    streamer = AudioStreamer()
    if not streamer.start():
        print("스트림 시작 실패")
        return
    print("1단계 통과 : 마이크 열림")


    # 2. 청크 하나 읽기
    chunk = streamer.read_chunk()
    if chunk is None:
        print("오디오 읽기 실패")
        return
    print("2단계 통과 : 오디오 청크 수신")
    print(f"청크 크기 : {len(chunk)}")
    print(f"청크 타입 : {type(chunk)}")

    # 3. 소리 크기 확인 ( 말하면 값이 커져야 함 )
    print("\n\n\n📢 마이크에 말하기")
    for i in range(5):
        chunk = streamer.read_chunk()
        volume = np.mean(np.abs(chunk))
        print(f"청크 {i+1} 볼륨: {volume}")
        time.sleep(0.2)
    print("3단계 통과 : 소리 반응 확인")


    # 4. 버퍼에 쌓이는지 확인
    clear_buffer()
    for _ in range(10):
        chunk = streamer.read_chunk()
        add_to_buffer(chunk)
    buffer_audio = get_buffer_audio()
    print(f"4단계 통과 : 버퍼 데이터 길이 : {len(buffer_audio)}")

    # 5. 순환 버퍼 확인 ( 무한히 커지지 않아야 함 )
    for i in range(100):
        chunk = streamer.read_chunk()
        add_to_buffer(chunk)
        buffer_len = len(get_buffer_audio())
        print(f"{i+1}회차 버퍼 길이 : {buffer_len}")
        time.sleep(0.02)
    print("5단계 통과 : 버퍼 크기 제한 정상")

    # 6. 파형 시각화
    print("\n\n\n  파형 확인")
    chunk = streamer.read_chunk()
    plt.plot(chunk)
    plt.title("Audio Waveform ( 말하면 파형이 흔들려야 함 )")
    plt.show()
    print("6단계 통과 : 파형 시각화 성공")

    # 7. 스트림 종료
    streamer.stop()
    print("7단계 통과 : 스트림 정상 종료")

if __name__ == "__main__":
    main()