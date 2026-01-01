import time
import numpy as np

from audio import (
    AudioStreamer,
    add_to_buffer,
    get_buffer_audio,
    clear_buffer,
)
from stt import transcribe

def main():
    streamer = AudioStreamer()
    if not streamer.start():
        print("마이크 시작 실패")
        return
    
    print("🎙️ 2초 동안 말해보세요...")

    start_time = time.time()
    while time.time() - start_time < 2.0:
        chunk = streamer.read_chunk()
        if chunk is not None:
            add_to_buffer(chunk)

    streamer.stop()

    audio_data = get_buffer_audio()
    print(f"녹음된 샘플 수 : {len(audio_data)}")

    print("\n📝 STT 변환 중...")
    text = transcribe(audio_data)

    print("\n 인식 결과 : ")
    print(type(text))
    print(text if text else "(인식된 텍스트 없음)")

if __name__ == "__main__":
    main()