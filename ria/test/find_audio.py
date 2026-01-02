import pyaudio

p = pyaudio.PyAudio()

print("=== 오디오 장치 목록 ===")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"Index {i}: {info['name']}")
    print(f"  - 입력 채널 수: {info['maxInputChannels']}")
    print(f"  - 출력 채널 수: {info['maxOutputChannels']}")
    print(f"  - 기본 샘플레이트: {info['defaultSampleRate']}")
    print()

p.terminate()