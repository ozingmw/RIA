from llm.llm import chat
from stt.stt import listen_and_transcribe
from tts.tts import play

def STT():
    text = listen_and_transcribe()
    return text


def LLM(text):
    response = chat(text)
    return response


def TTS(text):
    play(text)


def main():
    try:
        while True:
            text = STT()

            if not text:
                continue

            print(f"[STT]: {text}")
            response = LLM(text)

            TTS(response)

    except KeyboardInterrupt:
        print("Terminated")


if __name__ == "__main__":
    main()
