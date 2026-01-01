# RIA - AI 비서 프로젝트

## 📌 프로젝트 개요

**RIA(리아)**는 음성 기반 개인 AI 비서입니다. 사용자의 음성을 듣고, 호출어("리아")가 감지되면 이후 문장을 이해하여 적절한 응답을 음성으로 출력합니다.

### 핵심 흐름
```
[마이크 입력] → [호출어 "리아" 감지] → [후속 문장 STT 변환] → [LLM Agent 처리] → [TTS 음성 출력]
```

---

## 🎯 프로젝트 목표

1. **음성 인식 (STT)**: 마이크로 음성을 실시간 캡처하고 텍스트로 변환
2. **호출어 감지**: "리아"라는 이름이 불리면 활성화
3. **AI 에이전트**: 텍스트 입력을 받아 적절한 응답 또는 행동 수행
4. **음성 합성 (TTS)**: 응답 텍스트를 자연스러운 음성으로 출력
5. **확장성**: 다양한 도구(Tool)와 기능을 에이전트에 추가 가능

---

## 📁 프로젝트 구조

```
ria/
├── pyproject.toml      # 프로젝트 설정 및 의존성
├── .env                # API 키 등 환경 변수 (git 미포함)
├── agents.md           # 이 문서 (프로젝트 컨텍스트)
├── README.md           # 프로젝트 소개
└── ria/
    ├── __init__.py
    ├── main.py         # 메인 실행 파일 (STT → LLM → TTS 루프)
    ├── stt.py          # Speech-to-Text (Faster-Whisper 기반)
    ├── tts.py          # Text-to-Speech (OpenAI TTS 기반)
    └── llm.py          # LLM Agent (OpenAI Agents SDK 기반)
```

---

## ✅ 현재 구현된 기능

### 1. STT (stt.py)
- **Faster-Whisper** 모델을 사용한 음성 인식
- PyAudio를 통한 실시간 마이크 입력
- VAD(Voice Activity Detection) 기반 발화 구간 감지
  - 진폭 임계값: 500
  - 무음 지속 시간: 1.5초 후 발화 종료 판정
- 순환 버퍼 (최근 2초 오디오 보관)
- Whisper 모델 싱글톤 패턴으로 메모리 효율화

**설정값:**
- 샘플레이트: 16kHz
- 모델: `base` (cuda, float16)
- 입력 장치 인덱스: `1` (하드코딩됨)

### 2. LLM (llm.py)
- **OpenAI Agents SDK** 사용
- "리아"라는 이름의 한국어 AI 어시스턴트 에이전트
- 동기 방식 실행 (`Runner.run_sync`)

**현재 제한:**
- 대화 히스토리 미지원 (매 호출마다 새 세션)
- 도구(Tool) 미등록

### 3. TTS (tts.py)
- **OpenAI gpt-4o-mini-tts** 모델 사용
- PCM 형식으로 음성 합성
- 멀티 디바이스 동시 출력 지원
- 리샘플링 자동 처리

**설정값:**
- 출력 샘플레이트: 24kHz
- 출력 채널: 2 (스테레오)
- 출력 장치 인덱스: `[14, 15]` (하드코딩됨)
- 음성: shimmer

### 4. 메인 루프 (main.py)
- 무한 루프로 STT → LLM → TTS 순차 실행
- KeyboardInterrupt(Ctrl+C)로 종료

---

## ❌ 미구현 / 개선 필요 사항

### 🔴 핵심 미구현
| 항목             | 상태     | 설명                            |
| ---------------- | -------- | ------------------------------- |
| 호출어 감지      | ❌ 미구현 | "리아" 감지 후 활성화 기능 없음 |
| 대화 히스토리    | ❌ 미구현 | 맥락 유지 대화 불가             |
| Agent 도구(Tool) | ❌ 미구현 | 단순 응답만 가능                |

### 🟡 개선 필요
| 항목              | 상태     | 설명                      |
| ----------------- | -------- | ------------------------- |
| 설정 외부화       | ⚠️ 필요   | 장치 인덱스 등 하드코딩됨 |
| 에러 핸들링       | ⚠️ 부족   | API 실패 시 처리 없음     |
| 순환 버퍼 활용    | ⚠️ 미활용 | 선언만 되어 있음          |
| main.py 래퍼 함수 | ⚠️ 불필요 | 직접 호출로 단순화 가능   |

---

## 🛠️ 기술 스택

| 분류        | 기술                         | 버전     |
| ----------- | ---------------------------- | -------- |
| 런타임      | Python                       | ≥ 3.13   |
| STT         | faster-whisper               | ≥ 1.2.1  |
| LLM         | openai-agents                | ≥ 0.6.4  |
| TTS         | OpenAI API (gpt-4o-mini-tts) | -        |
| 오디오 입력 | pyaudio                      | ≥ 0.2.14 |
| 오디오 출력 | sounddevice                  | ≥ 0.5.3  |
| 기타        | numpy, scipy, python-dotenv  | -        |

---

## 🚀 실행 방법

```bash
# 의존성 설치
uv sync

# 환경 변수 설정 (.env 파일)
OPENAI_API_KEY=your_api_key_here

# 실행
uv run python -m ria.main
```

---

## 📋 다음 개발 우선순위

1. **[P0] 호출어 감지 기능**
   - 옵션 A: Whisper로 짧은 구간 계속 인식 → "리아" 포함 시 활성화
   - 옵션 B: OpenWakeWord / Porcupine 등 전용 라이브러리 사용

2. **[P1] 대화 히스토리 관리**
   - 세션별 대화 맥락 유지
   - 메모리 관리 (최근 N턴만 유지)

3. **[P2] API / 로컬 모델 선택 지원**
   - STT, LLM, TTS 각각 API 호출 또는 로컬 모델 방식 선택 가능하도록
   - **STT 옵션:**
     - API: OpenAI Whisper API
     - 로컬: Faster-Whisper (현재 구현됨)
   - **LLM 옵션:**
     - API: OpenAI GPT API (현재 구현됨)
     - 로컬: Ollama, llama.cpp 등
   - **TTS 옵션:**
     - API: OpenAI TTS API (현재 구현됨)
     - 로컬: Coqui TTS, VITS, StyleTTS2 등
   - 설정 파일에서 모드 전환 가능하도록 구현

4. **[P3] 설정 파일 분리**
   - config.py 또는 .env로 장치 인덱스 등 외부화

5. **[P4] Agent 도구 확장**
   - 날씨, 시간, 알람 등 기본 도구 추가

---

## 🔧 환경 설정 참고

### 오디오 장치 확인
```python
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(i, p.get_device_info_by_index(i)['name'])
```

```python
import sounddevice as sd
print(sd.query_devices())
```

---

## 📝 참고 사항

- Unity 폴더: VRM 아바타 관련 (Platinum 캐릭터) - 추후 3D 아바타 연동 예정
- CUDA 필요: Whisper 모델이 GPU에서 실행됨

---

*마지막 업데이트: 2026-01-01*
