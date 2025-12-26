"""
LLM 모듈 - OpenAI Agents SDK를 이용한 대화 에이전트
"""

from pathlib import Path

from agents import Agent, Runner
from dotenv import load_dotenv

# .env 파일 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


# 기본 에이전트 생성
agent = Agent(
    name="Ria",
    instructions="""당신은 Ria라는 이름의 친절하고 도움이 되는 AI 어시스턴트입니다.
사용자와 자연스럽게 대화하며, 질문에 명확하고 간결하게 답변합니다.
한국어로 대화합니다.""",
)


def chat(message: str) -> str:
    """
    메시지를 처리하고 응답을 반환합니다.

    Args:
        message: 사용자 입력 메시지

    Returns:
        에이전트의 응답 텍스트
    """
    result = Runner.run_sync(agent, message)
    return result.final_output
