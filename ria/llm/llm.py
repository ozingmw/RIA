"""
LLM 모듈 - OpenAI Agents SDK를 이용한 대화 에이전트
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

from .prompt import (
    build_prompt_text,
    get_prompt as get_prompt_data,
    update_prompt as update_prompt_data,
)

# .env 파일 로드
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)


@function_tool
def get_prompt() -> Dict[str, Any]:
    """현재 프롬프트 JSON을 반환한다."""
    return get_prompt_data()


@function_tool
def update_prompt(
    name: Optional[str] = None,
    style: Optional[str] = None,
    rules_set: Optional[List[str]] = None,
    rules_append: Optional[List[str]] = None,
    samples_append: Optional[List[Dict[str, str]]] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """프롬프트 JSON을 부분 업데이트한다."""
    return update_prompt_data(
        name=name,
        style=style,
        rules_set=rules_set,
        rules_append=rules_append,
        samples_append=samples_append,
        notes=notes,
    )


@function_tool(name_override="build_prompt_text")
def build_prompt_text_tool() -> str:
    """현재 프롬프트 JSON을 system prompt 문자열로 변환한다."""
    return build_prompt_text()


def create_agent() -> Agent:
    return Agent(
        name="리아",
        instructions=build_prompt_text(),
        tools=[get_prompt, update_prompt, build_prompt_text_tool],
    )


def chat(message: str) -> str:
    """
    메시지를 처리하고 응답을 반환합니다.

    Args:
        message: 사용자 입력 메시지

    Returns:
        에이전트의 응답 텍스트
    """
    agent = create_agent()
    result = Runner.run_sync(agent, message)
    return result.final_output
