# 요구사항 분석 에이전트 설정 스키마 모듈
# 에이전트의 설정 정보를 정의하는 데이터 클래스를 제공합니다.

## Using Dataclasses
from dataclasses import dataclass, field
from typing import Annotated
from estalan.prebuilt.requirement_analysis_agent.prompt import PROMPT_REQUIREMENT_ANALYSIS

@dataclass(kw_only=True)
class Configuration:
    """
    요구사항 분석 에이전트의 설정 정보
    
    이 클래스는 에이전트의 동작을 제어하는 설정값들을 정의합니다.
    시스템 프롬프트와 언어 모델 정보를 포함합니다.
    """

    system_prompt: str = field(
        default=PROMPT_REQUIREMENT_ANALYSIS,
        metadata={
            "description": "에이전트의 상호작용에 사용할 시스템 프롬프트입니다. "
            "이 프롬프트는 에이전트의 컨텍스트와 동작을 설정합니다.",
            "json_schema_extra": {"langgraph_nodes": ["react_agent"]},
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="azure_openai/gpt-4o",
        metadata={
            "description": "에이전트의 주요 상호작용에 사용할 언어 모델의 이름입니다. "
            "형식: provider/model-name",
            "json_schema_extra": {"langgraph_nodes": ["react_agent"]},
        },
    )

