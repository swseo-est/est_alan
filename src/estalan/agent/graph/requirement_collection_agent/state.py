from typing import List, Annotated, TypedDict, Optional, Dict, Any
from langgraph.prebuilt.chat_agent_executor import AgentState


# Question Generation Agent
class Question(TypedDict):
    question: str

class QuestionGenerationOutput(TypedDict):
    questions: list[Question]  # 사전정의 질문


# Requirement Analysis Agent
class ExtractRequirementOutput(TypedDict):
    requirements: list[dict]  # 추출된 요구사항들 (origin, summary, detail, update_existing 포함)


# Requirement Collection Agent
class Requirement(TypedDict):
    requirement_id: str  # 요구사항 id
    origin: str  # 'question' | 'user'

    summary: str
    detail: str


class RequirementCollectionAgentState(AgentState):
    requirements: list[Requirement]  # 수집된 모든 요구사항(origin으로 구분)
    questions: list[Question]  # 사전정의 질문

    initialization: bool
    is_complete: bool

    last_step: str
    next_step: str

