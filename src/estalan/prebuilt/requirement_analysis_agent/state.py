from typing import TypedDict
from estalan.agent.base.state import BaseAlanAgentState


# Question Generation Agent
class Question(TypedDict):
    question_id: str
    question: str
    category: str
    status: str  # 'pending' | 'asked' | 'answered'


# Requirement Collection Agent
class Requirement(TypedDict):
    requirement_id: str
    category: str
    detail: str
    priority: str
    status: str

    impact: list[str]
    origin: str  # 'user' | 'question' | 'inferred' (선택)


class RequirementCollectionState(TypedDict):
    requirements: list[Requirement]  # 수집된 모든 요구사항
    requirements_docs: str  # Markdown 형태
    requirements_json: str  # JSON 형태


class RequirementCollectionAgentState(BaseAlanAgentState):
    requirements: list[Requirement]  # 수집된 모든 요구사항
    requirements_docs: str  # Markdown 형태
    requirements_json: str  # JSON 형태
