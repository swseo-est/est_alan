from typing import TypedDict, Annotated
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


def update_requirement(existing_requirements, new_requirements):
    # 기존 요구사항과 새로 생성된 요구사항을 통합
    # 중복 제거: requirement_id를 기준으로 중복 제거
    unique_requirements = {}
    for req in existing_requirements + new_requirements:
        # 순서 상 new_requirements가 뒤에 호출되어 덮어씌어짐
        req_id = req.get('requirement_id')
        unique_requirements[req_id] = req

    # 중복 제거된 요구사항 리스트로 변환
    final_requirements = list(unique_requirements.values())
    return final_requirements


class RequirementCollectionAgentState(BaseAlanAgentState):
    requirements: Annotated[list[Requirement], update_requirement]  # 수집된 모든 요구사항
    requirements_docs: str  # Markdown 형태




