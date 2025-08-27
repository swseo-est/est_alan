from typing import TypedDict, Annotated
from estalan.agent.base.state import BaseAlanAgentState


# 질문 생성 에이전트 관련 데이터 구조
class Question(TypedDict):
    """
    질문 정보를 담는 데이터 구조

    Attributes:
        question_id: 질문 고유 식별자
        question: 질문 내용
        category: 질문 카테고리
        status: 질문 상태 ('pending' | 'asked' | 'answered')
    """
    question_id: str
    question: str
    category: str
    status: str  # 'pending' | 'asked' | 'answered'


# 요구사항 수집 에이전트 관련 데이터 구조
class Requirement(TypedDict):
    """
    요구사항 정보를 담는 데이터 구조

    Attributes:
        requirement_id: 요구사항 고유 식별자
        category: 요구사항 카테고리
        detail: 상세한 요구사항 설명
        priority: 우선순위 (High/Medium/Low)
        status: 요구사항 상태
        impact: 영향받는 시스템/프로세스/사용자 목록
        origin: 요구사항 출처 ('user' | 'question' | 'inferred')
    """
    requirement_id: str
    category: str
    detail: str
    priority: str
    status: str
    impact: list[str]
    origin: str  # 'user' | 'question' | 'inferred' (선택)


class RequirementCollectionState(TypedDict):
    """
    요구사항 수집 상태 정보

    Attributes:
        requirements: 수집된 모든 요구사항 목록
        requirements_docs: 요구사항 문서화 결과
    """
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
    """
    요구사항 수집 에이전트 상태 정보
    기본 에이전트 상태를 상속받아 요구사항 관련 필드를 추가합니다.

    Attributes:
        requirements: 수집된 모든 요구사항 목록
        requirements_docs: 요구사항 문서화 결과
    """
    requirements: list[Requirement]  # 수집된 모든 요구사항
    requirements_docs: str
    requirements: Annotated[list[Requirement], update_requirement]  # 수집된 모든 요구사항
    requirements_docs: str  # Markdown 형태




