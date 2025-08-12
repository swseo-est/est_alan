from typing import TypedDict, Literal, List, Dict, Any, Annotated
from langgraph.prebuilt.chat_agent_executor import AgentState, AgentStateWithStructuredResponse
from estalan.agent.base.state import private_state_updater

# Question Generation Agent
class Question(TypedDict):
    question: str

class QuestionGenerationOutput(TypedDict):
    questions: list[Question]  # 사전정의 질문


# Requirement Collection Agent

class Requirement(TypedDict):
    requirement_id: str
    category: str
    detail: str
    update_existing: bool


class RequirementCollectionAgentPrivateState(TypedDict):
    purpose: str
    requirements: str  # 마크다운 형식의 요구사항 문자열
    questions: list[Question]  # 사전정의 질문
    new_questions: list[Question]  # 새로 생성된 질문

    needs_more_questions: bool

    initialization: bool
    last_step: str


class RequirementCollectionAgentState(AgentStateWithStructuredResponse):
    requirement_collection_agent_state: Annotated[RequirementCollectionAgentPrivateState, private_state_updater]


