from typing import TypedDict
from langgraph.prebuilt.chat_agent_executor import AgentState


# Question Generation Agent
class Question(TypedDict):
    question: str

class QuestionGenerationOutput(TypedDict):
    questions: list[Question]  # 사전정의 질문

class RequirementCollectionAgentState(AgentState):
    purpose: str
    requirements: list[Requirement]  # 수집된 모든 요구사항(origin으로 구분)
    questions: list[Question]  # 사전정의 질문

    initialization: bool


