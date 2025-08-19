from langgraph_supervisor import create_supervisor

from estalan.llm.utils import create_chat_model
from estalan.agent.graph.requirement_collection_agent.state import RequirementCollectionAgentState
from estalan.agent.graph.requirement_collection_agent.prompt import prompt_supervisor2, SLIDE_GENERATION_TASK_PROMPT
from estalan.agent.graph.requirement_collection_agent.question_generation_agent import create_question_generation_agent
from estalan.agent.graph.requirement_collection_agent.requirement_analysis_agent import create_requirement_analysis_agent 



def create_requirement_collection_agent(predefined_question=[], name="requirement_collection_agent"):
    """요구사항 수집 에이전트 생성"""

    # 노드 생성
    question_generation_agent = create_question_generation_agent(predefined_question=predefined_question, name="question_generation_agent")
    requirement_analysis_agent = create_requirement_analysis_agent(name="requirement_analysis_agent")

    # 그래프 구성
    requirement_collection_agent = create_supervisor(
        [question_generation_agent, requirement_analysis_agent],
        model=create_chat_model(provider="azure_openai", model="gpt-4.1"),
        prompt=SLIDE_GENERATION_TASK_PROMPT + prompt_supervisor2,
        state_schema=RequirementCollectionAgentState,
        supervisor_name=name + "_supervisor",
    ).compile(name=name)

    return requirement_collection_agent
