from estalan.llm.utils import create_chat_model

from estalan.prebuilt.react_agent import create_react_agent
from estalan.prebuilt.requirement_analysis_agent.state import RequirementCollectionAgentState, RequirementCollectionState
from estalan.prebuilt.requirement_analysis_agent.tools import Tools
from estalan.prebuilt.requirement_analysis_agent.context_schema import Configuration
from estalan.prebuilt.requirement_analysis_agent.prompt import PROMPT_REQUIREMENT_ANALYSIS
from estalan.agent.base.state import state_to_json_pretty, state_to_json_compact


def post_agent_node(state):
    requirements = state.get('requirements', [])
    docs = state_to_json_compact(requirements)
    return {"requirements_docs": docs}


def create_requirement_analysis_agent(model=None, name="requirement_analysis_agent"):
    if model is None:
        model = create_chat_model(provider="azure_openai", model="gpt-4o")

    agent = create_react_agent(
        model=model,
        tools=Tools,
        prompt=PROMPT_REQUIREMENT_ANALYSIS,
        state_schema=RequirementCollectionAgentState,
        response_format=RequirementCollectionState,
        post_agent_node=post_agent_node,
        name=name
    )
    return agent


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    graph = create_requirement_analysis_agent()
    result = graph.invoke({"messages": "나는 3박 4일 제주도 여행을 계획 중이야."})
    print(result)