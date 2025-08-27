from langgraph.graph import StateGraph, START, END

from estalan.llm.utils import create_chat_model

from estalan.prebuilt.react_agent import create_react_agent
from estalan.prebuilt.requirement_analysis_agent.state import RequirementCollectionAgentState, RequirementCollectionState
from estalan.prebuilt.requirement_analysis_agent.tools import Tools
from estalan.prebuilt.requirement_analysis_agent.context_schema import Configuration
from estalan.prebuilt.requirement_analysis_agent.prompt import PROMPT_REQUIREMENT_ANALYSIS
from estalan.agent.base.state import state_to_json_pretty, state_to_json_compact
from estalan.prebuilt.requirement_analysis_agent.converter import requirements_to_markdown, markdown_to_requirements, validate_conversion
from estalan.messages.utils import create_ai_message
import json


def pre_agent_node(state):

    requirements_docs = state.get("requirements_docs", "")

    if requirements_docs:
        msg = "기존 등록된 요구사항은 다음과 같습니다. 새로운 유저 메시지를 분석해서 요구사항을 추가/업데이트 하세요\n"
        msg += requirements_docs
    else:
        msg = "현재 등록된 요구사항은 없습니다."

    return {"messages": [create_ai_message(content=msg)]}


def post_agent_node(state):
    """
    에이전트 실행 후 요구사항 상태를 정리하
    """
    requirements = state.get('requirements', [])
    # JSON과 Markdown 생성
    markdown_docs = requirements_to_markdown(requirements)
    
    return {
        "messages": state["messages"],
        "requirements_docs": markdown_docs,  # Markdown 형태
    }


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
        pre_agent_node=pre_agent_node,
        name=name
    )
    return agent

def create_input_subagent_node(name=None):
    if name is None:
        raise ValueError("name is required")
    
    def input_subagent_node(state):
        # 외부 state에서 internal state를 추출
        internal_state = state[name]
        internal_state["messages"] = state["messages"]
        return internal_state
    return input_subagent_node


def create_output_subagent_node(name=None):
    if name is None:
        raise ValueError("name is required")
    
    def output_subagent_node(state):
        # 내부 state를 외부 state로 변환

        private_state = state
        shared_state = {"requirements_docs": state["requirements_docs"]}

        return {
            "private_state" : {name: private_state},
            "shared_state" : {name: shared_state},
        }
    return output_subagent_node


def create_requirement_analysis_subagent(model=None, state_schema=None, name="requirement_analysis_agent", state_name="requirement_analysis_agent_state"):
    requirement_analysis_agent = create_requirement_analysis_agent(model, name)
    input_subagent_node = create_input_subagent_node(state_name)
    output_subagent_node = create_output_subagent_node(state_name)

    builder = StateGraph(state_schema)

    builder.add_node("input", input_subagent_node)
    builder.add_node("output", output_subagent_node)
    builder.add_node(name, requirement_analysis_agent)

    builder.add_edge(START, "input")
    builder.add_edge("input", name)
    builder.add_edge(name, "output")
    builder.add_edge("output", END) 

    graph = builder.compile(name=name)

    return graph


if __name__ == "__main__":
    from dotenv import load_dotenv
    from estalan.prebuilt.requirement_analysis_agent.prompt import test_prompt

    load_dotenv()

    agent = create_requirement_analysis_agent()
    result = agent.invoke({"messages": test_prompt})
    print(result["requirements_docs"])
