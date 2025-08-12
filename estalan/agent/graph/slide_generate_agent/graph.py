from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import START, END, StateGraph
from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage, AIMessage

from typing import List, Annotated, TypedDict, Optional
from estalan.agent.graph.slide_generate_agent.planning_agent import create_planning_agent, Section
from estalan.agent.graph.slide_generate_agent.research_agent import create_research_agent
from estalan.agent.graph.slide_generate_agent.slide_design_agent import create_slide_create_agent
from estalan.llm.utils import create_chat_model
from estalan.utils import get_last_human_message
import asyncio
from langgraph.types import Send
import operator
import re


class SlideGenerateAgentState(AgentState):
    topic: str
    requirements: str
    num_sections: int
    num_slides: int
    design_prompt: str

    sections: List[Section]
    slides: Annotated[List[Section], operator.add]
    status: str

class OutputState(TypedDict):
    topic: str
    requirements: str

class HiddenCommandState(TypedDict):
    design_prompt: Optional[str]
    user_msg: Optional[str]
    is_hidden_command: bool

def parse_hidden_command_node(state):
    """
    히든 명령어를 파싱하는 노드
    히든 명령어 형식: 
    /add_design_prompt
    """

    last_message = get_last_human_message(state["messages"]).content

    if "/add_design_prompt" in last_message:
        return {"design_prompt": last_message, "messages": [AIMessage(content="히든 명령어가 적용되었습니다.", name="agent")]}
    else:
        return {"design_prompt": None}
    
def preprocessing_node(state):
    llm = create_chat_model(provider="azure_openai", model="gpt-4.1").with_structured_output(OutputState)

    msg = HumanMessage(content="슬라이드 topic과 유저 요구사항 requirement를 추출하세요.")
    updated_state = llm.invoke([msg] + state["messages"])
    return updated_state | {"num_sections": 5, "num_slides": 7}


class ExecutorInput(Section):
    pass

class ExecutorOutput(TypedDict):
    # executor에서 출력되는 결과
    slides: List[Section]


def post_processing_node(state):
    # 생성된 HTML을 test.html로 저장
    # with open(f"{state['idx']}.html", "w", encoding="utf-8") as f:
    #     f.write(state['html'])

    print("Post processing state:", state)

    # executor의 output이 ExecutorOutput 형태이므로 slides 필드를 그대로 반환
    return {"slides": [state]}

def create_slide_generate_graph():
    """슬라이드 생성 메인 그래프"""
    planning_agent = create_planning_agent()

    ## subroutine
    executor = StateGraph(ExecutorInput, output=ExecutorOutput)

    research_agent = create_research_agent()
    slide_create_agent = create_slide_create_agent()

    executor.add_node("research_agent", research_agent)
    executor.add_node("slide_create_agent", slide_create_agent)
    executor.add_node("post_processing_node", post_processing_node)

    executor.add_edge(START, "research_agent")
    executor.add_edge("research_agent", "slide_create_agent")
    executor.add_edge("slide_create_agent", "post_processing_node")
    executor.add_edge("post_processing_node", END)

    # main graph
    builder = StateGraph(SlideGenerateAgentState)
    builder.add_node("preprocessing_node", preprocessing_node)
    builder.add_node("planning_agent", planning_agent)
    builder.add_node("executor", executor.compile(name="executor"))

    builder.add_edge(START, "preprocessing_node")
    builder.add_edge("preprocessing_node", "planning_agent")

    def generate_slide(state):
        # design_prompt가 존재할 때만 추가
        if state.get("design_prompt"):
            return[
                Send(
                    "executor",
                    s | {"design_prompt": state["design_prompt"]}
                ) for s in state["sections"]]
        else:
            return[
                Send(
                    "executor",
                    s
                ) for s in state["sections"]]

    builder.add_conditional_edges(
        "planning_agent",
        generate_slide,
        ["executor"]
    )
    builder.add_edge("executor", END)

    return builder.compile(name="slide_generate_agent")

def create_graph():
    # 히든 명령어 파싱 그래프 생성
    
    # 슬라이드 생성 그래프 생성
    slide_generate_graph = create_slide_generate_graph()
    
    # Supervisor 생성
    workflow = create_supervisor(
        [slide_generate_graph],
        model=create_chat_model(provider="azure_openai", model="gpt-4.1"),
        prompt= """
                사용자와 대화를 통해 슬라이드 생성에 필요한 정보들을 수집하세요.
                                
                충분한 정보가 모이면 slide_generate_agent를 이용하여, 슬라이드를 생성하세요.
                마지막 메시지를 통해 다음 에이전트에 충분한 정보를 전달하세요.
                다음 에이전트는 마지막 메시지만을 참조합니다.
            """
        ,
        state_schema=SlideGenerateAgentState,
        output_mode="full_history",
    ).compile()

    # 메인 그래프 생성 - parse_hidden_command에서 workflow로 연결
    builder = StateGraph(SlideGenerateAgentState)
    builder.add_node("parse_hidden_command", parse_hidden_command_node)
    builder.add_node("workflow", workflow)
    
    # parse_hidden_command에서 workflow로 연결
    builder.add_edge(START, "parse_hidden_command")
    builder.add_edge("parse_hidden_command", "workflow")
    builder.add_edge("workflow", END)
    
    # Compile and run
    app = builder.compile()
    return app


if __name__ == '__main__':
    import time

    s = time.time()

    graph = create_graph()
    result = asyncio.run(graph.ainvoke({"topic": "제주도 여행 가이드", "num_sections": 5}))
    print(result)

    e = time.time()
    print(e - s)