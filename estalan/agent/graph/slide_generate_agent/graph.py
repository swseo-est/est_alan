from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import START, END, StateGraph
from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage

from typing import List, Annotated, TypedDict
from estalan.agent.graph.slide_generate_agent.planning_agent import create_planning_agent, Section
from estalan.agent.graph.slide_generate_agent.research_agent import create_research_agent
from estalan.agent.graph.slide_generate_agent.slide_design_agent import create_slide_create_agent
from estalan.llm.utils import create_chat_model
import asyncio
from langgraph.types import Send
import operator


class SlideGenerateAgentState(AgentState):
    topic: str
    requirements: str
    num_sections: int
    num_slides: int

    sections: List[Section]
    slides: Annotated[List[Section], operator.add]
    status: str

class OutputState(TypedDict):
    topic: str
    requirements: str

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

def create_graph():
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

    slide_create_agent = builder.compile(name="slide_create_agent")
    workflow = create_supervisor(
        [slide_create_agent],
        model=create_chat_model(provider="azure_openai", model="gpt-4.1"),
        prompt= """
                사용자와 대화를 통해 슬라이드 생성에 필요한 정보들을 수집하세요.
                충분한 정보가 모이면 slide_create_agent를 이용하여, 슬라이드를 생성하세요.
                마지막 메시지를 통해 다음 에이전트에 충분한 정보를 전달하세요.
                다음 에이전트는 마지막 메시지만을 참조합니다.
            """
        ,
        state_schema=SlideGenerateAgentState,
        output_mode="full_history"
    )

    # Compile and run
    app = workflow.compile()
    return app


if __name__ == '__main__':
    import time

    s = time.time()

    graph = create_graph()
    result = asyncio.run(graph.ainvoke({"topic": "제주도 여행 가이드", "num_sections": 5}))
    print(result)

    e = time.time()
    print(e - s)