from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import START, END, StateGraph

from typing import List, Annotated, TypedDict
from estalan.agent.graph.slide_generate_agent.planning_agent import create_planning_agent, Section
from estalan.agent.graph.slide_generate_agent.research_agent import create_research_agent
from estalan.agent.graph.slide_generate_agent.slide_design_agent import create_slide_create_agent
import asyncio
from langgraph.types import Send
import operator


class SlideGenerateAgentState(AgentState):
    topic: str
    num_sections: int

    sections: List[Section]
    slides: Annotated[List[Section], operator.add]
    status: str

class ExecutorOutput(TypedDict):
    slides: Section


def post_processing_node(state):
    print(state)
    # 생성된 HTML을 test.html로 저장
    with open(f"{state['idx']}.html", "w", encoding="utf-8") as f:
        f.write(state['html'])

    return {"slides": [state]}

def create_graph():
    planning_agent = create_planning_agent()
    research_agent = create_research_agent()
    slide_create_agent = create_slide_create_agent()

    ## subroutine
    executor = StateGraph(Section, output=ExecutorOutput)

    executor.add_node("research_agent", research_agent)
    executor.add_node("slide_create_agent", slide_create_agent)
    executor.add_node("post_processing_node", post_processing_node)

    executor.add_edge(START, "research_agent")
    executor.add_edge("research_agent", "slide_create_agent")
    executor.add_edge("slide_create_agent", "post_processing_node")
    executor.add_edge("post_processing_node", END)

    # main graph
    builder = StateGraph(SlideGenerateAgentState)
    builder.add_node("planning_agent", planning_agent)
    builder.add_node("executor", executor.compile(name="executor"))


    builder.add_edge(START, "planning_agent")
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

    planning_agent = builder.compile()
    return planning_agent


if __name__ == '__main__':
    import os
    import time

    s = time.time()

    graph = create_graph()
    result = asyncio.run(graph.ainvoke({"topic": "제주도 여행 가이드", "num_sections": 5}))
    print(result)

    e = time.time()
    print(e - s)