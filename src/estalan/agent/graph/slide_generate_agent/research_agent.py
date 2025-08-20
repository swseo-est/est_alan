import os
from typing import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.prebuilt import create_react_agent
from langgraph.graph import START, END, StateGraph

from estalan.agent.graph.slide_generate_agent.prompt.research_agent import *
from estalan.llm import create_chat_model
from estalan.tools.search import GoogleSerperSearchResult
from estalan.messages.utils import create_ai_message
from estalan.agent.graph.slide_generate_agent.state import ExecutorState


class ResearchAgentState(ExecutorState):
    pass

class ResearchNodeOutput(TypedDict):
    research: bool
    content: str


def pre_processing_node(state):
    return {}

def post_processing_node(state):
    return {}

def pre_processing_research_node(state):
    content = f"""슬라이드 생성에 필요한 조사를 시작합니다.
"""

    msg = create_ai_message(
        content=content,
        name="msg_research_start",
        id="msg_research_start"
    )

    return {"messages": [msg]}

def post_processing_research_node(state):
    name = state["name"]

    content = f"""{name} 페이지에 대한 조사를 완료하였습니다."""

    msg = create_ai_message(
        content=content,
        name="msg_research_end",
        id = "msg_research_end"
    )
    return {}

def create_research_node(llm):
    async def research_node(state: ResearchAgentState):
        topic = state["topic"]
        name = state["name"]
        description = state["description"]

        section_writer_inputs_formatted = section_writer_inputs.format(topic=topic,
                                                                       section_name=name,
                                                                       section_topic=description,
                                                                       content="")
        # Format system instructions
        for i in range(10):
            try:
                results = await llm.ainvoke(
                    {
                        "messages":
                            [
                                SystemMessage(content=section_writer_instructions),
                                HumanMessage(content=section_writer_inputs_formatted),
                            ]
                    }
                )
                return results['structured_response']
            except Exception as e:
                print(i,e )

    return research_node


def create_research_agent(name=None):
    serper_api_key = os.getenv("SERPER_API_KEY")

    search_tool = GoogleSerperSearchResult.from_api_key(
        api_key=serper_api_key,
        k=15,
    )

    research_node_llm = create_chat_model(provider="azure_openai", model="gpt-5-mini")

    research_node_agent = create_react_agent(
        research_node_llm,
        tools =[search_tool],
        response_format = ResearchNodeOutput,
    )
    research_node = create_research_node(
        research_node_agent
    )

    builder = StateGraph(ResearchAgentState)
    builder.add_node("pre_processing_node", pre_processing_node)
    builder.add_node("post_processing_node", post_processing_node)

    builder.add_node("pre_processing_research_node", pre_processing_research_node)
    builder.add_node("research_node", research_node)
    builder.add_node("post_processing_research_node", post_processing_research_node)

    builder.add_edge(START, "pre_processing_node")
    builder.add_edge("pre_processing_node", "pre_processing_research_node")
    builder.add_edge("pre_processing_research_node", "research_node")
    builder.add_edge("research_node", "post_processing_research_node")
    builder.add_edge("post_processing_research_node", "post_processing_node")
    builder.add_edge("post_processing_node", END)

    research_agent = builder.compile(name=name)
    return research_agent


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    import asyncio

    load_dotenv()

    agent = create_research_agent()
    result = asyncio.run(agent.ainvoke(
        {
            'topic': "이스트소프트",
            'name': '기업 개요 및 연혁',
            'description': '이스트소프트의 설립 배경, 주요 연혁, 대표자 및 기업의 주요 역사적 변화를 소개하는 섹션.'
        }
    ))
    print(result)