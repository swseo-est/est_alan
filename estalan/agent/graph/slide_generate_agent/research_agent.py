import os
from typing import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.prebuilt import create_react_agent
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState

from estalan.agent.graph.slide_generate_agent.prompt.research_agent import *
from estalan.llm import create_chat_model
from estalan.tools.search import GoogleSerperSearchResult, GoogleSerperImageSearchResult, is_cors_violation
from estalan.agent.graph.slide_generate_agent.planning_agent import Section


class ResearchAgentState(Section):
    pass

class ResearchStateInput(TypedDict):
    topic: str # Report topic
    name: str
    description: str

class ResearchAgentOutput(TypedDict):
    name: str
    description: str

    # Research Node
    research: bool
    content: str

    # Search Img Node
    img_url: str
    is_cors_violation: bool

class ResearchNodeOutput(TypedDict):
    research: bool
    content: str

class SearchImgNodeOutput(TypedDict):
    img_url: str
    is_cors_violation: bool


def create_research_node(llm):
    async def research_node(state: ResearchAgentState):
        print(state)
        topic = state["topic"]
        name = state["name"]
        description = state["description"]

        section_writer_inputs_formatted = section_writer_inputs.format(topic=topic,
                                                                       section_name=name,
                                                                       section_topic=description,
                                                                       content="")

        # Format system instructions
        results = await llm.ainvoke(
            {
                "messages":
                    [
                        SystemMessage(content=section_writer_instructions),
                        HumanMessage(content=section_writer_inputs_formatted),
                        HumanMessage(content="search_tool을 사용하세요. 모든 content는 검색된 내용을 기반으로만 작성하세요."),
                    ]
            }
        )

        print(results['structured_response'])
        return results['structured_response']
    return research_node

def create_search_img_node(llm):
    async def search_img_node(state: ResearchAgentState):
        topic = state["topic"]
        name = state["name"]
        description = state["description"]
        content = state["content"]

        section_writer_inputs_formatted = section_writer_inputs.format(topic=topic,
                                                                       section_name=name,
                                                                       section_topic=description,
                                                                       content=content
                                                                       )

        # Format system instructions
        results = await llm.ainvoke(
            {
                "messages":
                    [
                        SystemMessage(content=section_search_img_instruction),
                        HumanMessage(content=section_writer_inputs_formatted),
                        HumanMessage(content="search_tool을 사용하세요. CORS 정책에 위배되지 않은 이미지만 수집하세요."),
                    ]
            }
        )

        print(results['structured_response'])
        return results['structured_response']
    return search_img_node


def create_research_agent(name=None):
    serper_api_key = os.getenv("SERPER_API_KEY")

    search_tool = GoogleSerperSearchResult.from_api_key(
        api_key=serper_api_key,
        k=15,
    )

    search_img_tool = GoogleSerperImageSearchResult.from_api_key(
        api_key=serper_api_key,
        k=5,
    )

    research_node_llm = create_chat_model(provider="azure_openai", model="gpt-4.1")

    research_node_agent = create_react_agent(
        research_node_llm,
        tools =[search_tool],
        response_format = ResearchNodeOutput,
    )
    research_node = create_research_node(
        research_node_agent
    )

    search_img_llm = create_chat_model(provider="azure_openai", model="gpt-4.1")

    search_img_agent = create_react_agent(
        search_img_llm,
        tools =[search_img_tool, is_cors_violation],
        response_format = SearchImgNodeOutput,
    )
    search_img_node = create_search_img_node(
        search_img_agent
    )

    builder = StateGraph(ResearchAgentState, output=ResearchAgentOutput)
    builder.add_node("research_node", research_node)
    builder.add_node("search_img_node", search_img_node)

    builder.add_edge(START, "research_node")
    builder.add_edge("research_node", "search_img_node")
    builder.add_edge("search_img_node", END)

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