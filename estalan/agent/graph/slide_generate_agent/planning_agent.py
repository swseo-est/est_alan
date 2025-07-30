import os
from typing import TypedDict, List

from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent
from langgraph.graph import START, END, StateGraph

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from estalan.agent.graph.slide_generate_agent.prompt.planning_agent import preliminary_investigation_instructions
from estalan.tools.search import GoogleSerperSearchResult
from estalan.llm import create_chat_model

class Section(TypedDict):
    topic: str

    idx: int
    name: str
    description: str
    research: bool

    content: str
    img_url: str

    html: str

class PlanningStateInput(TypedDict):
    topic: str # Report topic
    num_sections: int

class PlanningAgentState(AgentState):
    topic: str
    num_sections: int

    sections: List[str]

class GenerateSectionsOutput(TypedDict):
    sections: List[Section]

def create_init_planning_agent_node():
    def init_planning_agent_node(state: PlanningAgentState):
        init_msg = AIMessage(content="검색 도구를 사용하여 목차를 생성하기 위한 조사를 시작합니다.")
        return {"messages": [init_msg]}
    return init_planning_agent_node

def create_generate_sections_node(llm):
    def generate_section_result_msg(sections):
        msg = "생성된 목차는 다음과 같습니다. \n\n"
        for section in sections:
            msg_section = f"""{section["idx"]}. {section["topic"]} - {section["description"]} \n"""
            msg += msg_section

        msg = AIMessage(content=msg)
        return msg


    async def generate_sections_node(state: PlanningAgentState):
        topic = state["topic"]
        num_sections = state["num_sections"]

        # Format system instructions
        system_instructions_query = preliminary_investigation_instructions.format(
            topic=topic,
            number_of_queries=num_sections,
        )

        # Generate queries
        results = await llm.ainvoke({
            "messages":
            [
                SystemMessage(
                    content=system_instructions_query),
                HumanMessage(
                    content="""search queries that will help with planning the sections of the report. 
                    search_tool을 이용하세요.
                    Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: topic, idx, name, description, research, content, img and html fields.
                        research : False
                        content: ""
                        img: ""
                        html: ""
                    """)
            ]
        }
        )

        msg_result = generate_section_result_msg(results['structured_response']['sections'])
        updated_state = {"messages": [msg_result]}
        updated_state = updated_state | results['structured_response']
        return updated_state

    return generate_sections_node

def create_planning_agent(name=None):
    init_planning_agent_node = create_init_planning_agent_node()


    serper_api_key = os.getenv("SERPER_API_KEY")

    search_tool = GoogleSerperSearchResult.from_api_key(
        api_key=serper_api_key,
        k=15,
    )

    generate_sections_node_llm = create_chat_model(provider="azure_openai", model="gpt-4.1")

    generate_sections_node_agent = create_react_agent(
        generate_sections_node_llm,
        tools =[search_tool],
        response_format = GenerateSectionsOutput,
    )
    generate_sections_node = create_generate_sections_node(
        generate_sections_node_agent
    )


    builder = StateGraph(PlanningAgentState)
    builder.add_node("init_planning_agent_node", init_planning_agent_node)
    builder.add_node("generate_sections_node", generate_sections_node)

    builder.add_edge(START, "init_planning_agent_node")
    builder.add_edge("init_planning_agent_node", "generate_sections_node")
    builder.add_edge("generate_sections_node", END)

    planning_agent = builder.compile(name=name)
    return planning_agent

if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    import asyncio

    load_dotenv()

    agent = create_planning_agent()
    result = asyncio.run(agent.ainvoke({"topic": "이스트소프트", "num_sections": 5}))
    print(result['sections'])