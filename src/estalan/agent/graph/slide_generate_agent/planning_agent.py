import os

from typing import TypedDict, List
from estalan.agent.base.state import BaseAlanAgentState
from langgraph.prebuilt import create_react_agent
from langgraph.graph import START, END, StateGraph

from langchain_core.messages import SystemMessage, HumanMessage

from estalan.agent.graph.slide_generate_agent.prompt.planning_agent import preliminary_investigation_instructions
from estalan.tools.search import GoogleSerperSearchResult
from estalan.llm import create_chat_model
from estalan.messages.utils import create_ai_message, create_block_message
from estalan.agent.graph.slide_generate_agent.state import SlideGenerateAgentMetadata, Section


class PlanningAgentState(BaseAlanAgentState):
    metadata: SlideGenerateAgentMetadata
    sections: List[Section]

class GenerateSectionsOutput(TypedDict):
    sections: List[Section]

def create_init_planning_agent_node():
    def init_planning_agent_node(state: PlanningAgentState):
        init_msg = create_ai_message(content="검색 도구를 사용하여 목차를 생성하기 위한 조사를 시작합니다.", name="planning_agent")

        return {"messages": [init_msg]}
    return init_planning_agent_node

def print_tool_usage_msg(state: PlanningAgentState):
    tool_usage_msg = create_block_message(content="웹 검색 도구를 사용 중 입니다...", block_tag="web_search_tool", name="planning_tool_usage")
    return {"messages": [tool_usage_msg]}

def create_generate_sections_node(llm):
    async def generate_sections_node(state: PlanningAgentState):
        def generate_section_result_msg(sections):
            msg = "슬라이드 구성은 다음과 같습니다. \n\n"

            msg += "1. 타이틀 페이지\n"
            msg += "2. 목차\n"

            for section in sections:
                msg_section = f"""{int(section["idx"]) + 1}. {section["name"]} - {section["description"]} \n"""
                msg += msg_section

            msg = create_ai_message(content=msg, name="planning_agent")
            return msg

        topic = state["metadata"]["topic"]
        num_sections = state["metadata"]["num_sections"]

        # Format system instructions
        system_instructions_query = preliminary_investigation_instructions.format(
            topic=topic,
            number_of_queries=num_sections,
        )

        num_try = 10
        # Generate queries
        for i in range(num_try):
            try:
                results = await llm.ainvoke({
                    "messages":
                    [
                        SystemMessage(
                            content=system_instructions_query),
                        HumanMessage(
                            content="Let's do it."
                        )
                    ]
                }
                )

                msg_result = generate_section_result_msg(results['structured_response']['sections'])

                sections = results['structured_response']['sections']

                list_check_field = ["topic", "idx", "name", "description"]
                for s in sections:
                    for field in list_check_field:
                        if field not in s.keys():
                            print(f"there are no field name {field}")
                            raise Exception()

                break
            except Exception as e:
                print(e)



        sections_refined = list()
        for s in sections:
            s_refined = s.copy()
            s_refined["idx"] = int(s["idx"])

            sections_refined.append(s)

        metadata = state["metadata"].copy()
        metadata["num_sections"] = len(sections_refined)

        updated_state = {"messages": [msg_result], "metadata": metadata, "sections": sections_refined}
        return updated_state

    return generate_sections_node


def create_add_tile_slide_node():
    def add_title_slide(state: PlanningAgentState):
        title = state["metadata"]["topic"]
        
        # 타이틀 슬라이드에 해당하는 section 정의
        title_section: Section = {
            "slide_type": "title",
            "topic": title,
            "idx": 0,  # 타이틀 슬라이드는 첫 번째
            "name": "Title",
            "description": f"Presentation title page for {title}",
            "requirements": ["Title text", "Subtitle or description"],
            "research": False, 
            "content": "",
            "img_url": "", 
            "design": "",
            "html": "" 
        }
        
        # 기존 sections가 있다면 title_section을 맨 앞에 추가
        current_sections = state.get("sections", [])
        updated_sections = [title_section] + current_sections
        
        return {"sections": updated_sections}
        
    return add_title_slide


def create_add_toc_slide_node():
    def add_toc_slide(state: PlanningAgentState):
        topic = state["metadata"]["topic"]
        current_sections = state.get("sections", [])
        
        # 기존 섹션들에서 목차 정보 추출
        section_list = []
        for section in current_sections:
            if section.get("slide_type") != "title":  # 타이틀 슬라이드가 아닌 경우
                section_list.append(f"{section.get('idx', 0)}. {section.get('name', '')}")
        
        # 목차 문자열 생성
        contents_text = "\n".join(section_list) if section_list else "Table of contents items are not defined."
        
        # 목차 슬라이드에 해당하는 section 정의
        contents_section: Section = {
            "slide_type": "toc",
            "topic": topic,
            "idx": 1,
            "name": "Table of Contents",
            "description": f"Presentation table of contents page for {topic}\n\n{contents_text}",
            "requirements": [
                "When listing table of contents for content slide_type, remove idx numbers and output"
            ],
            "research": False,
            "content": "", 
            "img_url": "",
            "design": "", 
            "html": ""  
        }
        
        # 타이틀 슬라이드 다음에 목차 슬라이드 추가
        # 타이틀 슬라이드가 idx 0이므로, 목차는 idx 1에 위치
        updated_sections = [contents_section] + current_sections

        return {"sections": updated_sections}
        
    return add_toc_slide

def create_planning_agent(name="planning_agent"):
    init_planning_agent_node = create_init_planning_agent_node()

    serper_api_key = os.getenv("SERPER_API_KEY")

    search_tool = GoogleSerperSearchResult.from_api_key(
        api_key=serper_api_key,
        k=15,
    )

    generate_sections_node_llm = create_chat_model(provider="azure_openai", model="gpt-5-mini", lazy=True)

    generate_sections_node_agent = create_react_agent(
        generate_sections_node_llm,
        tools =[search_tool],
        response_format=GenerateSectionsOutput,
    )
    generate_sections_node = create_generate_sections_node(
        generate_sections_node_agent
    )

    add_tile_slide_node = create_add_tile_slide_node()
    add_toc_slide_node = create_add_toc_slide_node()

    builder = StateGraph(PlanningAgentState)
    builder.add_node("init_planning_agent_node", init_planning_agent_node)
    builder.add_node("print_tool_usage_msg", print_tool_usage_msg)
    builder.add_node("generate_sections_node", generate_sections_node)
    builder.add_node("add_tile_slide_node", add_tile_slide_node)
    builder.add_node("add_toc_slide_node", add_toc_slide_node)

    builder.add_edge(START, "init_planning_agent_node")
    builder.add_edge("init_planning_agent_node", "print_tool_usage_msg")
    builder.add_edge("print_tool_usage_msg", "generate_sections_node")
    builder.add_edge("generate_sections_node", "add_toc_slide_node")
    builder.add_edge("add_toc_slide_node", "add_tile_slide_node")

    builder.add_edge("add_tile_slide_node", END)

    planning_agent = builder.compile(name=name)
    return planning_agent

if __name__ == '__main__':
    from dotenv import load_dotenv
    import asyncio

    load_dotenv()

    agent = create_planning_agent()
    result = asyncio.run(agent.ainvoke({"topic": "이스트소프트", "num_sections": 5}))
    print(result['sections'])