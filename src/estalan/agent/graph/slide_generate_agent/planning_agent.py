import os

from typing import TypedDict, List
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent
from langgraph.graph import START, END, StateGraph

from langchain_core.messages import SystemMessage

from estalan.agent.graph.slide_generate_agent.prompt.planning_agent import preliminary_investigation_instructions
from estalan.tools.search import GoogleSerperSearchResult
from estalan.llm import create_chat_model
from estalan.messages.utils import create_ai_message
from estalan.agent.graph.slide_generate_agent.state import SlideGenerateAgentMetadata, Section


class PlanningAgentState(AgentState):
    metadata: SlideGenerateAgentMetadata

    sections: List[Section]

class GenerateSectionsOutput(TypedDict):
    sections: List[Section]

def create_init_planning_agent_node():
    def init_planning_agent_node(state: PlanningAgentState):
        init_msg = create_ai_message(content="검색 도구를 사용하여 목차를 생성하기 위한 조사를 시작합니다.", name="planning_agent")
        return {"messages": [init_msg]}
    return init_planning_agent_node

def create_generate_sections_node(llm):
    async def generate_sections_node(state: PlanningAgentState):
        def generate_section_result_msg(sections):
            msg = "생성된 목차는 다음과 같습니다. \n\n"

            msg += "1. 타이틀 페이지\n"
            msg += "2. 목차 페이지\n"

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

        # Generate queries
        results = await llm.ainvoke({
            "messages":
            [
                SystemMessage(
                    content=system_instructions_query),
                SystemMessage(
                    content="""
                        목차를 한글로 작성하세요.
                    
                        search queries that will help with planning the sections of the report.  
                        Please use the search_tool.  
                        Generate the sections of the report. Your response must include a 'sections' field containing a list of sections.  
                        Each section must have: topic, idx, name, description, research, content, img and html fields.  
                        
                        slide_type: content
                        research : False  
                        
                        topic: str
                        idx: int
                        description: str
                        name: str
                        
                        content: ""  
                        img: ""  
                        html: ""  
                        
                        **Note:** Start the idx from 2.
                    """)
            ]
        }
        )

        print(results)
        msg_result = generate_section_result_msg(results['structured_response']['sections'])
        updated_state = {"messages": [msg_result]}
        updated_state = updated_state | results['structured_response']
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
            "name": "타이틀",
            "description": f"{title}에 대한 프레젠테이션 타이틀 페이지",
            "requirements": ["타이틀 텍스트", "부제목 또는 설명"],
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
        contents_text = "\n".join(section_list) if section_list else "목차 항목들이 정의되지 않았습니다."
        
        # 목차 슬라이드에 해당하는 section 정의
        contents_section: Section = {
            "slide_type": "toc",
            "topic": topic,
            "idx": 1,
            "name": "목차",
            "description": f"{topic}에 대한 프레젠테이션 목차 페이지\n\n{contents_text}",
            "requirements": ["목차 항목들"],
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

    generate_sections_node_llm = create_chat_model(provider="azure_openai", model="gpt-5-mini")

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
    builder.add_node("generate_sections_node", generate_sections_node)
    builder.add_node("add_tile_slide_node", add_tile_slide_node)
    builder.add_node("add_toc_slide_node", add_toc_slide_node)

    builder.add_edge(START, "init_planning_agent_node")
    builder.add_edge("init_planning_agent_node", "generate_sections_node")
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