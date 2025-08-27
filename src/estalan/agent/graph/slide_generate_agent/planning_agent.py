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
from estalan.agent.graph.slide_generate_agent.state import SlideGenerateAgentMetadata, Section, SlideGenerateAgentState


def generate_section_result_msg(sections):
    msg = "슬라이드 구성은 다음과 같습니다. \n\n"

    msg += "1. 타이틀 페이지\n"
    msg += "2. 목차\n"

    for section in sections:
        msg_section = f"""{int(section["idx"]) + 1}. {section["name"]} - {section["description"]} \n"""
        msg += msg_section

    msg = create_ai_message(content=msg, name=None)
    return msg


class GenerateSectionsOutput(TypedDict):
    sections: List[Section]
    num_sections: int

class AnalyzeRequirementsOutput(TypedDict):
    topic: str
    num_sections: int

def create_init_planning_agent_node():
    def init_planning_agent_node(state: SlideGenerateAgentState):
        init_msg = create_ai_message(content="검색 도구를 사용하여 목차를 생성하기 위한 조사를 시작합니다.", name="planning_agent")

        return {"messages": [init_msg]}
    return init_planning_agent_node

def print_tool_usage_msg(state: SlideGenerateAgentState):
    tool_usage_msg = create_block_message(content="웹 검색 도구를 사용 중 입니다...", block_tag="web_search_tool", name="planning_tool_usage")
    return {"messages": [tool_usage_msg]}


def create_analyze_requirements_node(llm):
    async def analyze_requirements_node(state: SlideGenerateAgentState):
        def generate_analysis_result_msg(topic, num_sections):
            msg = f"요구사항 분석 결과:\n"
            msg += f"- 주제: {topic}\n"
            msg += f"- 추천 섹션 개수: {num_sections}개\n"
            msg = create_ai_message(content=msg, name=None, metadata={"log_level": "debug"})
            return msg

        # requirements_docs에서 요구사항 정보 추출
        requirements_docs = state.get("requirements_docs", "")
        
        if not requirements_docs:
            # ToDo 이 구현 해야함
            # 요구사항이 없는 경우 기본값 사용
            requirements_docs = f""


            # 요구사항 분석을 위한 시스템 프롬프트
        system_prompt = f"""사용자의 요구사항을 분석하여 프레젠테이션 주제와 적절한 섹션 개수를 추출하세요.

요구사항:
{requirements_docs}

다음 형식으로 응답하세요:
- topic: 프레젠테이션의 주요 주제 (간결하고 명확하게)
- num_sections: 섹션 개수 (요구사항에 명시된 섹션 개수가 있다면 그것을 사용하고, 없다면 3-8개 사이의 적절한 숫자로 설정)

주의사항:
1. 요구사항에 "섹션 X개", "슬라이드 X개", "페이지 X개" 등이 명시되어 있다면 그 숫자를 그대로 사용
2. 요구사항에 섹션 개수가 명시되지 않았다면 3-8개 사이의 적절한 숫자로 설정
3. 요구사항의 복잡도와 내용을 고려하여 적절한 개수 결정

예시:
- topic: "AI 기술의 현재와 미래"
- num_sections: 5
"""
        for i in range(10):
            try:
                results = await llm.ainvoke(
                    [
                        SystemMessage(content=system_prompt.format(requirements=requirements_docs)),
                        HumanMessage(content="요구사항을 분석해주세요.")
                    ]
                )

                topic = results["topic"]
                num_sections = int(results["num_sections"])
                break

            except Exception as e:
                print(f"요구사항 분석 중 오류 발생: {e}")

        msg_result = generate_analysis_result_msg(topic, num_sections)
        print(msg_result)
        
        # metadata 업데이트
        metadata = state["metadata"].copy()
        metadata["topic"] = topic
        metadata["num_sections"] = num_sections
        metadata["num_slides"] = num_sections + 2
        
        return {
            "messages": [msg_result], 
            "metadata": metadata
        }
    
    return analyze_requirements_node


def create_generate_sections_node(llm):
    async def generate_sections_node(state: SlideGenerateAgentState):
        topic = state["metadata"]["topic"]
        num_sections = state["metadata"]["num_sections"]
        
        # requirements_docs만을 사용하여 요구사항 정보 추출
        requirements_docs = state.get("requirements_docs", "")

        # Format system instructions
        system_instructions_query = preliminary_investigation_instructions.format(
            topic=topic,
            number_of_queries=num_sections,  # 추출된 섹션 개수 사용
            requirements=requirements_docs,
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
    def add_title_slide(state: SlideGenerateAgentState):
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
    def add_toc_slide(state: SlideGenerateAgentState):
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
            "requirements": [
                "content slide_type에 대해서 목차를 나열시 idx 번호를 제거하고 출력하세요"
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

    # 요구사항 분석을 위한 LLM
    analyze_requirements_llm = create_chat_model(provider="google_vertexai", model="gemini-2.5-flash", lazy=True).with_structured_output(AnalyzeRequirementsOutput)
    analyze_requirements_node = create_analyze_requirements_node(analyze_requirements_llm)

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

    builder = StateGraph(SlideGenerateAgentState)
    builder.add_node("init_planning_agent_node", init_planning_agent_node)
    builder.add_node("analyze_requirements_node", analyze_requirements_node)
    builder.add_node("print_tool_usage_msg", print_tool_usage_msg)
    builder.add_node("generate_sections_node", generate_sections_node)
    builder.add_node("add_tile_slide_node", add_tile_slide_node)
    builder.add_node("add_toc_slide_node", add_toc_slide_node)

    builder.add_edge(START, "init_planning_agent_node")
    builder.add_edge("init_planning_agent_node", "analyze_requirements_node")
    builder.add_edge("analyze_requirements_node", "print_tool_usage_msg")
    builder.add_edge("print_tool_usage_msg", "generate_sections_node")
    builder.add_edge("generate_sections_node", "add_toc_slide_node")
    builder.add_edge("add_toc_slide_node", "add_tile_slide_node")

    builder.add_edge("add_tile_slide_node", END)

    planning_agent = builder.compile(name=name)
    return planning_agent

if __name__ == '__main__':
    from dotenv import load_dotenv
    import asyncio
    from estalan.agent.graph.slide_generate_agent.tests.inputs import requirements_docs

    load_dotenv()

    agent = create_planning_agent()
    result = asyncio.run(
        agent.ainvoke
            (
        {"requirements_docs": requirements_docs}
        )
    )
    print(result)
    print(generate_section_result_msg(result["sections"]))