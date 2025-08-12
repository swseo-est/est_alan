import os
import re
from typing import TypedDict, List

from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent
from langgraph.graph import START, END, StateGraph

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from estalan.agent.graph.slide_generate_agent.prompt.planning_agent import preliminary_investigation_instructions
from estalan.tools.search import GoogleSerperSearchResult
from estalan.llm import create_chat_model

# Planning Agent에 대한 상세한 설명 프롬프트
PLANNING_AGENT_SYSTEM_PROMPT = """당신은 슬라이드 생성 워크플로우의 핵심 기획자(Planning Agent)입니다.

## 주요 역할과 책임

### 1. 요구사항 분석 및 해석
- requirement_collection_agent로부터 수집된 요구사항을 분석
- 사용자 메시지와 요구사항을 종합하여 슬라이드 주제(topic) 추출
- 적절한 슬라이드 개수(num_sections) 결정 (1-20 범위)
- 요구사항이 부족한 경우 기본값 적용

### 2. 슬라이드 구조 설계
- 주제에 맞는 논리적이고 체계적인 슬라이드 구조 설계
- 웹 검색을 통한 주제 관련 정보 수집 및 분석
- 각 섹션별 내용 구성 및 우선순위 결정

### 3. 슬라이드 타입별 구성
- **타이틀 슬라이드 (idx: 0)**: 프레젠테이션 제목 및 부제목
- **목차 슬라이드 (idx: 1)**: 전체 슬라이드 구조 개요
- **내용 슬라이드 (idx: 2+)**: 주제별 상세 내용

### 4. 품질 관리
- 수집된 요구사항을 충족하는 논리적인 구조 검증
- 각 섹션의 명확성과 일관성 확보
- 사용자 의도에 부합하는 최종 결과물 생성

## 작업 프로세스
1. 요구사항 분석 → topic, num_sections 결정
2. 웹 검색을 통한 주제 조사
3. 섹션별 내용 구성 및 설계
4. 타이틀 및 목차 슬라이드 추가
5. 최종 슬라이드 구조 완성

## 출력 형식
각 섹션은 다음 필드를 포함해야 합니다:
- slide_type: 슬라이드 유형 (title, toc, content)
- topic: 주제
- idx: 순서 (0부터 시작)
- name: 섹션명
- description: 상세 설명
- requirements: 요구사항 목록
- research: 연구 필요 여부
- content: 내용
- img_url: 이미지 URL
- design: 디자인 지침
- html: HTML 코드
- width/height: 크기 정보

**중요**: 사용자의 요구사항을 최우선으로 고려하여 논리적이고 체계적인 슬라이드 구조를 설계하세요."""

class Section(TypedDict):
    slide_type: str # title, contents, etc

    topic: str

    idx: int
    name: str
    description: str
    requirements: List[str]
    research: bool

    content: str
    img_url: str

    design: str
    html: str
    width: int
    height: int

class PlanningStateInput(TypedDict):
    topic: str # Report topic
    num_sections: int

class PlanningAgentState(AgentState):
    topic: str
    num_sections: int
    requirement_collection_agent_state: dict  # requirement_collection_agent의 전체 state
    sections: List[str]

class GenerateSectionsOutput(TypedDict):
    sections: List[Section]


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
        
        # requirement_collection_agent_state에서 수집된 요구사항들을 추출
        requirements_text = ""
        requirement_state = state.get("requirement_collection_agent_state", {})
        private_state = requirement_state.get("requirement_collection_agent_private_state", {})
        requirements = private_state.get("requirements", [])
        
        if requirements:
            requirements_text = "\n수집된 요구사항들:\n"
            for i, req in enumerate(requirements, 1):
                requirements_text += f"{i}. [{req.get('category', '카테고리')}] - {req.get('detail', '상세내용')}\n"

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
                    content=f"""
                        {requirements_text}
                        
                        위 요구사항들을 바탕으로 search queries that will help with planning the sections of the report.  
                        Please use the search_tool.  
                        Generate the sections of the report. Your response must include a 'sections' field containing a list of sections.  
                        Each section must have: topic, idx, name, description, research, content, img and html fields.  
                        slide_type: content
                        research : False  
                        content: ""  
                        img: ""  
                        html: ""  
                        
                        **Note:** Start the idx from 2.
                        
                        **중요:** 수집된 요구사항을 충족하는 논리적인 구조로 목차를 생성하세요.
                    """)
            ]
        }
        )

        msg_result = generate_section_result_msg(results['structured_response']['sections'])
        updated_state = {"messages": [msg_result]}
        updated_state = updated_state | results['structured_response']
        return updated_state

    return generate_sections_node


def create_add_tile_slide_node():
    def add_title_slide(state: PlanningAgentState):
        title = state["topic"]
        
        # 타이틀 슬라이드에 해당하는 section 정의
        title_section: Section = {
            "slide_type": "title",
            "topic": title,
            "idx": 0,  # 타이틀 슬라이드는 첫 번째
            "name": "타이틀 슬라이드",
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
        topic = state["topic"]
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
            "name": "목차 슬라이드",
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
    """planning_agent 생성 - SlideGenerateAgentState를 받아서 처리"""
    
    def planning_wrapper(state):
        """SlideGenerateAgentState를 PlanningAgentState로 변환하는 wrapper"""
        # requirement_collection_agent의 결과에서 요구사항들을 추출
        requirement_state = state.get("requirement_collection_agent_state", {})
        private_state = requirement_state.get("requirement_collection_agent_private_state", {})
        
        # LLM을 사용하여 요구사항에서 topic과 num_sections 추출
        requirements = private_state.get("requirements", [])
        messages = state.get("messages", [])
        
        # 기본값 설정
        topic = "슬라이드 생성"
        num_sections = 5
        
        if requirements or messages:
            # LLM을 사용하여 요구사항 분석
            from pydantic import BaseModel, Field
            
            class RequirementsAnalysis(BaseModel):
                topic: str = Field(description="슬라이드의 주제 (구체적이고 명확하게)")
                num_sections: int = Field(description="슬라이드 개수 (1-20 범위)", ge=1, le=20)
                reasoning: str = Field(description="topic과 num_sections를 결정한 이유 (간단하게)")
            
            llm = create_chat_model(provider="azure_openai", model="gpt-4.1", structured_output=RequirementsAnalysis)
            
            # 요구사항 텍스트 구성
            requirements_text = ""
            if requirements:
                requirements_text = "\n수집된 요구사항들:\n"
                for i, req in enumerate(requirements, 1):
                    category = req.get('category', '카테고리')
                    detail = req.get('detail', '상세내용')
                    requirements_text += f"{i}. [{category}] - {detail}\n"
            
            # 사용자 메시지 텍스트 구성
            user_message_text = ""
            if messages:
                for msg in messages:
                    if hasattr(msg, 'content') and isinstance(msg.content, str):
                        user_message_text += f"사용자 메시지: {msg.content}\n"
            
            # LLM에게 분석 요청
            analysis_prompt = f"""
{PLANNING_AGENT_SYSTEM_PROMPT}

## 현재 작업: 요구사항 분석
당신은 슬라이드 생성 요구사항을 분석하는 전문가입니다.
다음 정보를 바탕으로 슬라이드의 주제(topic)와 적절한 슬라이드 개수(num_sections)를 추출해주세요.

{user_message_text}
{requirements_text}

**중요 규칙:**
1. topic은 구체적이고 명확해야 합니다 (예: "제주도 여행" vs "여행")
2. num_sections는 1-20 범위 내에서 설정하세요
3. 사용자가 명시적으로 슬라이드 수를 요청했다면 그 수를 우선적으로 반영하세요
4. 요구사항이 부족하면 기본값을 사용하세요 (topic: "슬라이드 생성", num_sections: 5)
"""

            try:
                response = llm.invoke([{"role": "user", "content": analysis_prompt}])
                
                # structured_output을 사용하므로 직접 필드 접근 가능
                if hasattr(response, 'topic') and response.topic:
                    topic = response.topic
                
                if hasattr(response, 'num_sections'):
                    extracted_num = response.num_sections
                    if isinstance(extracted_num, int) and 1 <= extracted_num <= 20:
                        num_sections = extracted_num
                
                if hasattr(response, 'reasoning'):
                    print(f"LLM 분석 결과 - topic: {topic}, num_sections: {num_sections}")
                    print(f"이유: {response.reasoning}")
                else:
                    print(f"LLM 분석 결과 - topic: {topic}, num_sections: {num_sections}")
                    
            except Exception as e:
                print(f"LLM 분석 중 오류 발생: {e}")
                # 기본값 사용
        
        # requirement_collection_agent_state를 직접 참조하도록 수정
        planning_state = {
            "topic": topic,
            "num_sections": num_sections,
            "requirement_collection_agent_state": requirement_state,  # 전체 state 참조
            "sections": []
        }
        
        return planning_state
    
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

    add_tile_slide_node = create_add_tile_slide_node()
    add_toc_slide_node = create_add_toc_slide_node()

    # 초기화 노드에 planning_wrapper 로직 통합
    def enhanced_init_node(state):
        """초기화 노드 - planning_wrapper 로직을 포함하여 topic과 num_sections 설정"""
        # planning_wrapper를 사용하여 topic과 num_sections 추출
        planning_input = planning_wrapper(state)
        
        # 추출된 정보를 state에 설정
        init_msg = AIMessage(content=f"슬라이드 생성 계획 수립을 시작합니다. 주제: {planning_input['topic']}, 예상 슬라이드 수: {planning_input['num_sections']}개")
        init_msg2 = AIMessage(content="검색 도구를 사용하여 목차를 생성하기 위한 조사를 시작합니다.")

        return {
            "messages": [init_msg, init_msg2],
            "topic": planning_input["topic"],
            "num_sections": planning_input["num_sections"],
            "requirement_collection_agent_state": planning_input["requirement_collection_agent_state"]
        }
    
    # enhanced_init_node를 사용하여 그래프 구성
    builder = StateGraph(PlanningAgentState)
    builder.add_node("init_planning_agent_node", enhanced_init_node)
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
    import os
    from dotenv import load_dotenv
    import asyncio

    load_dotenv()

    agent = create_planning_agent()
    result = asyncio.run(agent.ainvoke({"topic": "이스트소프트", "num_sections": 5}))
    print(result['sections'])