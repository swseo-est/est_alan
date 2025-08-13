from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, TypedDict, Literal, Annotated
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState
from estalan.agent.graph.slide_generate_agent.prompt.slide_design import *
from estalan.llm import create_chat_model
from estalan.messages.utils import create_ai_message
from estalan.agent.graph.slide_generate_agent.state import ExecutorState

class SlideDesignAgentState(ExecutorState):
    pass

class SlideDesignNodeOutput(TypedDict):
    design: str

class HtmlGenerateNodeOutput(TypedDict):
    html: str
    width: int
    height: int

def pre_processing_node(state):
    return {}

def post_processing_node(state):
    return {}


def pre_processing_slide_design_node(state):
    name = state["name"]

    content = f"""
    {name} 페이지 디자인을 시작합니다.
    """

    msg = create_ai_message(
        content=content,
        name="msg_slide_design_start"
    )
    print(msg)

    return {"messages": [msg], "name": state["name"]}

def post_processing_slide_design_node(state):
    name = state["name"]

    content = f"""
    {name} 페이지 디자인을 완료하였습니다.
    """

    msg = create_ai_message(
        content=content,
        name="msg_slide_design_end"
    )
    print(msg)

    return {"messages": [msg], "name": state["name"]}


def pre_processing_html_generate_node(state):
    name = state["name"]

    content = f"""
    {name} 페이지 슬라이드를 생성하고 있습니다.
    """

    msg = create_ai_message(
        content=content,
        name="msg_html_generate_start"
    )
    print(msg)

    return {"messages": [msg], "name": state["name"]}

def post_processing_html_generate_node(state):
    name = state["name"]

    content = f"""
    {name} 페이지 라이드를 생성하였습니다.
    """

    msg = create_ai_message(
        content=content,
        name="msg_html_generate_end"
    )
    print(msg)

    return {"messages": [msg], "name": state["name"]}


def create_slide_design_node(slide_design_llm):
    def slide_design_node(state: SlideDesignAgentState):
        topic = state["topic"]
        name = state["name"]
        description = state["description"]
        content = state["content"]
        img_url = state["img_url"]

        content_info = section_inputs.format(
            topic=topic,
            section_name=name,
            section_topic=description,
            content=content,
            img_url=img_url,
        )
        system_instruction = SystemMessage(content=prompt_slide_design)
        response = slide_design_llm.invoke(
            [HumanMessage(content=content_info), system_instruction]
        )
        # response를 dict 형태로 반환하여 state에 병합되도록 함
        return {"design": response["design"]}
    return slide_design_node

def create_html_generate_node(html_generate_llm):
    def html_generate_node(state: SlideDesignAgentState):
        print("Current state:", state)
        system_instruction = SystemMessage(content=prompt_html_generator + prompt_slide_style)
        
        # design이 없으면 기본값 사용
        design_content = state.get("design", "기본 디자인을 적용합니다.")
        print("Design content:", design_content)
        
        # design_prompt가 존재하면 이를 활용
        if state.get("design_prompt"):
            print(f"Design prompt found: {state['design_prompt']}")
            # design_prompt를 포함한 강화된 디자인 지시사항 생성
            enhanced_design = f"{design_content}\n\n추가 디자인 요구사항: {state['design_prompt']}"
            design_content = enhanced_design
            print("Enhanced design content:", design_content)

        try:
            response = html_generate_llm.invoke([
                HumanMessage(content=state["content"]), 
                HumanMessage(content=design_content), 
                system_instruction
            ])
            return response
        except Exception as e:
            print(f"Error in html_generate_node: {e}")
            # 에러 발생 시 기본 HTML 반환
            return {
                "html": f"<div>Error generating HTML: {str(e)}</div>",
                "width": 1280,
                "height": 720
            }
    return html_generate_node

def create_slide_create_agent(name=None):
    slide_design_llm =create_chat_model(provider="azure_openai", model="gpt-4.1").with_structured_output(SlideDesignNodeOutput)
    html_generate_llm = create_chat_model(provider="azure_openai", model="gpt-4.1").with_structured_output(HtmlGenerateNodeOutput)

    slide_design_node = create_slide_design_node(slide_design_llm)
    html_generate_node = create_html_generate_node(html_generate_llm)

    builder = StateGraph(SlideDesignAgentState)
    builder.add_node("pre_processing_node", pre_processing_node)
    builder.add_node("post_processing_node", post_processing_node)
    builder.add_node("pre_processing_slide_design_node", pre_processing_slide_design_node)
    builder.add_node("post_processing_slide_design_node", post_processing_slide_design_node)
    builder.add_node("pre_processing_html_generate_node", pre_processing_html_generate_node)
    builder.add_node("post_processing_html_generate_node", post_processing_html_generate_node)
    builder.add_node("slide_design_node", slide_design_node)
    builder.add_node("html_generate_node", html_generate_node)

    builder.add_edge(START, "pre_processing_node")
    builder.add_edge("pre_processing_node", "pre_processing_slide_design_node")
    builder.add_edge("pre_processing_slide_design_node", "slide_design_node")
    builder.add_edge("slide_design_node", "pre_processing_html_generate_node")
    builder.add_edge("pre_processing_html_generate_node", "html_generate_node")
    builder.add_edge("html_generate_node", "post_processing_html_generate_node")
    builder.add_edge("post_processing_html_generate_node", "post_processing_node")
    builder.add_edge("post_processing_node", END)


    slide_crate_agent = builder.compile(name=name)
    return slide_crate_agent


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    slide_create_agent = create_slide_create_agent()

    response = slide_create_agent.invoke(
        {
            'topic': '이스트소프트',
             'name': '기업 개요 및 연혁',
             'description': '이스트소프트의 설립 배경, 주요 연혁, 대표자 및 기업의 주요 역사적 변화를 소개하는 섹션.',
             'content': '## 기업 개요 및 연혁\n\n이스트소프트는 1993년 10월 2일 설립된 대한민국의 종합 소프트웨어 개발 및 서비스 제공 기업이다. 한양대학교 경영대학원 전략벤처경영 과정 중이던 김장중 창업자를 중심으로, 대학 재학 중 친구들과 공동 창업하여 시작했다. 설립 초기에는 워드프로세서 프로그램 등 소프트웨어 개발과 유통에 집중하였으나, 경쟁과 시장 환경에 어려움을 겪으며 방향을 모색하게 된다[1][2].\n\n이스트소프트의 본격적 성장 계기는 1999년 자체 개발한 압축 프로그램 ‘알집’의 히트였다. 이후 ‘알씨’(이미지 뷰어), ‘알송’(음악 플레이어), ‘알약’(백신) 등 ‘알툴즈’ 시리즈를 연이어 선보이며 국민 유틸리티 소프트웨어 기업으로 자리매김했다. 2005년에는 게임 사업(대표작: 카발 온라인)에도 진출하며 사업 다각화에 나섰고, 2008년에는 코스닥 시장에 상장하였다. 2011년에는 자회사 줌인터넷을 통해 인터넷 포털 서비스 ‘줌닷컴(zum.com)’을 런칭했다[1][2][3].\n\n2016년 정상원 대표이사가 취임하면서 이스트소프트는 인공지능(AI) 분야를 미래 성장동력으로 삼아 사업 비전을 전환하였다. 이후 AI 소프트웨어와 플랫폼, 게임, 보안, 포털, 커머스, 자산운용 등 다양한 분야로 사업을 확대하고 있다. 2020년대에는 AI 신기술 기반 기업부설연구소도 설립하며 혁신 기업으로 도약하고 있다. 이스트소프트는 ‘더 쉽고 편리하며 안전한 IT환경’이라는 핵심 가치를 기반으로 국민 생활과 산업 현장에 기여하고 있다. 현재 대표이사는 정상원이 맡고 있으며, 본사는 서울특별시 서초구 반포대로에 위치한다[2][3][4].\n\n### Sources\n[1] 이스트소프트 - 나무위키: https://namu.wiki/w/%EC%9D%B4%EC%8A%A4%ED%8A%B8%EC%86%8C%ED%94%84%ED%8A%B8\n[2] 이스트소프트 - 위키백과: https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%8A%A4%ED%8A%B8%EC%86%8C%ED%94%84%ED%8A%B8\n[3] [PDF] 이스트소프트(047560) - 한국IR협의회: https://w4.kirs.or.kr/download/research/20250428_IT_%EC%9D%B4%EC%8A%A4%ED%8A%B8%EC%86%8C%ED%94%84%ED%8A%B8(047560)_AI%20%ED%9C%94%EB%A8%BC%20%EA%B8%B0%EC%88%A0%EC%9D%84%20%EA%B8%B0%EB%B0%98%EC%9C%BC%EB%A1%9C%20%ED%95%9C%20AI%20SW%20%EC%A0%84%EB%AC%B8%EA%B8%B0%EC%97%85_NICE%ED%8F%89%EA%B0%80%EC%A0%95%EB%B3%B4_%EC%88%98%EC%A0%95.pdf\n[4] 회사소개 - 이스트소프트: https://estsoft.ai/introduce',
             'img_url': 'https://upload.wikimedia.org/wikipedia/commons/4/4c/EST_NEW_CI_BLUE%28240%29.png'
         }
    )
    print(response['html'])

    # 생성된 HTML을 test.html로 저장
    with open("test.html", "w", encoding="utf-8") as f:
        f.write(response['html'])

