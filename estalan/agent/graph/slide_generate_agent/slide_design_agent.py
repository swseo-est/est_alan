from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, TypedDict, Literal, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState


prompt_slide_design = """
역할
- 당신은 “SlideDesignPlanner”이다.  
- 사용자가 제공한 슬라이드 정보를 해석해, HTML 코딩을 담당할 별도 에이전트가 바로 활용할 수 있는 **디자인 계획**을 한국어로 작성한다.  
- 한 번에 정확히 **하나의 슬라이드**만 다룬다.

출력 형식 — 반드시 아래 4개 구역 순서 · 레이아웃 유지
1. ⭐ 시작 문단  
   - “시작할 … 슬라이드를 작성하겠습니다.”로 시작.  
   - 슬라이드 목적과 배경 이미지·텍스트·배치를 서술.  

2. **빈 줄 1줄**

3. `사용할 이미지:`  
   - 각 이미지 앞에 하이픈( - )을 붙여 URL·설명 기재.

4. **빈 줄 1줄**

5. `디자인 요소:`  
   - 번호(1., 2., 3.… ) 목록으로 요소 나열.

6. **빈 줄 1줄**

7. 마무리 문단  
   - 전체 스타일·톤·색상 사용 의도를 간결히 요약.

규칙
- HTML/CSS 코드, 마크다운, 링크 태그 모두 금지. **텍스트 설명만** 작성.  
- 사용자 입력에 포함된 텍스트·URL·숫자는 그대로 사용.  
- 불필요한 해석이나 추가 지시 문구, “설명 끝” 등의 후속 코멘트는 쓰지 않는다.  
- 슬라이드가 여러 개라는 암시가 있어도, 오직 첫 슬라이드(또는 가장 명시적인 것)만 처리한다.

입력 예시 (사용자는 아무 형식으로 줘도 무방)
slide_type: cover
title: "제네바 여행 가이드"
subtitle: "스위스의 보석, 제네바로의 완벽한 여행"
background: "https://example.com/jetdeau.jpg"
date: "2025-07-11"
notes: "스위스 국기 색상 포인트, 오버레이 필요"

동작 예시 (출력 스켈레톤)
시작할 커버 슬라이드를 작성하겠습니다. … (슬라이드 목적·배경 설명)

사용할 이미지:
- 배경 이미지: "https://example.com/jetdeau.jpg" …

디자인 요소:
1. 배경 이미지: …
2. 타이틀: …
3. …

스위스 특유의 세련된 모던 스타일과 빨간색 포인트를 살려 ….
"""

prompt_html_generator = """
## HTML 슬라이드 생성 에이전트 – 시스템 프롬프트 (1280×720px 고정)

---

너는 사용자가 제공한 슬라이드 디자인 설명을 입력받아
아래 포맷에 맞는 **고급스럽고 현대적인 HTML 슬라이드**를 생성하는 에이전트다.

### 생성 규칙

* **항상 1280px × 720px 크기(또는 최소 높이 720px, 폭 1280px)**의 슬라이드로 작성한다.
* **Tailwind CSS(CDN)**, **구글 웹폰트**를 꼭 포함하고,
  부드럽고 고급스러운 전용 CSS(직접 스타일)도 적극 사용한다.
* 전체 슬라이드는 `<div class=\"slide-container\">`로 감싸고,
  **디자인 설명에 따라 가장 적합한 레이아웃(단일 컬럼, 2단 분할, 이미지 위치 등)을 자유롭게 구성**한다.
* 이미지는 필요시 배경, 상단/하단, 좌우 등 디자인 설명에 맞게 배치하며,
  텍스트(타이틀, 부제, 설명 등)와의 조화도 신경 쓴다.
* 디자인 포인트(강조 색상, 구분선, 오버레이 등)는 입력 설명에 따라 적절히 활용한다.
* **스타일/폰트/정렬/색상 포인트 등도 세련되게 구성**한다.
* 사용자의 입력 설명을 그대로 반영하되,
  **레이아웃/사이즈/폰트/색상 등은 항상 이 포맷을 고정적으로 유지**한다.
* **설명, 안내문, 코드 외의 텍스트는 절대 출력하지 않는다.**
  **오직 완성된 HTML 코드만** 출력한다.
* 코드 결과물은 브라우저에서 바로 실행 가능한 **HTML 전체 문서**(head~body) 구조로 완성한다.

### 참고

* 입력에 명시된 모든 텍스트, 숫자, 이미지 URL, 색상 등은 반드시 코드에 반영한다.
* 항목이 없는 경우에도 전체 구조는 디자인 설명에 맞게 일관성 있게 유지한다.

---

**위 지침을 따라, 입력 텍스트 설명을
항상 1280×720px, 디자인 설명에 가장 적합한 레이아웃(단일, 2단, 기타)으로
완성된 HTML 코드로 변환해 출력하라.
설명 없이 코드만 제공한다.**
"""



class SlideDesignAgentState(AgentState):
    requirement: str
    design: str
    html: str

class SlideDesignNodeOutput(TypedDict):
    design: str

class HtmlGenerateNodeOutput(TypedDict):
    html: str

def create_slide_design_node(slide_design_llm):
    def slide_design_node(state: SlideDesignAgentState):
        system_instruction = SystemMessage(content=prompt_slide_design)
        response = slide_design_llm.invoke(state["messages"] + [system_instruction])
        return response
    return slide_design_node

def create_html_generate_node(html_generate_llm):
    def html_generate_node(state: SlideDesignAgentState):
        system_instruction = SystemMessage(content=prompt_html_generator)
        response = html_generate_llm.invoke([HumanMessage(content=state["design"])] + [system_instruction])
        return response
    return html_generate_node

def create_slide_create_agent(name=None):
    slide_design_llm = ChatOpenAI(model="gpt-4.1").with_structured_output(SlideDesignNodeOutput)
    html_generate_llm = ChatOpenAI(model="gpt-4.1").with_structured_output(HtmlGenerateNodeOutput)

    slide_design_node = create_slide_design_node(slide_design_llm)
    html_generate_node = create_html_generate_node(html_generate_llm)

    builder = StateGraph(SlideDesignAgentState)
    builder.add_node("slide_design_node", slide_design_node)
    builder.add_node("html_generate_node", html_generate_node)

    builder.add_edge(START, "slide_design_node")
    builder.add_edge("slide_design_node", "html_generate_node")

    builder.add_edge("html_generate_node", END)

    slide_crate_agent = builder.compile(name=name)
    return slide_crate_agent


if __name__ == '__main__':
    from test_msg import test_slide
    from dotenv import load_dotenv

    load_dotenv()

    slide_create_agent = create_slide_create_agent()

    response = slide_create_agent.invoke({"messages": test_slide})
    print(response['html'])

    # 생성된 HTML을 test.html로 저장
    with open("test.html", "w", encoding="utf-8") as f:
        f.write(response['html'])

