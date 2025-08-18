import os
from langchain_core.messages import HumanMessage
from typing import TypedDict, List
from langgraph.graph import START, END, StateGraph
from estalan.llm import create_chat_model
from estalan.messages.utils import create_ai_message
from estalan.agent.graph.slide_generate_agent.state import ExecutorState
from langgraph.prebuilt import create_react_agent
from estalan.agent.graph.slide_generate_agent.utils import get_html_template_list, get_html_template_content_tool
from estalan.agent.graph.slide_generate_agent.prompt.slide_design import prompt_slide_design
from langgraph.prebuilt.chat_agent_executor import AgentState, AgentStateWithStructuredResponse
from estalan.tools.search import GoogleSerperSearchResult, GoogleSerperImageSearchResult



class Image(TypedDict):
    title: str
    description: str
    url: str


class SlideDesignAgentState(ExecutorState):
    list_image: List[Image]


class ImageSearchAgentOutput(TypedDict):
    list_image: List[Image]

class SlideTemplateSelectNodeOutput(TypedDict):
    html_template: str

class SlideDesignNodeOutput(TypedDict):
    design: str
    list_image: List[Image]

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
    슬라이드 디자인을 시작합니다.
    """

    msg = create_ai_message(
        content=content,
        name="msg_slide_design_start",
        id="msg_slide_design_start"
    )

    return {"messages": [msg], "name": state["name"]}

def post_processing_slide_design_node(state):
    name = state["name"]

    content = f"""
    {name} 페이지 디자인을 완료하였습니다.
    """

    msg = create_ai_message(
        content=content,
        name="msg_slide_design_end",
        id="msg_slide_design_end"
    )
    return {}


def pre_processing_html_generate_node(state):
    name = state["name"]

    content = f"""
    슬라이드를 생성하고 있습니다.
    """

    msg = create_ai_message(
        content=content,
        name="msg_html_generate_start",
        id="msg_html_generate_start"
    )

    return {"messages": [msg], "name": state["name"]}

def post_processing_html_generate_node(state):
    name = state["name"]

    content = f"""
    {name} 페이지 슬라이드를 생성하였습니다.
    """

    msg = create_ai_message(
        content=content,
        name="msg_html_generate_end",
        id="msg_html_generate_end"
    )

    return {}


def create_slide_template_select_node(slide_design_react_agent):
    async def slide_template_select_node(state: SlideDesignAgentState):
        topic = state["topic"]
        name = state["name"]
        description = state["description"]
        content = state["content"]
        img_url = state["img_url"]

        template_folder = state["template_folder"]

        list_html_file = get_html_template_list(template_folder)


        # React 에이전트를 위한 프롬프트 템플릿
        prompt_slide_template_select = f"""
당신은 슬라이드 디자인 전문가입니다. 주어진 내용에 가장 적합한 HTML 템플릿을 선택하고 디자인을 제안해야 합니다.

## 데이터
주제: {topic}
섹션명: {name}
섹션 설명: {description}
내용: {content}
슬라이드 타입: {state.get("slide_type", "contents")}
섹션 인덱스: {state.get("idx", 0)}
요구사항: {state.get("requirements", [])}


## 템플릿 선택 기준:
- 슬라이드 타입 (title, contents 등)에 따른 적합성
- 내용의 성격 (텍스트 중심, 이미지 중심, 데이터 시각화 등)
- 레이아웃의 적합성 (가로/세로 배치, 그리드 레이아웃 등)
- 시각적 효과의 필요성
- 요구사항과 디자인 프롬프트의 반영

# template_folder: {template_folder}
### 사용 가능한 HTML 템플릿 목록:
{list_html_file}

## html_template 작성 규칙
1. 위의 템플릿 목록에서 내용에 가장 적합한 템플릿을 선택하세요.
2. 선택한 템플릿의 내용을 get_html_template_content_tool 도구로 확인하여 구체적인 구조를 파악하세요.
3. 만일, template의 html 코드가 내용에 적합하지 않다면 다른 template를 조회하세요.
4. 조회한 template html 코드를 html_template에 변경없이 넣으세요
"""
        input_state = state.copy()
        input_state["messages"] = [HumanMessage(content=prompt_slide_template_select)]

        # 에이전트 실행
        result = await slide_design_react_agent.ainvoke(input_state)

        # 결과에서 디자인 정보 추출
        return {"html_template": result['structured_response']['html_template']}
    
    return slide_template_select_node


def create_slide_design_node(slide_design_llm):
    async def slide_design_node(state: SlideDesignAgentState):
        html_template = state["html_template"]
        topic = state["topic"]
        name = state["name"]
        description = state["description"]
        content = state["content"]
        img_url = state["img_url"]

        # React 에이전트를 위한 프롬프트 템플릿
        msg = f"""
## guide
{prompt_slide_design}

## 데이터
주제: {topic}
섹션명: {name}
섹션 설명: {description}
내용: {content}
슬라이드 타입: {state.get("slide_type", "contents")}
섹션 인덱스: {state.get("idx", 0)}
요구사항: {state.get("requirements", [])}

## html_template
{html_template}
"""

        result = await slide_design_llm.ainvoke([
            HumanMessage(content=msg),
        ])

        design = result['design']
        print(design)

        msg2 = f"아래 design에서 list_image를 추출하세요. \n {design}"
        result2 = await slide_design_llm.ainvoke([
            HumanMessage(content=msg2)
        ])

        list_image = result2['list_image']

        design += f"\n\n {content}"


        return {'design': design, "list_image": list_image}
    return slide_design_node


def create_image_search_agent(agent):
    async def image_search_agent(state: SlideDesignAgentState):
        list_image = state["list_image"]

        str_list_image = "list_image 정보\n"
        for img in list_image:
            str_list_image += f"title: {img['title']}\ndescription: {img['description']}\n\n"
        
        msg = create_ai_message(content=f"list_image의 title과 description에 맞는 이미지를 검색하고, url을 업데이트 하세요. 추가 질문은 하지말고 작업을 수행하세요. \n\n{str_list_image}")

        result = await agent.ainvoke(
            {
                "messages": [msg],
                "list_image": list_image,
            }
        )
        return {"list_image": result['structured_response']['list_image']}
    return image_search_agent



def create_html_generate_node(html_generate_llm):
    async def html_generate_node(state: SlideDesignAgentState):
        # design이 없으면 기본값 사용
        design_content = state.get("design", "기본 디자인을 적용합니다.")

        html_template = state["html_template"]
        
        # design_prompt가 존재하면 이를 활용
        if state.get("design_prompt"):
            # design_prompt를 포함한 강화된 디자인 지시사항 생성
            enhanced_design = f"{design_content}\n\n추가 디자인 요구사항: {state['design_prompt']}"
            design_content = enhanced_design

        topic = state["topic"]
        name = state["name"]
        description = state["description"]
        content = state["content"]
        list_image = state["list_image"]

        str_list_image = ""
        for img in list_image:
            str_list_image += f"\ntitle: {img['title']}\ndescription: {img['description']} \n url: {img['url']}\n\n"


        msg_content = f"""
아래 내용을 기반으로 슬라이드를 생성하세요.
html template과 동일한 포맷으로 슬라이드를 생성하세요

# 슬라이드 정보
주제: {topic}
섹션명: {name}
섹션 설명: {description}
슬라이드 타입: {state.get("slide_type", "contents")}
섹션 인덱스: {state.get("idx", 0)}
요구사항: {state.get("requirements", [])}
디자인 프롬프트: {state.get("design_prompt", "")}

# content
{content}

# 사용가능한 이미지 정보
{str_list_image}

# design guide
{design_content}

# html template
```html
{html_template}
```

# 생성 지침
1. HTML 템플릿을 참고해서 슬라이드를 생성하세요.
2. 슬라이드 타입에 맞는 적절한 제목과 레이아웃을 사용하세요
3. 요구사항과 디자인 프롬프트를 반영하여 슬라이드를 구성하세요
4. 이미지 URL이 제공된 경우 적절한 위치에 배치하세요
5. 색상, 폰트, 레이아웃은 템플릿의 디자인 가이드를 따르세요
6. 이미지가 영역을 초과하지 않도록, 적절리 resize하거나 crop 하세요
"""
        response = await html_generate_llm.ainvoke([
            HumanMessage(content=msg_content),
        ])
        return response
    return html_generate_node

def create_slide_create_agent(name=None):
    serper_api_key = os.getenv("SERPER_API_KEY")

    search_img_tool = GoogleSerperImageSearchResult.from_api_key(
        api_key=serper_api_key,
        k=5,
    )


    # React 에이전트용 LLM (structured output 불필요)
    slide_template_select_llm = create_chat_model(provider="azure_openai", model="gpt-5-mini")
    slide_design_llm = create_chat_model(provider="azure_openai", model="gpt-5-mini").with_structured_output(SlideDesignNodeOutput)
    image_search_llm = create_chat_model(provider="azure_openai", model="gpt-5-mini")
    html_generate_llm = create_chat_model(provider="azure_openai", model="gpt-5-mini").with_structured_output(HtmlGenerateNodeOutput)

    # React 에이전트 생성
    tools = [get_html_template_content_tool]
        
    slide_design_react_agent = create_react_agent(
        model=slide_template_select_llm,
        tools=tools,
        response_format=SlideTemplateSelectNodeOutput
    )

    image_search_agent = create_react_agent(
        model=image_search_llm,
        tools=[search_img_tool],
        response_format=ImageSearchAgentOutput
    )


    slide_template_select_node = create_slide_template_select_node(slide_design_react_agent)
    slide_design_node = create_slide_design_node(slide_design_llm)
    image_search_node = create_image_search_agent(image_search_agent)
    html_generate_node = create_html_generate_node(html_generate_llm)

    builder = StateGraph(SlideDesignAgentState)
    builder.add_node("pre_processing_node", pre_processing_node)
    builder.add_node("post_processing_node", post_processing_node)
    builder.add_node("pre_processing_slide_design_node", pre_processing_slide_design_node)
    builder.add_node("post_processing_slide_design_node", post_processing_slide_design_node)
    builder.add_node("pre_processing_html_generate_node", pre_processing_html_generate_node)
    builder.add_node("post_processing_html_generate_node", post_processing_html_generate_node)


    builder.add_node("slide_template_select_node", slide_template_select_node)
    builder.add_node("slide_design_node", slide_design_node)
    builder.add_node("image_search_node", image_search_node)
    builder.add_node("html_generate_node", html_generate_node)

    builder.add_edge(START, "pre_processing_node")
    builder.add_edge("pre_processing_node", "pre_processing_slide_design_node")
    builder.add_edge("pre_processing_slide_design_node", "slide_template_select_node")
    builder.add_edge("slide_template_select_node", "slide_design_node")
    builder.add_edge("slide_design_node", "image_search_node")
    builder.add_edge("image_search_node", "pre_processing_html_generate_node")
    builder.add_edge("pre_processing_html_generate_node", "html_generate_node")
    builder.add_edge("html_generate_node", "post_processing_html_generate_node")
    builder.add_edge("post_processing_html_generate_node", "post_processing_node")
    builder.add_edge("post_processing_node", END)


    slide_crate_agent = builder.compile(name=name)
    return slide_crate_agent


if __name__ == '__main__':
    from dotenv import load_dotenv
    import asyncio
    import os

    load_dotenv()

    # 1. 전체 슬라이드 생성 에이전트 테스트
    print("=== 전체 슬라이드 생성 에이전트 테스트 ===")
    slide_create_agent = create_slide_create_agent()

    test_state = {
        'topic': '제주도 여행 가이드',
        'name': '제주도 소개',
        'description': '제주도의 기본 정보와 특징을 소개하는 섹션',
        'content': '## 제주도 소개\n\n제주도는 한국의 가장 큰 섬으로, 아름다운 자연과 독특한 문화를 가지고 있습니다. 화산 활동으로 형성된 섬으로, 한라산을 중심으로 한 자연 경관이 뛰어납니다.\n\n### 주요 특징\n- 화산섬으로 형성된 독특한 지형\n- 아름다운 해변과 바다 경관\n- 독특한 제주 문화와 전통\n- 다양한 관광 명소와 활동',
        'img_url': 'https://example.com/jeju-image.jpg',
        "template_folder": "general",
        'metadata': {
            'topic': '제주도 여행 가이드',
            'requirements': '제주도 여행 가이드 슬라이드',
            'num_sections': 5,
            'num_slides': 7,
            'status': 'start'
        }
    }

    response = asyncio.run(slide_create_agent.ainvoke(test_state))
    print("생성된 HTML:")
    print(response.get('html', 'HTML이 생성되지 않았습니다.'))

    # 생성된 HTML을 test_slide.html로 저장
    if 'html' in response:
        with open("test_slide.html", "w", encoding="utf-8") as f:
            f.write(response['html'])
        print("\nHTML이 test_slide.html 파일로 저장되었습니다.")

    # serper_api_key = os.getenv("SERPER_API_KEY")
    #
    # search_img_tool = GoogleSerperImageSearchResult.from_api_key(
    #     api_key=serper_api_key,
    #     k=5,
    # )
    #
    #
    # image_search_llm = create_chat_model(provider="azure_openai", model="gpt-5-mini")
    # image_search_agent = create_react_agent(
    #     model=image_search_llm,
    #     tools=[search_img_tool],
    #     response_format=ImageSearchAgentOutput
    # )
    #
    # image_search_node = create_image_search_agent(image_search_agent)
    #
    # state = {"list_image": [{'title': '제주 풍경(한라산과 해안)', 'description': '슬라이드 전체 배경으로 사용. 하늘·바다·산이 함께 보이는 넓은 가로 비율의 풍경 이미지.'}, {'title': '한라산 클로즈업', 'description': '소개 문구 옆이나 카드의 작은 일러스트로 사용 가능한 세로 또는 정사각형 이미지.'}, {'title': '제주 해변(맑은 바다)', 'description': '카드나 체크리스트 옆 비주얼로 활용 가능한 가로 이미지.'}, {'title': '제주 문화/전통 이미지', 'description': "해녀나 전통가옥 등 '독특한 제주 문화'를 시각화하기 위한 작은 사진 또는 아이콘형 이미지."}, {'title': '활동 이미지(관광·하이킹·레저)', 'description': '다양한 관광 명소와 활동을 나타내는 장면 이미지로 카드 썸네일에 사용.'}]}
    # result = asyncio.run(image_search_node(state))
    # print(result)
