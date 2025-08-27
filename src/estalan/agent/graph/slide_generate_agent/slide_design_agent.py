import os
import json
from langchain_core.messages import HumanMessage
from typing import TypedDict, List
from langgraph.graph import START, END, StateGraph
from estalan.llm import create_chat_model
from estalan.messages.utils import create_ai_message
from estalan.agent.graph.slide_generate_agent.state import ExecutorState
from langgraph.prebuilt import create_react_agent
from estalan.agent.graph.slide_generate_agent.utils import get_html_template_list, get_html_template_content_tool
from estalan.agent.graph.slide_generate_agent.prompt.slide_design import prompt_slide_design
from estalan.tools.search import GoogleSerperImageSearchResult



class Image(TypedDict):
    title: str
    description: str
    url: str


class SlideDesignAgentState(ExecutorState):
    list_image: List[Image]
    guideline: dict


class ImageSearchAgentOutput(TypedDict):
    list_image: List[Image]

class SlideTemplateSelectNodeOutput(TypedDict):
    html_template: str
    guideline: dict

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

    content = f"""슬라이드 디자인을 시작합니다.
    """

    msg = create_ai_message(
        content=content,
        name="msg_slide_design_start",
        id="msg_slide_design_start"
    )

    return {"messages": [msg], "name": state["name"]}

def post_processing_slide_design_node(state):
    name = state["name"]

    content = f"""{name} 페이지 디자인을 완료하였습니다.
    """

    msg = create_ai_message(
        content=content,
        name="msg_slide_design_end",
        id="msg_slide_design_end"
    )
    return {}


def pre_processing_html_generate_node(state):
    name = state["name"]

    content = f"""슬라이드를 생성하고 있습니다.
    """

    msg = create_ai_message(
        content=content,
        name="msg_html_generate_start",
        id="msg_html_generate_start"
    )

    return {"messages": [msg], "name": state["name"]}

def post_processing_html_generate_node(state):
    name = state["name"]

    content = f"""{name} 페이지 슬라이드를 생성하였습니다.
    """

    msg = create_ai_message(
        content=content,
        name="msg_html_generate_end",
        id="msg_html_generate_end"
    )

    return {}


def pre_processing_image_search_node(state):
    content = f"""슬라이드에 사용할 이미지를 검색하고 있습니다.
    """

    msg = create_ai_message(
        content=content,
        name="msg_image_search_start",
        id="msg_image_search_start"
    )

    return {"messages": [msg]}

def post_processing_image_search_node(state):
    name = state["name"]

    content = f"""{name} 페이지에 사용할 이미지를 검색하였습니다.
    """

    msg = create_ai_message(
        content=content,
        name="msg_image_search_end",
        id="msg_image_search_end"
    )
    return {}


def create_slide_template_select_node(slide_design_react_agent):
    async def slide_template_select_node(state: SlideDesignAgentState):
        topic = state["topic"]
        name = state["name"]
        description = state["description"]
        content = state["content"]
        # img_url = state["img_url"]

        template_folder = state["template_folder"]

        list_html_file = get_html_template_list(template_folder)


        # Prompt template for React agent
        prompt_slide_template_select = f"""
You are a slide design expert. You need to select the most suitable HTML template for the given content.

## Data
Topic: {topic}
Section Name: {name}
Section Description: {description}
Content: {content}
Slide Type: {state.get("slide_type", "contents")}
Section Index: {state.get("idx", 0)}
Requirements: {state.get("requirements", [])}


## Template Selection Criteria:
- Suitability according to slide type (title, contents, etc.)
- Nature of content (text-focused, image-focused, data visualization, etc.)
- Select if used for similar tasks, even if content doesn't match exactly
- Layout suitability (horizontal/vertical arrangement, grid layout, etc.)
- Need for visual effects
- Reflection of requirements and design prompts


## Available HTML Template List:
{list_html_file}

## get_html_template_content_tool parameter
template_folder: {template_folder}

## Rules
1. Select the most suitable template from the above template list for the content.
2. Put the queried template html code into html_template without changes
3. Insert the template queried through get_html_template_content_tool.
4. ** Do not generate html arbitrarily. **
"""
        input_state = state.copy()
        input_state["messages"] = [HumanMessage(content=prompt_slide_template_select)]

        # 에이전트 실행
        for i in range(10):
            try:
                result = await slide_design_react_agent.ainvoke(input_state)

                for message in result['messages']:
                    if message.type == "tool":
                        tool_result = json.loads(message.content)

                # 결과에서 디자인 정보 추출
                return {
                    "html_template": tool_result['content'],
                    "guideline": tool_result['guideline']
                }
            except Exception as e:
                print(i, e)
    
    return slide_template_select_node


def create_slide_design_node(slide_design_llm):
    async def slide_design_node(state: SlideDesignAgentState):
        html_template = state["html_template"]
        topic = state["topic"]
        name = state["name"]
        description = state["description"]
        content = state["content"]
        # img_url = state["img_url"]

        # Prompt template for React agent
        msg = f"""
{prompt_slide_design}

## Data
Topic: {topic}
Section Name: {name}
Section Description: {description}
Content: {content}
Slide Type: {state.get("slide_type", "contents")}
Section Index: {state.get("idx", 0)}
Requirements: {state.get("requirements", [])}
"""

        for i in range(10):
            try:
                result = await slide_design_llm.ainvoke([
                    HumanMessage(content=msg),
                ])

                design = result['design']
                list_image = result['list_image']
                for img in list_image:
                    img['url'] = ""
                print(result)
                break
            except Exception as e:
                print(i, e)

        return {'design': design, "list_image": list_image}
    return slide_design_node


def create_image_search_agent(agent):
    async def image_search_agent(state: SlideDesignAgentState):
        list_image = state["list_image"]

        str_list_image = "list_image information\n"
        for img in list_image:
            str_list_image += f"title: {img['title']}\ndescription: {img['description']}\n\n"
        
        msg = create_ai_message(content=f"Search for images that match the title and description of list_image, and update the url. Don't ask additional questions and perform the task. \n\n{str_list_image}")

        for i in range(10):
            try:
                result = await agent.ainvoke(
                    {
                        "messages": [msg],
                        "list_image": list_image,
                    }
                )
                list_image = result['structured_response']['list_image']
                design = state['design']

                design += f"\n\n Searched Images \n"
                for img in list_image:
                    design += f"\ntitle: {img['title']}\ndescription: {img['description']} \n url: {img['url']}\n\n"
                break
            except Exception as e:
                print(i, e)

        # print(design)
        return {"list_image": list_image, "design": design}
    return image_search_agent


def create_html_generate_node(html_generate_llm):
    async def html_generate_node(state: SlideDesignAgentState):
        # design이 없으면 기본값 사용
        design_content = state.get("design", "기본 디자인을 적용합니다.")

        html_template = state["html_template"]
        guideline = state["guideline"]
        
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
        try:
            for img in list_image:
                str_list_image += f"\ntitle: {img['title']}\ndescription: {img['description']} \n url: {img['url']}\n\n"
        except Exception as e:
            print(e)
            print(list_image)


        msg_content = f"""
Generate a slide based on the content below.

# Slide Information
Topic: {topic}
Section Name: {name}
Section Description: {description}
Slide Type: {state.get("slide_type", "contents")}
Section Index: {state.get("idx", 0)}
Requirements: {state.get("requirements", [])}

# Content
{content}

# Available Image Information
{str_list_image}

# Guideline
{guideline}

# HTML Template
{html_template}

# Generation Guidelines
## Prohibited Actions
- Do not change HTML tag structures like <div>, <h3>, <p>, <i>, etc.
- Do not change CSS class names like class="text-2xl font-bold mb-3 title-text"
- Do not change template layout: maintain existing container structure
- Do not add new HTML elements: do not add new tags or structures not in template

## Allowed Operations
- Replace text content only: change only text inside <p>, <h3> tags
- Change icons: change only the fa-hiking part in <i class="fas fa-hiking"> to appropriate icons
- Replace image URLs: change only the src attribute value

## Other Guidelines
1. Use appropriate titles and layouts suitable for the slide type
2. Configure slides reflecting requirements and design prompts
3. Place images in appropriate positions when image URLs are provided
4. Maximize text areas for rich content while maintaining readability
5. Include specific examples and explanations rather than simple keyword listings
6. Content should not exceed what is specified in the guideline
7. **Exclude meta descriptions and include only core content the audience needs to see**

## Example Guidelines
Focus on direct, actionable content rather than meta-descriptions about creating the slide.

## ❌ Wrong Example (included meta descriptions):
```html
<h1>맛집 선정 기준</h1>
<p>"떡볶이 맛집 선정 기준" 슬라이드를 작성하겠습니다. 이 슬라이드는 "전국 떡볶이 맛집 도장깨기 프로젝트"를 위한 객관적이고 매력적인 맛집 선정 기준을 명확히 제시하는 것이 목적입니다.</p>
<ul>
  <li>맛의 균형성을 평가합니다</li>
  <li>가격 대비 만족도를 고려합니다</li>
</ul>
...
```

## ✅ Correct Example:
```html
<h1>떡볶이 맛집 선정 기준</h1>
<div class="criteria-section">
  <h2>맛 평가 기준</h2>
  <ul>
    <li><strong>소스의 균형</strong>: 단맛, 매운맛, 감칠맛의 조화</li>
    <li><strong>떡의 식감</strong>: 쫄깃함과 부드러움의 적절한 비율</li>
    <li><strong>토핑 퀄리티</strong>: 어묵, 만두, 야채의 신선도</li>
  </ul>
  
  <h2>서비스 평가 기준</h2>
  <ul>
    <li><strong>가격 합리성</strong>: 1인분 기준 3,000~5,000원</li>
    <li><strong>접근성</strong>: 대중교통 도보 10분 이내</li>
    <li><strong>위생 상태</strong>: 청결한 조리 환경과 식기</li>
  </ul>
</div>
...
```

"""
        for i in range(10):
            try:
                response = await html_generate_llm.ainvoke([
                    HumanMessage(content=msg_content),
                ])
                break
            except Exception as e:
                print(i, e)
        return response

    return html_generate_node

def create_slide_create_agent(name=None):
    serper_api_key = os.getenv("SERPER_API_KEY")

    search_img_tool = GoogleSerperImageSearchResult.from_api_key(
        api_key=serper_api_key,
        k=5,
    )


    # React 에이전트용 LLM (structured output 불필요)
    # slide_template_select_llm = create_chat_model(provider="google_vertexai", model="gemini-2.5-flash")
    slide_template_select_llm = create_chat_model(provider="azure_openai", model="gpt-5-mini")
    slide_design_llm = create_chat_model(provider="azure_openai", model="gpt-5-mini", lazy=True).with_structured_output(SlideDesignNodeOutput)
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
    builder.add_node("pre_processing_image_search_node", pre_processing_image_search_node)
    builder.add_node("post_processing_image_search_node", post_processing_image_search_node)


    builder.add_node("slide_template_select_node", slide_template_select_node)
    builder.add_node("slide_design_node", slide_design_node)
    builder.add_node("image_search_node", image_search_node)
    builder.add_node("html_generate_node", html_generate_node)

    builder.add_edge(START, "pre_processing_node")
    builder.add_edge("pre_processing_node", "pre_processing_slide_design_node")
    builder.add_edge("pre_processing_slide_design_node", "slide_template_select_node")
    builder.add_edge("slide_template_select_node", "slide_design_node")
    builder.add_edge("slide_design_node", "pre_processing_image_search_node")
    builder.add_edge("pre_processing_image_search_node", "image_search_node")
    builder.add_edge("image_search_node", "post_processing_image_search_node")
    builder.add_edge("post_processing_image_search_node", "pre_processing_html_generate_node")
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
    # image_search_llm = create_chat_model(provider="google_vertexai", model="gemini-2.5-flash")
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
