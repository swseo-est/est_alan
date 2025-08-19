section_writer_instructions = """
Write one section for a slide-oriented research report in Korean.

<Task>
1) Carefully read the report topic, section name, and section topic.
2) Review any existing section content, if provided.
3) Use the search_tool to find domain-specific, high-signal sources.
4) Synthesize findings into concrete, actionable content for slides.
5) Output only the required fields of the schema.
</Task>

<Hard Rules>
- Prohibited: meta sections/content such as overview/intro/background/research methodology/API/crawler/data-collection pipelines.
- Focus: concrete details users can execute now (routes, schedules, area breakdowns, selection criteria, checklists, budget/transport, risks, deliverables/templates).
- If the section name implies methodology or overview, reinterpret into concrete actions for the given topic.
- Write in Korean. Avoid generic statements; ground claims in sources.
</Hard Rules>

<Search Keyword Guidance>
- Tailor queries to the topic domain. 
- For travel/맛집 topics, include 지역명 + 핵심키워드(맛집/리뷰/주소/지도/영업시간/가격/웨이팅/위생) in queries, and prefer credible map/blog/review sources.
- For 기업/산업 주제, include 공식 사이트/보도자료/사업보고서/시장조사 키워드.

<Writing Guidelines>
- If existing section content is not populated, write from scratch
- If existing section content is populated, synthesize it with the source material
- Strict 800-1200 word limit
- Use simple, clear language
- Use short paragraphs (2-3 sentences max)
- Use ## for section title (Markdown format)
</Writing Guidelines>

<Final Check>
1) 모든 근거는 명시된 Source에 기반하는지 확인
2) 각 URL은 Source 목록에 한 번만 등장
3) 출처 번호는 1,2,3... 순차 부여(누락 금지)
</Final Check>
"""

section_search_img_instruction = """
보고서 주제, 섹션 이름, 섹션 주제와 가장 연관성이 높은 이미지를 1장 선택하세요.

<Task>
1. 보고서 주제, 섹션 이름, 섹션 주제를 꼼꼼하게 검토합니다.
2. 보고서 주제, 섹션 이름, 섹션 주제와 가장 연관성이 높은 이미지를 1장 선택하세요.
</Task>
"""

section_writer_inputs=""" 
<Report topic>
{topic}
</Report topic>

<Section name>
{section_name}
</Section name>

<Section topic>
{section_topic}
</Section topic>

<Content>
{content}
</Content>
"""
