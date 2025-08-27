section_writer_instructions = """
Write one section for a presentation slide in Korean.

<Target Audience>
- general public
</Target Audience>

<Task>
1) Carefully read the report topic, section name, and section topic.
2) Review any existing section content, if provided.
3) Use the search_tool to find domain-specific, high-signal sources.
4) Synthesize findings into concrete informable content or concrete actionable content for slides.
5) Output only the required fields of the schema.
</Task>

<Hard Rules>
- Prohibited: meta sections/content such as overview/intro/background/research methodology/API/crawler/data-collection pipelines.
- Focus: Specific information tailored to the audience, or suggestions based on it
- Write in Korean. Avoid generic statements; ground claims in sources.
</Hard Rules>

<Search Keyword Guidance>
- Tailor queries to the topic domain. 
- For travel/restaurant topics, include location names + core keywords (restaurants/reviews/addresses/maps/hours/prices/waiting/hygiene) in queries, and prefer credible map/blog/review sources.
- For corporate/industry topics, include official sites/press releases/business reports/market research keywords.

<Writing Guidelines>
- If existing section content is not populated, write from scratch
- If existing section content is populated, synthesize it with the source material
- Strict 800-1200 word limit
- Use simple, clear language
- Use short paragraphs (2-3 sentences max)
- Use ## for section title (Markdown format)
</Writing Guidelines>

<Final Check>
1) Verify that all evidence is based on specified sources
2) Each URL appears only once in the source list
3) Source numbers are assigned sequentially 1,2,3... (no omissions allowed)
</Final Check>
"""

section_search_img_instruction = """
Select 1 image that is most relevant to the report topic, section name, and section topic.

<Task>
1. Carefully review the report topic, section name, and section topic.
2. Select 1 image that is most relevant to the report topic, section name, and section topic.
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
