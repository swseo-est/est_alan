section_writer_instructions = """Write one section of a research report.

<Task>
1. Review the report topic, section name, and section topic carefully.
2. If present, review any existing section content. 
3. Then, look at the provided Source material.
4. Decide the sources that you will use it to write a report section.
5. Write the report section and list your sources. 
</Task>

<Writing Guidelines>
- If existing section content is not populated, write from scratch
- If existing section content is populated, synthesize it with the source material
- Strict 800-1500 word limit
- Use simple, clear language
- Use short paragraphs (2-3 sentences max)
- Use ## for section title (Markdown format)
</Writing Guidelines>

<Final Check>
1. Verify that EVERY claim is grounded in the provided Source material
2. Confirm each URL appears ONLY ONCE in the Source list
3. Verify that sources are numbered sequentially (1,2,3...) without any gaps
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
