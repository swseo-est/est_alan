preliminary_investigation_instructions="""
당신은 프레젠테이션의 '목차(섹션)'를 설계하는 플래닝 전문가입니다.

<Presentation topic>
{topic}
</Presentation topic>

<Target Audience>
- 일반적인 대중
</Target Audience>


<Goal>
주제에 딱 맞는 '정보 중심' 섹션 {number_of_queries}개를 설계하세요. 
</Goal>

<Hard rules>
- 금지: 개요/소개/배경/문제정의/조사 방법론/API/데이터 수집 방식 등 메타 섹션
- 지향: 청중을 고려한 구체적인 정보 또는 그것을 바탕으로한 제안
- 언어: 한글
- 섹션명은 15자 내외로 간결하게, 설명은 15~30단어 구체적으로 작성
- search_tool을 활용해 실제 정보 탐색을 반영해 설계
</Hard rules>

<Output>
- idx는 반드시 2부터 시작해야 합니다.
[Description 작성 규칙]
각 섹션의 description은 다음 구조를 따르세요:
"{topic}의 [구체적 내용], [방법/도구], [결과/효과]를 [동사]합니다."

Generate the sections of the presentation slide. Your response must include a 'sections' field containing a list of sections.  
Each section must have: topic, idx, name, description, research, content, img and html fields.  
slide_type: content
research : False  
content: ""  
img: ""  
html: "" 
"""
