# info.json Guideline 생성 관련 가이드

html 작성 시 컨텐츠의 양을 제어하기 위해 template 정보에 guideline을 추가하는 것이 목적입니다.

# 진행
- 사용 LLM: Claude 4.0 Sonnet
- 입력 파일: html 파일들, info.json
- instruction:
```
해당 html은 LLM이 html을 만들기 위한 예시 템플릿이야. 그런데 내부 텍스트 길이에 대한 guideline을 주지 않으면 제대로 안먹는거 같아서 guideline을 만들려고 하는데, (e.g. text-item 컴포넌트(? 이거 맞는지도 체크) 는 300자 내외로 작성할것) 일단 각각 다 해봐

...

좋아, 해당 설명을 적절히 dict마다 guideline이라는 key로 넣고 싶은데 해줘봐

...

응 guideline이상한데? class="text-5xl font-bold mb-10 title-text" 이렇게 class로 해야 알아듣지 않을까?

```