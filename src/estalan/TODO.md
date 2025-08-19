## deployment
- estalan/agent/graph 에 정의된 각 에이전트 관련 코드들을 전부 이곳으로 가져올 것
- 각 agent에 대한 config를 어떻게 관리할지 데이터 구조 정의 필요
  - 가독성을 위해 yaml 방식으로 정의
    - config는 agent 내부에 정의된 llm provider, model에 대한 구조가 포함되어야함
    - config_schema에 대한 정의가 필요함
      - langgraph config_schema를 좀 더 공부해보고, 일반적일 구조로 yaml을 정의해보자
- 각 agent를 테스트할 수 있는 코드는 deployment 내에 각 agent 폴더에 정의하는 것이 좋아보임
- notebook도 폴더를 추가

## estalan
### agent
- 내부 폴더 및 코드 구조 리팩토링 필요
- 현재 agent/graph 내부에 agent관련 코드가 들어가 있는데, 해당 코드들을 라이브러리로서 기능을 위해 필요한 것이 아니고 에이전트들을 정의한 코드
  - 따라서, 관련 코드들은 deployment로 옮기는 것이 타당함.
  - 현재는 deployment 폴더의 역할이 너무 적음

### core
- deep research 및 검색 에이전트에 정의된 도구를 사용하기 위해 임시로 붙여놓은 코드, 나중에 정리가 필요함

### deployment
-  서버 운용 중에 발생하는 장애에 대한 예외처리를 지속적으로 추가 필요.

### llm
- Structured Output이 정의된 경우, 해당 structured output을 검증하는 로직을 wrapper에 추가
  - 외부에서 정의된 validation fuction을 input으로 받을 수 있도록
  - default로는 structured output에 정의된 field를 llm이 출력을 했는지 확인
  
### logging
- 내부적으로 사용할 logger를 정의
- 
### messages
- 백엔드 및 프론트와 합의하여 message tag를 정의

### prebuilt
- 재활용 가능성이 높고, 안정성이 확인된 agent의 경우 prebuilt에 추가 ex) requirement_agent, supervisor_agent, react_agent, mcp_agent 등

### tools
- tool에 대한 규격화된 정의가 필요함, 현재는 형태가 너무 다양함 (우선순위는 낮음, 추후 필요시)


## tests
- 현재 테스트 코드에 대한 부분이 많이 미흠함

