from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from estalan.llm.utils import create_chat_model
from estalan.agent.base.node import create_initialization_node
from estalan.agent.graph.requirement_collection_agent.state import RequirementCollectionAgentState
from estalan.utils import get_last_human_message



def create_extract_requirements_node(extract_llm, private_state_key):
    """사용자 답변에서 요구사항 추출"""

    async def extract_requirements(state: RequirementCollectionAgentState):
        # 마지막 사용자 메시지에서 답변 추출
        user_message = get_last_human_message(state["messages"])

        # requirement_collection_agent_private_state에서 필요한 정보 추출
        private_state = state.get(private_state_key, {})
        
        purpose = private_state.get("purpose", "")
        new_questions = private_state.get("new_questions", [])
        existing_requirements_md = private_state.get("requirements", "")
        
        # LLM을 사용하여 요구사항을 마크다운 형식으로 직접 관리
        prompt = f"""
당신은 {purpose}를 위한 요구사항 분석 에이전트입니다.

## 작업 목표
사용자의 답변에서 새로운 요구사항을 추출하고, 기존 요구사항과 통합하여 마크다운 형식으로 정리합니다.

## 입력 정보
- **질문**: {new_questions}
- **사용자 답변**: {user_message}

## 기존 요구사항
{existing_requirements_md if existing_requirements_md else "수집된 요구사항이 없습니다."}

## 출력 형식
다음 마크다운 형식으로 응답해주세요:

```markdown
## 요구사항

### 카테고리명
- 요구사항 내용

### 카테고리명
- 요구사항 내용
```

## 요구사항 관리 규칙

### 1. 중복 방지
- **동일한 카테고리 + 동일한 내용**: 중복으로 간주하여 추가하지 않음
- **예시**: "여행 계획 - 유럽 여행"이 이미 있다면 동일한 내용을 다시 추가하지 않음

### 2. 같은 카테고리 내 여러 요구사항 허용
- 같은 카테고리라도 서로 다른 내용이면 별도 요구사항으로 추가
- **예시**:
  - [여행 계획] - 유럽 여행 계획 만들기
  - [여행 계획] - 예산 계획 수립
  - [여행 계획] - 교통편 예약

### 3. 카테고리 분류 기준
- **목표**: 최종 목표나 달성하고자 하는 것
- **장소/목적지**: 여행지, 방문할 곳
- **일정**: 시간, 기간, 스케줄
- **예산**: 비용, 금액 관련
- **교통**: 이동 수단, 교통편
- **숙박**: 숙박 시설, 호텔
- **활동**: 관광, 체험, 할 일
- **기타**: 위에 해당하지 않는 요구사항

### 4. 요구사항 작성 가이드
- **구체적이고 명확하게**: "좋은 음식점" → "맛집 추천"
- **행동 가능한 형태로**: "편리하게" → "접근성이 좋은 장소"
- **측정 가능하게**: "적당한 가격" → "5만원 이하"

### 5. 마크다운 형식 규칙
- 헤딩은 `## 요구사항` (depth 2)로 시작
- 카테고리는 `### 카테고리명` (depth 3)으로 표시
- 각 요구사항은 `- 내용` 형태의 bullet point로 표시
- 카테고리별로 빈 줄로 구분

## 예시

### 입력
사용자: "유럽 여행을 3박 4일로 계획하고 싶어요. 예산은 100만원 정도로 잡고 있어요."

### 출력
```markdown
## 요구사항

### 목적지
- 유럽 여행

### 일정
- 3박 4일

### 예산
- 100만원
```

위 규칙을 따라 사용자 답변에서 요구사항을 추출하고, 기존 요구사항과 통합하여 마크다운 형식으로 정리해주세요.
"""

        response = await extract_llm.ainvoke([HumanMessage(content=prompt)])
        
        # 상태 업데이트 (마크다운 문자열로 저장)
        return {
            private_state_key: {
                "requirements": response.content,
            }
        }
    
    return extract_requirements


def create_print_requirement_node(private_state_key):
    """요구사항을 마크다운 형식으로 출력"""

    def print_requirement_node(state: RequirementCollectionAgentState):
        private_state = state.get(private_state_key, {})
        requirements = private_state.get("requirements", "")
        
        if not requirements or requirements == "## 요구사항\n\n수집된 요구사항이 없습니다.":
            msg = "## 📝 요구사항 수집 현황\n\n아직 수집된 요구사항이 없습니다.\n\n질문을 통해 요구사항을 수집해주세요."
        else:
            msg = f"""## 📋 수집된 요구사항

{requirements}

---

**위와 같은 요구사항으로 작업을 진행할까요?**

추가로 수집하고 싶은 요구사항이 있으시면 말씀해주세요."""

        return {"messages": [AIMessage(content=msg)]}
    
    return print_requirement_node


def create_requirement_analysis_agent(purpose=None, predefined_questions=[], name="requirement_analysis_agent", private_state_key="requirement_collection_agent_private_state"):
    """요구사항 분석 에이전트 생성"""

    def create_custom_initialization_node():
        """커스텀 초기화 노드 생성"""
        def initialization_node(state):
            private_state = state.get(private_state_key, {})
            
            if not private_state.get("initialization", False):
                updated_private_state = {
                    "purpose": purpose or "",
                    "requirements": "## 요구사항\n\n수집된 요구사항이 없습니다.",
                    "questions": predefined_questions or [],
                    "new_questions": [],
                    "needs_more_questions": True,
                    "initialization": True
                }
                
                return {
                    private_state_key: updated_private_state
                }
            return {}
        
        return initialization_node

    initialization_node = create_custom_initialization_node()

    # LLM 모델 생성
    extract_llm = create_chat_model(provider="azure_openai", model="gpt-4.1")
    
    # 노드 생성
    extract_requirements_node = create_extract_requirements_node(extract_llm, private_state_key)
    print_requirement_node = create_print_requirement_node(private_state_key)
    
    # 그래프 구성
    builder = StateGraph(RequirementCollectionAgentState)

    # 노드 추가
    builder.add_node("initialization_node", initialization_node)
    builder.add_node("extract_requirements", extract_requirements_node)
    builder.add_node("print_requirement_node", print_requirement_node)

    # 엣지 추가
    builder.add_edge(START, "initialization_node")
    builder.add_edge("initialization_node", "extract_requirements")
    builder.add_edge("extract_requirements", "print_requirement_node")
    builder.add_edge("print_requirement_node", END)

    requirement_analysis_agent = builder.compile(name=name)
    return requirement_analysis_agent

