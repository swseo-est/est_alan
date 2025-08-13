# Browser-Use-Agent 설계서 (최신화)

---

## 1. 시스템 개요

본 시스템은 LangGraph 스타일의 그래프 기반 multi-agent architecture로, 브라우저 환경(예: 쿠팡, 네이버 등)에서 복합적인 사용자 요청을 자율적으로 수행한다. 주요 agent는 Planner, Navigator, Validator로 구성되며, Playwright 기반 tool을 활용해 실제 브라우저를 제어한다.

---

## 2. 주요 Agent 역할 및 책임

### 2.1 Planner Agent

* **역할:** 사용자 요구와 전체 state(브라우저 상태, 액션 결과, 대화 이력 등)를 참고하여 고수준 전략을 세우고, 이를 세분화해 **각 단기 목표 단위의 작업(steps)으로 분할**한다.
* **입력:** 사용자 요청, state(브라우저 snapshot/요약, action 결과, 실행 옵션 등)
* **출력:** 다음에 실행할 **단기 목표 기반 step**(예: "검색창에 '김치찌개' 입력 후 검색 결과 보기", "이 페이지에서 김치찌개 재료를 모두 장바구니에 담기" 등) 리스트, 완료 여부, observation
* **특징:**
  * LLM을 활용해 동적으로 전체 목표를 **단기 목표(Short-term Goal)** 단위로 분해
  * 각 step은 Navigator가 자체적으로 여러 하위 액션(클릭, 입력 등)을 조합/추론하여 완수할 수 있는 범위의 목표여야 함
  * 계획 실패/예외 발생 시 state 기반으로 재계획 및 분기 처리
  * 브라우저 상태(state에 저장된 DOM/요약/스크린샷 등) 활용해 컨텍스트 기반 의사결정
  * **Navigator가 각 단기 목표 step을 실행한 최종 결과만 Planner가 전달받으며, Navigator의 내부 context에는 접근하지 않는다.**
  * **PlanStep** 모델을 사용하여 각 step을 구조화하며, PlannerState는 `remaining_steps`, `completed_steps`로 step을 분리 관리

### 2.2 Navigator Agent

* **역할:** Planner가 넘긴 **단기 목표 step**을 받아서, Playwright tool과 LLM 추론력을 활용해 실제 브라우저 조작을 수행하고 결과를 state에 기록한다.
* **입력:** 단기 목표 step(예: "이 페이지에서 n개 상품 장바구니 담기" 등), state(현재 브라우저 컨텍스트, 실행 옵션 등)
* **출력:** action 결과(ActionResult), 브라우저 상태 snapshot, action 이력
* **특징:**
  * Navigator는 **사용자의 최종 목표/의도는 모른다. Planner가 넘겨주는 단기 목표 step만 받아, 그 하위 세부 액션(클릭, 입력, 대기 등)은 LLM 스스로 계획/실행**
  * **Navigator는 각 단기 목표(step) 실행 동안에만 자체 context(내부 상태, 임시 데이터, 실행 log 등)를 유지하며, 해당 step이 종료되고 Planner에게 결과를 반환하면 Navigator의 내부 context는 즉시 초기화(삭제)한다.**
  * Playwright tool과 LLM 추론으로 여러 하위 액션을 조합해 목표 달성
  * **sub_plan**: 각 단기 목표를 달성하기 위한 세부 작업 시퀀스를 LLM/규칙 기반으로 미리 생성하여 순차적으로 실행
  * **action_history**: Playwright MCP Tool 기반의 구조화된 액션 이력(`ActionHistory` 모델)로 관리
  * **snapshot_history**: step 실행 중 주요 시점의 브라우저 상태(`BrowserSnapshot`)를 기록 가능
  * 예상과 다른 상황(예외/실패) 발생 시 상세 action 결과, 오류 메시지, 현재 브라우저 상태를 state에 기록
  * **Navigator가 MCP 서버에 연결되면 사용 가능한 도구 목록(tool_list)을 조회하여 공유 state에 기록**

### 2.3 Validator Agent

* **역할:** action 결과 및 전체 state가 사용자 목표와 계획에 부합하는지 검증, 실패 시 Planner에 피드백
* **입력:** action 결과, state, 계획 정보
* **출력:** 유효성 판단(is_valid), 실패/성공 이유(reason), 최종 답변(answer)
* **특징:**
  * LLM 기반 평가 및 복구 흐름 제어
  * 실패 시, 상세 원인과 현재 브라우저 상태를 Planner에게 전달
  * 최종 성공 시, 최종 결과(예: 장바구니 목록 등) 반환

---

## 3. state 구조 및 정보 흐름 (2024-06 최신)

### 3.1 구조화된 모델
- **PlanStep**: 단기 목표 step의 구조(설명, 완료여부, 오류, 실행 이력 등)
- **ActionHistory**: Playwright tool 기반 액션 이력(도구명, 파라미터, 실행시각, 결과 등)
- **BrowserSnapshot**: 브라우저 상태(요약, URL, 스크린샷 등)

### 3.2 각 agent의 state
- **PlannerState**: goal, remaining_steps(List[PlanStep]), completed_steps(List[PlanStep])
- **NavigatorState**: description(단기 목표), sub_plan(List[str]), action_history(List[ActionHistory]), snapshot_history(List[BrowserSnapshot])
- **ValidatorState**: ruleset, violations 등 검증 관련 정보

### 3.3 공유 state(BrowserUseAgentState)
- **current_plan_step**: 현재 진행 중인 PlanStep(단일 step만 공유)
- **tool_list**: 현재 세션에서 사용 가능한 Playwright MCP 도구 목록(동적으로 Navigator가 갱신)
- **snapshot**: 최신 브라우저 스냅샷
- **각 agent의 private state**(planner, navigator, validator)는 공유 state에 포함되지만, agent 간 직접 참조는 금지(공유 필드만 사용)

### 3.4 정보 흐름
1. 유저 요청/목표, 초기 state 저장
2. **Planner가 remaining_steps에서 단기 목표 기반 step(plan) 생성 및 current_plan_step에 기록**
3. **Navigator는 current_plan_step만 받아, sub_plan(세부계획)을 생성하고, 각 세부 작업을 Playwright tool로 실행하며 action_history에 기록**
4. Validator가 action 결과 검증 및 state 갱신, 실패/성공 판단
5. 실패 시 Planner가 state의 오류정보/브라우저 상태를 참고해 새로운 계획을 동적으로 수립
6. **tool_list는 Navigator가 MCP 서버에서 조회하여 공유 state에 갱신, Planner/Validator 등도 참조 가능**

---

## 4. 예시 state 구조 (pydantic 기반)

```python
class PlanStep(BaseModel):
    id: int
    description: str
    is_done: bool = False
    error: Optional[str] = None

class ActionHistory(BaseModel):
    tool_name: str
    params: Dict[str, Any]
    timestamp: datetime.datetime
    result: Optional[str] = None

class BrowserSnapshot(BaseModel):
    dom: Optional[str]
    url: Optional[str]
    screenshot_path: Optional[str]
    # ...

class PlannerState(BaseModel):
    goal: Optional[str]
    completed_steps: List[PlanStep]
    remaining_steps: List[PlanStep]

class NavigatorState(BaseModel):
    description: str
    sub_plan: List[str]
    action_history: List[ActionHistory]
    snapshot_history: List[BrowserSnapshot]

class ValidatorState(BaseModel):
    ruleset: Dict[str, Any]
    violations: List[str]

class BrowserUseAgentState(BaseModel):
    planner: PlannerState
    navigator: NavigatorState
    validator: ValidatorState
    current_plan_step: Optional[PlanStep]
    snapshot: Optional[BrowserSnapshot]
    tool_list: List[str]
```

---

## 5. Best Practice 및 확장 방향

- **공유 state에 꼭 필요한 정보만 명시적으로 노출** (current_plan_step, tool_list 등)
- **각 agent는 자신의 private state만 직접 수정, 타 agent의 private state는 공유 필드로만 접근**
- **Navigator의 sub_plan, action_history 등은 step 단위 reasoning/traceability에 매우 유용**
- **tool_list 공유로 Planner/Validator도 실제 사용 가능한 도구만 고려**
- **실패/예외/분기 처리, 동적 플랜 수정, 리플레이 등에 구조화된 이력 적극 활용**

---

## 6. 결론

- 본 설계는 LangGraph 기반 multi-agent 구조와 Playwright 기반 브라우저 제어의 Best Practice를 결합
- state 중심 정보흐름, 역할 분리, self-healing 루프, 이벤트 기반 제어 및 안전장치, 그리고 단기 목표 기반 granularity 설계로 실전 e-commerce 자동화/agent 서비스에 강건한 아키텍처 제공
- 최신 구조(pydantic 기반, 구조화된 이력, 동적 도구 공유 등)를 반영하여, 확장성과 유지보수성이 뛰어난 시스템을 지향함
