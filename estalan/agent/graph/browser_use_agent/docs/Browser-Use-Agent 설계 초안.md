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

### 2.2 Navigator Agent

* **역할:** Planner가 넘긴 **단기 목표 step**을 받아서, Playwright tool과 LLM 추론력을 활용해 실제 브라우저 조작을 수행하고 결과를 state에 기록한다.
* **입력:** 단기 목표 step(예: "이 페이지에서 n개 상품 장바구니 담기" 등), state(현재 브라우저 컨텍스트, 실행 옵션 등)
* **출력:** action 결과(ActionResult), 브라우저 상태 snapshot
* **특징:**

  * Navigator는 **사용자의 최종 목표/의도는 모른다. Planner가 넘겨주는 단기 목표 step만 받아, 그 하위 세부 액션(클릭, 입력, 대기 등)은 LLM 스스로 계획/실행**
  * **Navigator는 각 단기 목표(step) 실행 동안에만 자체 context(내부 상태, 임시 데이터, 실행 log 등)를 유지하며, 해당 step이 종료되고 Planner에게 결과를 반환하면 Navigator의 내부 context는 즉시 초기화(삭제)한다.**
  * Playwright tool과 LLM 추론으로 여러 하위 액션을 조합해 목표 달성
  * step별 실행 전/후 브라우저 상태 snapshot, DOM/주요 요소 요약 등 state에 기록
  * 예상과 다른 상황(예외/실패) 발생 시 상세 action 결과, 오류 메시지, 현재 브라우저 상태를 state에 기록

### 2.3 Validator Agent

* **역할:** action 결과 및 전체 state가 사용자 목표와 계획에 부합하는지 검증, 실패 시 Planner에 피드백
* **입력:** action 결과, state, 계획 정보
* **출력:** 유효성 판단(is\_valid), 실패/성공 이유(reason), 최종 답변(answer)
* **특징:**

  * LLM 기반 평가 및 복구 흐름 제어
  * 실패 시, 상세 원인과 현재 브라우저 상태를 Planner에게 전달
  * 최종 성공 시, 최종 결과(예: 장바구니 목록 등) 반환

---

## 3. 단위 작업(단기 목표 기반 step) granularity 설계 원칙

* **단위 작업(step)은 “액션” 단위가 아니라, 해당 UI/페이지/상황에서 LLM Navigator가 스스로 sub-action을 설계·수행할 수 있는 “단기 목표(Short-term Goal)” 단위로 분할**
* 예시:

  * “검색창에 ‘김치찌개’를 입력하고 검색 결과를 보이게 해줘”
  * “이 페이지에서 김치찌개 재료 상품 n개를 장바구니에 모두 담아줘”
  * “장바구니 페이지에서 담긴 상품 리스트를 모두 읽어와줘”
* **Navigator는 해당 단기 목표 step 내에서 클릭/입력/대기 등 세부 작업을 자체적으로 시퀀싱/추론/오류 복구**
* 너무 큰 범위(여러 페이지를 오가는 등)는 Planner가 다시 쪼개고, 너무 작은 액션(단순 입력 등)은 여러 개 묶어도 무방

---

## 4. 시스템 Graph/Flow 구조

```mermaid
graph TD;
    Start((Start)) --> Planner
    Planner -->|단기 목표 step 생성| Navigator
    Navigator -->|Action 결과| Validator
    Validator --|is_valid: false| Planner
    Validator --|is_valid: true| End((End))
    Planner --|done: true, web_task: false| End
```

* **loop 구조:** Planner → Navigator → Validator → (실패시) Planner 반복
* **state는 모든 agent가 공유** (입출력으로 state 사용)
* **예외/실패 발생 시 self-healing 루프**
* **메인 루프 종료 조건:** 최대 스텝 수 도달, 명시적 중단, 최대 연속 실패, Planner의 완료 플래그 등 운영 옵션 기반 안전장치 내장

---

## 5. state 구조 및 정보 흐름

* **공유 state 주요 필드:**

  * `taskId`, `messageManager`(대화 이력), `eventManager`(이벤트 관리), `browserContext`(Playwright context), `actionResults`(최근/전체 액션 결과), `browserSnapshot`(DOM/스크린샷/요약 등), 실행 옵션(`maxSteps`, `planningInterval`, `useVision` 등), 연속 실패 횟수(`consecutiveFailures`), `stopped`, `paused` 플래그 등 기타 상태 플래그
* **정보 흐름:**

  1. 유저 요청/목표, 초기 state 저장
  2. **Planner가 단기 목표 기반 step(plan) 생성 및 state에 기록**
  3. **Navigator는 step별 단기 목표만 받아 해당 step 실행 동안만 내부 context를 유지, 자체적으로 sub-action을 시퀀싱/수행 후 결과만 Planner에 리턴함. 브라우저 상태 snapshot 및 action 결과를 state에 기록**
  4. Validator가 action 결과 검증 및 state 갱신, 실패/성공 판단
  5. 실패 시 Planner가 state 기반으로 동적 재계획

---

## 6. 대표적 시나리오: "쿠팡에서 김치찌개 재료 검색 및 장바구니 담기"

1. 유저 요청: "쿠팡에서 김치찌개 재료 검색 후 장바구니 담아줘"
2. **Planner: 전체 목표를 “단기 목표 step”으로 분해**

   * step1: "검색창에 김치찌개 입력 후 검색 결과 보기"
   * step2: "검색 결과에서 김치찌개 재료 n개를 모두 장바구니에 담기"
3. Navigator: 각 단기 목표 step만 받아서, 그 하위 액션(입력/클릭/대기 등)을 스스로 판단해 playwright tool로 실행. step 실행 동안 내부 context 사용, 끝나면 초기화. 브라우저 상태/DOM/요약을 state에 저장
4. Planner: 최신 브라우저 상태(state) 참고해 다음 step을 새롭게 분해/수정 가능
5. Validator: 모든 재료가 정상적으로 담겼는지 최종 검증, 실패시 Planner에 피드백(예: 상품 미존재, 버튼 없음 등)
6. Planner: 실패 원인 분석, 상황별 동적 계획(재검색, 로그인 등) 추가
7. 모든 재료 성공 시 최종 완료

---

## 7. 예외/실패 처리 (self-healing loop 및 무한루프 방지)

* Navigator가 예상과 다른 상황(버튼 없음, 페이지 구조 변경 등)을 만나면,

  * 실행 실패/오류 정보를 state에 상세 기록
* Validator가 이를 감지, 실패(is\_valid: false) 및 원인 reason을 Planner로 피드백
* Planner가 state의 오류정보/브라우저 상태를 참고해 새로운 계획을 동적으로 수립(예: 재검색, 키워드 변경, 로그인 시도 등)
* 위 루프가 성공할 때까지 반복
* **운영 옵션(최대 스텝 수 `maxSteps`, 최대 연속 실패 `maxFailures`)을 활용해 무한 반복/비정상 루프를 방지하며, 조건 도달 시 안전하게 종료**

---

## 8. 이벤트 기반 아키텍처

* **EventManager**를 통해 각 단계의 주요 상태 변화(TASK\_START, STEP\_OK, ACT\_FAIL 등)가 이벤트로 발행됨
* 외부에서는 이벤트 구독(Subscribe) 방식으로 실시간 상태 변화 감지, UI 업데이트, 로깅, 외부 연동 등에 활용 가능
* 이벤트에는 주체(Actor: SYSTEM, PLANNER, NAVIGATOR, VALIDATOR), 상태, 관련 데이터 등이 포함

---

## 9. 확장 및 Best Practice

* Navigator만 playwright tool의 조작 권한(쓰기) 보유, 모든 agent는 state를 통해 브라우저 snapshot/요약 정보(읽기) 활용
* 필요시 agent 추가(예: Extractor, Summarizer, Vision 등)
* Planner/Validator 프롬프트 설계시 state 내 DOM/스크린샷 요약 정보 적극 활용
* 모든 예외 상황을 action 결과 및 state에 투명하게 기록해, 진단/회복력 강화
* **단기 목표 단위 step 분할을 통해 Planner는 전체 목표와 시나리오의 큰 흐름, Navigator는 실제 브라우저 조작 및 예외 복구에 집중**

---

## 10. 결론

* 본 설계는 LangGraph 기반 multi-agent 구조와 Playwright 기반 브라우저 제어의 Best Practice를 결합
* state 중심 정보흐름, 역할 분리, self-healing 루프, 이벤트 기반 제어 및 안전장치, 그리고 단기 목표 기반 granularity 설계로 실전 e-commerce 자동화/agent 서비스에 강건한 아키텍처 제공
