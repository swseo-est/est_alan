from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage

from estalan.llm.utils import create_chat_model
from estalan.agent.graph.requirement_collection_agent.state import RequirementCollectionAgentState, ExtractRequirementOutput
from estalan.agent.graph.requirement_collection_agent.prompt import SLIDE_GENERATION_TASK_PROMPT

def create_extract_requirements_node(extract_llm):
    """사용자 답변에서 요구사항 추출"""

    async def extract_requirements(state: RequirementCollectionAgentState):
        # 마지막 사용자 메시지에서 답변 추출
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            return {"requirements": state.get("requirements", [])}
        
        # 현재 질문 목록 가져오기
        questions = state.get("questions", [])
        
        # 기존 요구사항들
        existing_requirements = state.get("requirements", [])
        
        # LLM을 사용하여 요구사항 추출
        prompt = f"""
{SLIDE_GENERATION_TASK_PROMPT}

당신의 임무는 아래 사용자의 답변에서 요구사항들을 추출하는 것입니다.

질문: {questions}
사용자 답변: {user_message}

기존 요구사항들:
{chr(10).join([f"- {req['requirement_id']}: {req['summary']} - {req['detail']}" for req in existing_requirements]) if existing_requirements else "없음"}

다음 형식으로 요구사항을 추출해주세요:
- origin: 질문과 관련된 요구사항이면 'question', 관련없는 요구사항이면 'user'
- summary: 핵심 요구사항을 간단히 요약. 예) '예산: 100만원', '청중: 회사 동료'
- detail: 구체적인 요구사항 내용
- update_existing: 기존 요구사항을 수정해야하는 경우 해당 요구사항의 requirement_id, 새로운 요구사항이면 null

요구사항 업데이트 판단 기준:
1. 기존 요구사항과 동일한 주제나 내용
2. 기존 요구사항의 세부사항을 보완하거나 수정하는 내용
3. 기존 요구사항과 연관된 추가 정보

질문과 관련된 요구사항 판단 기준:
1. 질문에서 묻는 내용에 대한 직접적인 답변
2. 질문의 맥락과 관련된 추가 정보
3. 질문에서 파악하려는 요구사항과 일치하는 내용
4. 사용자 답변에서 명시적/암시적으로 파악할 수 있는 내용만 추출

질문과 관련없는 요구사항:
1. 질문과 전혀 다른 주제의 요구사항
2. 질문의 맥락을 벗어난 새로운 요구사항
3. 사용자가 자발적으로 언급한 추가 요구사항
4. 사용자 답변에 언급되지 않은 내용
"""
        message = [
            SystemMessage(content="당신은 요구사항 분석 전문가입니다. 사용자의 답변에서 명시적/암묵적 요구사항을 추출하고, 질문과의 관련성을 판단하며, 기존 요구사항과의 연관성을 파악해주세요."),
            HumanMessage(content=prompt)
        ]
        response = await extract_llm.ainvoke(message)

        # 추출된 요구사항을 처리하여 기존 요구사항 업데이트 또는 새 요구사항 추가
        updated_requirements = existing_requirements.copy()
        
        for new_req in response["requirements"]:
            if new_req.get("update_existing"):
                # 기존 요구사항 업데이트
                existing_req_id = new_req["update_existing"]
                for i, req in enumerate(updated_requirements):
                    if req["requirement_id"] == existing_req_id:
                        # 기존 요구사항 업데이트
                        updated_requirements[i] = {
                            "requirement_id": req["requirement_id"],
                            "origin": new_req["origin"],
                            "summary": new_req["summary"],
                            "detail": new_req["detail"]
                        }
                        break
            else:
                # 새로운 요구사항 추가
                new_requirement = {
                    "requirement_id": f"req_{len(updated_requirements) + 1}",
                    "origin": new_req["origin"],
                    "summary": new_req["summary"],
                    "detail": new_req["detail"]
                }
                updated_requirements.append(new_requirement)

        return {
            "questions": [],
            "requirements": updated_requirements
        }

    return extract_requirements


def create_requirement_analysis_agent(name="requirement_analysis_agent"):
    """요구사항 분석 에이전트 생성"""

    # LLM 모델 생성
    extract_llm = create_chat_model(provider="azure_openai", model="gpt-4.1", structured_output=ExtractRequirementOutput)

    # 노드 생성
    extract_requirements_node = create_extract_requirements_node(extract_llm)

    # 그래프 구성
    builder = StateGraph(RequirementCollectionAgentState)

    # 노드 추가
    builder.add_node("extract_requirements", extract_requirements_node)

    # 엣지 추가
    builder.add_edge(START, "extract_requirements")
    builder.add_edge("extract_requirements", END)

    requirement_analysis_agent = builder.compile(name=name)
    return requirement_analysis_agent


if __name__ == '__main__':
    import asyncio

    async def requirement_analysis():
        agent = create_requirement_analysis_agent()

        print("=== 요구사항 분석 에이전트 테스트 ===\n")

        # 테스트 1: 초기 요구사항 수집
        print("1. 초기 요구사항 수집 테스트")
        initial_state = {
            "messages": [
                HumanMessage(content="제주도 여행을 계획하고 있어요. 예산은 100만원 정도이고, 3박 4일 정도로 계획하고 싶어요.")
            ],
            "questions": [
                {
                    "requirement_id": "req_1",
                    "question": "여행 비용은 어느정도 계획하고 계신가요?"
                }
            ],
            "requirements": [],
            "initialization": True,
            "is_complete": False
        }

        result1 = await agent.ainvoke(initial_state)
        print("초기 요구사항:")
        for req in result1["requirements"]:
            print(f"  - {req['requirement_id']}: {req['summary']} ({req['origin']})")
        print()

        # 테스트 2: 기존 요구사항 업데이트
        print("2. 기존 요구사항 업데이트 테스트")
        update_state = {
            "messages": [
                HumanMessage(content="예산을 120만원으로 늘리고 싶어요. 그리고 렌트카도 빌릴 예정이에요.")
            ],
            "questions": [
                {
                    "requirement_id": "req_1",
                    "question": "여행 비용은 어느정도 계획하고 계신가요?"
                }
            ],
            "requirements": result1["requirements"],
            "initialization": True,
            "is_complete": False
        }

        result2 = await agent.ainvoke(update_state)
        print("업데이트된 요구사항:")
        for req in result2["requirements"]:
            print(f"  - {req['requirement_id']}: {req['summary']} ({req['origin']})")
        print()

        # 테스트 3: 명시적 요구사항 테스트
        print("3. 명시적 요구사항 테스트")
        explicit_state = {
            "messages": [
                HumanMessage(content="숙박은 호텔로 하고 싶고, 항공편은 오전에 출발하는 걸로 해주세요.")
            ],
            "questions": [
                {
                    "requirement_id": "req_1",
                    "question": "어떤 종류의 숙박을 선호하시나요?"
                }
            ],
            "requirements": result2["requirements"],
            "initialization": True,
            "is_complete": False
        }

        result3 = await agent.ainvoke(explicit_state)
        print("명시적 요구사항:")
        for req in result3["requirements"]:
            print(f"  - {req['requirement_id']}: {req['summary']} ({req['origin']})")
        print()

        # 테스트 4: 암시적 요구사항 테스트
        print("4. 암시적 요구사항 테스트")
        implicit_state = {
            "messages": [
                HumanMessage(content="가족과 함께 가는데, 아이가 있어서 안전한 곳으로 가고 싶어요.")
            ],
            "questions": [
                {
                    "requirement_id": "req_1",
                    "question": "누구와 함께 여행하시나요?"
                }
            ],
            "requirements": result3["requirements"],
            "initialization": True,
            "is_complete": False
        }

        result4 = await agent.ainvoke(implicit_state)
        print("암시적 요구사항:")
        for req in result4["requirements"]:
            print(f"  - {req['requirement_id']}: {req['summary']} ({req['origin']})")
        print()

        # 테스트 5: 질문과 관련없는 요구사항 테스트
        print("5. 질문과 관련없는 요구사항 테스트")
        unrelated_state = {
            "messages": [
                HumanMessage(content="그리고 제주도에서 맛집도 많이 가보고 싶어요. 특히 해산물 맛집을 추천받고 싶어요.")
            ],
            "questions": [
                {
                    "requirement_id": "req_1",
                    "question": "어떤 종류의 숙박을 선호하시나요?"
                }
            ],
            "requirements": result4["requirements"],
            "initialization": True,
            "is_complete": False
        }

        result5 = await agent.ainvoke(unrelated_state)
        print("질문과 관련없는 요구사항:")
        for req in result5["requirements"]:
            print(f"  - {req['requirement_id']}: {req['summary']} ({req['origin']})")
        print()

        # 최종 요구사항 요약
        print("=== 최종 요구사항 요약 ===")
        for req in result5["requirements"]:
            print(f"• {req['requirement_id']}: {req['summary']}")
            print(f"  - {req['detail']}")
            print(f"  - Origin: {req['origin']}")
            print()

    # 테스트 실행
    asyncio.run(requirement_analysis())
