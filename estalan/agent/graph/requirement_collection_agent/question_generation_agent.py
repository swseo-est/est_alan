from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from estalan.agent.base.node import create_initialization_node
from estalan.llm.utils import create_chat_model
from estalan.utils import get_last_human_message
from estalan.agent.graph.requirement_collection_agent.state import Question, RequirementCollectionAgentState, QuestionGenerationOutput


def create_generate_additional_questions_node(generate_question_llm, private_state_key):
    """2단계: AI를 통해 추가 질문 생성"""

    async def generate_additional_questions(state: RequirementCollectionAgentState):
        last_user_msg = get_last_human_message(state["messages"])
        
        # requirement_collection_agent_private_state에서 필요한 정보 추출
        private_state = state.get(private_state_key, {})
        purpose = private_state.get("purpose", "")
        existing_questions = private_state.get("questions", [])
        existing_requirements = private_state.get("requirements", "")

        # 기존 질문들의 내용을 추출하여 중복 방지용으로 사용
        existing_question_texts = [q['question'].lower().strip() for q in existing_questions]
        
        # LLM을 사용하여 추가 질문 생성
        # 기존 요구사항 정보를 포함한 컨텍스트 생성
        existing_requirements_text = ""
        if existing_requirements and existing_requirements != "## 요구사항\n\n수집된 요구사항이 없습니다.":
            existing_requirements_text = "이미 수집된 요구사항들:\n" + existing_requirements

        # 기존 질문 정보를 포함한 컨텍스트 생성
        existing_questions_text = ""
        if existing_questions:
            existing_questions_text = "이미 생성된 질문들:\n"
            for i, q in enumerate(existing_questions):
                existing_questions_text += f"{i+1}. {q['question']}\n"

        prompt = f"""
        당신은 {purpose}를 위한 에이전트의 일부입니다.
        다음 목적을 위한 요구사항 수집을 위해 추가로 물어보면 좋을 질문을 3-5개 생성해주세요.

        유저 메시지: {last_user_msg}

        {existing_requirements_text}

        {existing_questions_text}

        다음 조건을 만족하는 질문을 생성해주세요:
        1. 목적과 관련된 핵심 요구사항을 파악할 수 있는 질문
        2. 이미 생성된 질문과 의미적으로 중복되지 않는 새로운 관점의 질문
        3. 구체적이고 명확한 질문
        4. 아직 파악되지 않은 요구사항 영역에 대한 질문

        **중요한 제약사항**:
        - 이미 생성된 질문과 동일하거나 유사한 의미의 질문은 절대 생성하지 마세요
        - 질문의 표현이 달라도 핵심 의도가 같다면 중복으로 간주합니다
        - 새로운 관점이나 세부사항을 파악할 수 있는 질문만 생성하세요
        - 사용자의 답변에서 추가 요구사항을 도출할 수 있는 질문을 우선적으로 생성하세요

        **중복 방지 예시**:
        - 기존: "언제 여행을 가고 싶으신가요?" → 새로 생성하면 안됨
        - 기존: "여행 기간은 얼마나 되나요?" → 새로 생성하면 안됨
        - 새로 생성 가능: "특별히 방문하고 싶은 명소가 있나요?" (새로운 관점)
        """

        # LLM 호출을 통해 추가 질문 생성
        response = await generate_question_llm.ainvoke(
            state["messages"] + 
            [
                SystemMessage(content="당신은 요구사항 수집 전문가입니다. 주어진 목적에 맞는 질문을 생성해주세요."),
                HumanMessage(content=prompt)
            ]
        )

        # 중복 검사 및 필터링
        new_questions = []
        for new_q in response["questions"]:
            new_question_text = new_q['question'].lower().strip()
            
            # 중복 검사
            is_duplicate = False
            for existing_text in existing_question_texts:
                # 간단한 유사도 검사 (키워드 기반)
                if any(keyword in new_question_text for keyword in existing_text.split() if len(keyword) > 3):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                new_questions.append(new_q)
                existing_question_texts.append(new_question_text)

        # 기존 질문과 새로운 질문을 합침
        updated_questions = existing_questions + new_questions
        
        return {
            private_state_key: {
                "questions": updated_questions,
                "new_questions": new_questions,  # 새로 생성된 질문들만 별도로 저장
            }
        }

    return generate_additional_questions


def create_print_question_node(private_state_key):
    """3단계: 질문을 사용자에게 전달"""

    def print_question_node(state: RequirementCollectionAgentState):
        # requirement_collection_agent_private_state에서 새로 생성된 질문들 추출
        private_state = state.get(private_state_key, {})
        new_questions = private_state.get("new_questions", [])
        
        if not new_questions:
            msg = "추가 질문이 없습니다. 요구사항 수집이 완료되었습니다."
        else:
            msg = "다음 질문에 답변해주세요:\n"
            for i, q in enumerate(new_questions):
                msg += f"{i+1}. {q['question']}\n"

        return {"messages": [AIMessage(content=msg)]}
    
    return print_question_node


def create_question_generation_agent(
        purpose=None,
        predefined_questions=[],
        generate_question_llm=None,
        name="question_generation_agent",
        private_state_key="requirement_collection_agent_state"):
    """요구사항 수집 에이전트 생성"""

    if generate_question_llm is None:
        generate_question_llm = create_chat_model(provider="azure_openai", model="gpt-4.1", structured_output=QuestionGenerationOutput)

    # 노드 생성
    initialization_node = create_initialization_node(
        purpose=purpose,
        predefined_questions=predefined_questions,
        private_state=private_state_key
    )
    generate_additional_questions_node = create_generate_additional_questions_node(generate_question_llm, private_state_key)
    print_question_node = create_print_question_node(private_state_key)

    # 그래프 구성
    builder = StateGraph(RequirementCollectionAgentState)

    # 노드 추가
    builder.add_node("initialization_node", initialization_node)
    builder.add_node("generate_additional_questions", generate_additional_questions_node)
    builder.add_node("print_question_node", print_question_node)

    # 엣지 추가
    builder.add_edge(START, "initialization_node")
    builder.add_edge("initialization_node", "generate_additional_questions")
    builder.add_edge("generate_additional_questions", "print_question_node")
    builder.add_edge("print_question_node", END)

    question_generation_agent = builder.compile(name=name)
    return question_generation_agent


if __name__ == '__main__':
    import asyncio

    agent = create_question_generation_agent()
    result = asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="제주도 여행 계획 생성")]}))

    print(result)