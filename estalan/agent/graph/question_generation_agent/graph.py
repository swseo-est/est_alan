from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from estalan.agent.base.node import create_initialization_node
from estalan.llm.utils import create_chat_model
from estalan.utils import get_last_human_message
from estalan.agent.graph.requirement_collection_agent.state import Question, RequirementCollectionAgentState, QuestionGenerationOutput


def create_generate_additional_questions_node(generate_question_llm):
    """2단계: AI를 통해 추가 질문 생성"""

    async def generate_additional_questions(state: RequirementCollectionAgentState):
        last_user_msg = get_last_human_message(state["messages"])
        purpose = state["purpose"]
        existing_questions = state.get("questions", [])
        existing_requirements = state.get("requirements", [])

        # LLM을 사용하여 추가 질문 생성
        # 기존 요구사항 정보를 포함한 컨텍스트 생성
        existing_requirements_text = ""
        if existing_requirements:
            existing_requirements_text = "이미 수집된 요구사항들:\n"
            for req in existing_requirements:
                existing_requirements_text += f"- {req['summary']}: {req['detail']}\n"

        prompt = f"""
        당신은 {purpose}를 위한 에이전트의 일부입니다.
        다음 목적을 위한 요구사항 수집을 위해 추가로 물어보면 좋을 질문을 3-5개 생성해주세요.

        목적: {last_user_msg}

        {existing_requirements_text}

        다음 조건을 만족하는 질문을 생성해주세요:
        1. 목적과 관련된 핵심 요구사항을 파악할 수 있는 질문
        2. 이미 수집된 요구사항과 중복되지 않는 새로운 관점의 질문
        3. 구체적이고 명확한 질문
        4. 아직 파악되지 않은 요구사항 영역에 대한 질문

        질문 생성 시 고려사항:
        - 이미 수집된 요구사항과 유사한 질문은 제외
        - 새로운 관점이나 세부사항을 파악할 수 있는 질문 우선
        - 사용자의 답변에서 추가 요구사항을 도출할 수 있는 질문
        """

        # LLM 호출을 통해 추가 질문 생성
        response = await generate_question_llm.ainvoke([
                SystemMessage(content="당신은 요구사항 수집 전문가입니다. 주어진 목적에 맞는 질문을 생성해주세요."),
                HumanMessage(content=prompt)
            ]
        )

        updated_questions = existing_questions + response["questions"]
        return {
            "questions": updated_questions,
        }

    return generate_additional_questions


def create_ask_question_node():
    """3단계: 질문을 사용자에게 전달"""

    def ask_question(state: RequirementCollectionAgentState):
        questions = state.get("questions", [])
        msg = ""
        for i, q in enumerate(questions):
            msg += f"{i+1}.{q['question']}\n"

        return {"messages": msg}

    return ask_question


def create_question_generation_agent(
        purpose=None,
        predefined_questions=[],
        generate_question_llm=None,
        name="question_generation_agent"):
    """요구사항 수집 에이전트 생성"""

    if purpose is None:
        raise Exception()

    if generate_question_llm is None:
        generate_question_llm = create_chat_model(provider="azure_openai", model="gpt-4.1", structured_output=QuestionGenerationOutput)

    # 노드 생성
    initialization_node = create_initialization_node(
        purpose=purpose,
        predefined_questions=predefined_questions
    )
    generate_additional_questions_node = create_generate_additional_questions_node(generate_question_llm)
    ask_question_node = create_ask_question_node()

    # 그래프 구성
    builder = StateGraph(RequirementCollectionAgentState)

    # 노드 추가
    builder.add_node("initialization_node", initialization_node)
    builder.add_node("generate_additional_questions", generate_additional_questions_node)
    builder.add_node("ask_question", ask_question_node)

    # 엣지 추가
    builder.add_edge(START, "initialization_node")
    builder.add_edge("initialization_node", "generate_additional_questions")
    builder.add_edge("generate_additional_questions", "ask_question")
    builder.add_edge("ask_question", END)

    question_generation_agent = builder.compile(name=name)
    return question_generation_agent

if __name__ == '__main__':
    import asyncio

    agent = create_question_generation_agent()
    result = asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="제주도 여행 계획 생성")]}))

    print(result)