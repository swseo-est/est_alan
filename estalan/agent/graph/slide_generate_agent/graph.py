from langgraph.prebuilt.chat_agent_executor import AgentState, AgentStateWithStructuredResponse
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph_supervisor import create_supervisor

from typing import List, TypedDict, Annotated
from estalan.agent.graph.requirement_collection_agent import create_requirement_collection_agent
from estalan.agent.graph.requirement_collection_agent.state import RequirementCollectionAgentPrivateState
from estalan.agent.graph.slide_generate_agent.planning_agent import create_planning_agent
from estalan.llm.utils import create_chat_model
from estalan.agent.base.state import private_state_updater


class SlideGenerateAgentState(AgentStateWithStructuredResponse):
    """슬라이드 생성 에이전트 상태 - 요구사항 수집 후 계획 수립"""
    
    # requirement_collection_agent state
    requirement_collection_agent_state: Annotated[RequirementCollectionAgentPrivateState, private_state_updater]
    
    # planning_agent state (직접 사용, private으로 감싸지 않음)
    topic: str  
    num_sections: int
    sections: List[str]


def create_slide_generate_agent(
        purpose="슬라이드 생성",
        predefined_questions=[],
        name="slide_generate_agent",
        state_schema=SlideGenerateAgentState,
        private_state_key="requirement_collection_agent_state"
        ):
    """슬라이드 생성 에이전트 생성"""
    
    # 요구사항 수집 에이전트 생성
    requirement_collection_agent = create_requirement_collection_agent(
        purpose=purpose, 
        predefined_questions=predefined_questions, 
        name="requirement_collection_agent",
        state_schema=state_schema,
        private_state_key=private_state_key
    )
    
    # 계획 수립 에이전트 생성
    planning_agent = create_planning_agent(name="planning_agent")

    # langgraph_supervisor를 사용한 supervisor 생성
    supervisor_prompt = f"""
    당신은 {purpose}를 위한 슬라이드 생성 워크플로우의 supervisor입니다. 
    당신의 역할은 오직 적절한 에이전트를 선택하여 라우팅하는 것입니다.
    ** 최대한 사용자 메시지를 하위 에이전트로 라우팅하는 것이 중요합니다.**

    ## 에이전트 역할

    1. **requirement_collection_agent**: 슬라이드 생성에 필요한 요구사항 수집
    - 슬라이드 주제, 섹션 수, 내용 등 요구사항 파악
    - 사용자와의 대화를 통한 요구사항 명확화
    - 요구사항 수집이 완료될 때까지 질문 생성

    2. **planning_agent**: 슬라이드 구조 및 계획 수립
    - 수집된 요구사항을 바탕으로 슬라이드 구조 설계
    - 섹션별 내용 계획 및 아웃라인 작성
    - 슬라이드 제작을 위한 상세 계획 수립


    ## 호출 순서
    - 제일 처음음에는 requirement_collection_agent를 호출하세요
        - requirement_collection_agent 다음은 END를 호출하고 유저 답변을 기다리세요

    - requirement_collection_agent가 요구사항 수집을 완료한 후 planning_agent 호출
        - planning_agent 다음으로는 END 호출

    ## 출력
    - last_step : END 전에 마지막으로 호출한 agent 이름
    """

    llm = create_chat_model(provider="azure_openai", model="gpt-4.1")

    # langgraph_supervisor를 사용하여 supervisor 생성
    supervisor_agent = create_supervisor(
        agents=[requirement_collection_agent, planning_agent],
        model=llm,
        prompt=supervisor_prompt,
        state_schema=SlideGenerateAgentState,
        output_mode="full_history"
    ).compile()

    # 그래프 구성
    builder = StateGraph(SlideGenerateAgentState)

    # 노드 추가
    builder.add_node("supervisor_agent", supervisor_agent)

    # 엣지 추가
    builder.add_edge(START, "supervisor_agent")
    builder.add_edge("supervisor_agent", END)

    # 그래프 컴파일
    slide_generate_agent = builder.compile(name=name)
    
    return slide_generate_agent


def create_graph():
    """슬라이드 생성 에이전트 그래프 생성 - supervisor를 통한 유동적 에이전트 호출"""
    return create_slide_generate_agent()


if __name__ == '__main__':
    import asyncio

    agent = create_graph()
    
    # 사용자 메시지
    user_message = "제주도 여행에 대한 슬라이드를 만들어주세요"
    
    result = asyncio.run(agent.ainvoke({
        "messages": [HumanMessage(content=user_message)],
        "requirement_collection_agent_state": {
            "requirement_collection_agent_private_state": {
                "purpose": "슬라이드 생성",
                "requirements": "## 요구사항\n\n수집된 요구사항이 없습니다.",
                "questions": [],
                "new_questions": [],
                "needs_more_questions": True,
                "initialization": False
            }
        }
    }))

    print("=== 테스트 결과 ===")
    print("Final state:")
    print(result)
    
    # requirement_collection_agent_state 확인
    if "requirement_collection_agent_state" in result:
        print("\n=== requirement_collection_agent_state 내용 ===")
        req_state = result["requirement_collection_agent_state"]
        if "requirement_collection_agent_private_state" in req_state:
            private_state = req_state["requirement_collection_agent_private_state"]
            print(f"purpose: {private_state.get('purpose')}")
            print(f"requirements: {private_state.get('requirements')}")
            print(f"questions: {private_state.get('questions')}")
            print(f"needs_more_questions: {private_state.get('needs_more_questions')}")
            print(f"initialization: {private_state.get('initialization')}")
        else:
            print("requirement_collection_agent_state가 없습니다!")
    else:
        print("requirement_collection_agent_state가 없습니다!")
    
    # planning_agent 결과 확인 (직접 필드들)
    print("\n=== planning_agent 결과 내용 ===")
    print(f"topic: {result.get('topic')}")
    print(f"num_sections: {result.get('num_sections')}")
    print(f"sections: {result.get('sections')}")