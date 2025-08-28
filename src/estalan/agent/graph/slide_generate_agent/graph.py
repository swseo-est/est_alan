from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from typing import List, Annotated, TypedDict, Sequence
from estalan.agent.graph.slide_generate_agent.planning_agent import create_planning_agent
from estalan.agent.graph.slide_generate_agent.research_agent import create_research_agent
from estalan.agent.graph.slide_generate_agent.slide_design_agent import create_slide_create_agent
from estalan.agent.graph.slide_generate_agent.state import ExecutorState, Section, SlideGenerateAgentState
from estalan.agent.graph.slide_generate_agent.prompt.supervisor import prompt_supervisor

from estalan.agent.base.node import create_alan_agent_start_node, alan_agent_finish_node
from estalan.agent.base.state import BaseAlanAgentState

from estalan.prebuilt.requirement_analysis_agent import create_requirement_analysis_agent, create_requirement_analysis_subagent

from estalan.llm.utils import create_chat_model
from estalan.messages.utils import create_ai_message
from estalan.logging.base import get_logger

import asyncio
from langgraph.types import Send
from estalan.prebuilt.supervisor import create_supervisor

# 로거 초기화
logger = get_logger(__name__)

class OutputState(TypedDict):
    template_folder: str

LIST_TEMPLATE_FOLDER = {
    "general": "일반적인 주제에 대해 사용",
    # "travel": "여행과 관련된 주제에 사용",
    # "compare": "비교와 관련된 주제에 사용",
    # "education": "교육과 관련된 주제에 사용",
    # "data_analysis": "데이터 분석과 관련된 주제에 사용"
}


def msg_test_node(state):
    logger.info("테스트 노드 실행 시작")
    
    # supervisor의 결과에서 요구사항 정보 추출
    requirements_docs = state.get("requirements_docs", "")
    logger.debug(f"요구사항 문서 길이: {len(requirements_docs)}자")
    
    # 요구사항 정보를 포함하여 다음 단계로 전달
    logger.info("테스트 노드 실행 완료")
    return {
        "requirements_docs": requirements_docs
    }


def preprocessing_node(state):
    logger.info("전처리 노드 실행 시작")
    
    llm = create_chat_model(provider="azure_openai", model="gpt-4o").with_structured_output(OutputState)

    list_tempalte_folder = ""
    for key in LIST_TEMPLATE_FOLDER.keys():
        list_tempalte_folder += f"{key}: {LIST_TEMPLATE_FOLDER[key]}\n"

    topic = state["metadata"]["topic"]
    requirements_docs = state.get("requirements_docs", "")
    
    logger.debug(f"전처리 파라미터: topic='{topic}', requirements_docs 길이={len(requirements_docs)}자")

    msg = f"""
    - 슬라이드 topic 주제와 유저 요구사항 requirement을 고려해서 template_folder를 추출하세요.
    - template_folder는 제시된 폴더 중에서만 선택하세요. ex) general
    
    # template_folder
    {list_tempalte_folder}

    # topic
    {topic}
    
    # requirements
    {requirements_docs}
    
    
    Output
        template_folder: str, topic에 적합한 template 폴더 이름, ex) general
    """
    msg = HumanMessage(content=msg)

    num_retry = 10
    logger.info(f"템플릿 폴더 선택 시작: 최대 {num_retry}번 시도")
    
    for i in range(num_retry):
        try:
            logger.debug(f"템플릿 폴더 선택 시도 {i+1}/{num_retry}")
            
            updated_state = llm.invoke([msg])
            template_folder = updated_state["template_folder"]
            
            logger.info(f"템플릿 폴더 선택 성공: '{template_folder}'")

            node_message = create_ai_message(content=f"{topic}을 주제로 슬라이드를 생성하도록 하겠습니다.",
                                             name="msg_planning_start")

            metadata = state["metadata"].copy()
            metadata["template_folder"] = template_folder
            break
            
        except Exception as e:
            logger.error(f"템플릿 폴더 선택 중 오류 발생 (시도 {i+1}/{num_retry}): {e}")
            if i == num_retry - 1:  # 마지막 시도에서도 실패
                logger.critical("템플릿 폴더 선택이 모든 시도 후에도 실패함")
                raise

    logger.info("전처리 노드 실행 완료")
    return {
        "metadata": metadata, 
        "messages": [node_message], 
    }


class ExecutorOutput(TypedDict):
    # executor에서 출력되는 결과
    slides: List[Section]
    messages: Sequence[BaseMessage]


def post_processing_executor_node(state):
    logger.info("실행기 후처리 노드 실행 시작")
    
    # state에서 messages를 제거하고 반환
    state_without_messages = {k: v for k, v in state.items() if k != 'messages'}
    
    logger.debug(f"메시지 제거 후 상태 키: {list(state_without_messages.keys())}")

    # executor의 output이 ExecutorOutput 형태이므로 slides 필드를 그대로 반환
    logger.info("실행기 후처리 노드 실행 완료")
    return {"slides": [state_without_messages]}


def post_processing_node(state):
    logger.info("최종 후처리 노드 실행 시작")
    
    msg = create_ai_message(content="슬라이드 생성이 완료되었습니다.", name="end_msg")
    logger.info("슬라이드 생성 완료 메시지 생성")
    
    metadata = state["metadata"].copy()
    metadata["status"] = "finish"
    
    logger.info("최종 후처리 노드 실행 완료")
    return {"messages": [msg], "metadata": metadata}


def create_slide_generate_agent(name="slide_generate_agent"):
    """슬라이드 생성 메인 그래프"""
    logger.info(f"슬라이드 생성 에이전트 생성 시작: name='{name}'")
    
    ## subroutine
    logger.debug("실행기 서브루틴 생성 시작")
    executor = StateGraph(ExecutorState, output_schema=ExecutorOutput)

    research_agent = create_research_agent()
    slide_create_agent = create_slide_create_agent()
    
    logger.debug("실행기 노드 추가")
    executor.add_node("research_agent", research_agent)
    executor.add_node("slide_create_agent", slide_create_agent)
    executor.add_node("post_processing_executor_node", post_processing_executor_node)

    executor.add_edge(START, "research_agent")
    executor.add_edge("research_agent", "slide_create_agent")
    executor.add_edge("slide_create_agent", "post_processing_executor_node")
    executor.add_edge("post_processing_executor_node", END)
    
    logger.debug("실행기 컴파일 완료")

    # main graph
    logger.debug("메인 그래프 빌더 생성")
    builder = StateGraph(SlideGenerateAgentState)
    builder.add_node("preprocessing_node", preprocessing_node)
    builder.add_node("executor", executor.compile(name="executor"))
    builder.add_node("post_processing_node", post_processing_node)

    builder.add_edge(START, "preprocessing_node")

    def generate_slide(state):
        logger.debug(f"슬라이드 생성 조건부 엣지 실행: {len(state['sections'])}개 섹션")
        return [
            Send(
                "executor",
                s  | {"template_folder": state["metadata"]["template_folder"]}
            ) for s in state["sections"]]

    builder.add_conditional_edges(
        "preprocessing_node",
        generate_slide,
        ["executor"]
    )
    builder.add_edge("executor", "post_processing_node")
    builder.add_edge("post_processing_node", END)

    slide_generate_agent = builder.compile(name=name)
    logger.info(f"슬라이드 생성 에이전트 생성 완료: name='{name}'")
    
    return slide_generate_agent


def create_graph(in_memory=False):
    logger.info("슬라이드 생성 그래프 생성 시작")
    
    requirement_analysis_agent = create_requirement_analysis_agent()
    planning_agent = create_planning_agent(name="planning_agent")
    slide_generate_graph = create_slide_generate_agent(name="slide_generate_agent")
    
    logger.debug("에이전트들 생성 완료")

    # Supervisor 생성
    logger.debug("Supervisor 생성 시작")
    workflow = create_supervisor(
        [requirement_analysis_agent, planning_agent, slide_generate_graph],
        model=create_chat_model(provider="azure_openai", model="gpt-4o"),
        prompt=prompt_supervisor,
        state_schema=SlideGenerateAgentState,
        output_mode="full_history",
        add_handoff_messages=True,
        add_handoff_back_messages=True
    ).compile()
    
    logger.debug("Supervisor 생성 완료")

    builder = StateGraph(SlideGenerateAgentState)

    alan_agent_start_node = create_alan_agent_start_node([BaseAlanAgentState, SlideGenerateAgentState])
    logger.debug("Alan 에이전트 시작 노드 생성 완료")

    # 노드 추가
    logger.debug("메인 그래프 노드 추가")
    builder.add_node("start_node", alan_agent_start_node)
    builder.add_node("test_node", msg_test_node)
    builder.add_node("agent", workflow)
    builder.add_node("finish_node", alan_agent_finish_node)

    # 엣지 연결
    logger.debug("메인 그래프 엣지 연결")
    builder.add_edge(START, "start_node")
    builder.add_edge("start_node", "test_node")
    builder.add_edge("test_node", "agent")
    builder.add_edge("agent", "finish_node")
    builder.add_edge("finish_node", END)

    if in_memory:
        checkpointer = InMemorySaver()
        logger.debug("메모리 체크포인터 설정")
    else:
        checkpointer = None
        logger.debug("체크포인터 없음")

    # Compile and run
    app = builder.compile(checkpointer=checkpointer)
    logger.info("슬라이드 생성 그래프 생성 완료")
    
    return app


async def run_agent(list_user_inputs):
    logger.info(f"에이전트 실행 시작: {len(list_user_inputs)}개 사용자 입력")
    
    graph = create_graph(in_memory=True)

    for i, msg in enumerate(list_user_inputs):
        logger.info(f"사용자 입력 {i+1}/{len(list_user_inputs)} 처리: {msg[:100]}...")
        
        result = await graph.ainvoke(
                {"messages": [msg]},
                {"configurable": {"thread_id": "1"}}
            )
        logger.info(f"사용자 입력 {i+1} 처리 완료")
        logger.debug(f"업데이트된 상태 키: {list(result.keys())}")

    logger.info("모든 사용자 입력 처리 완료")
    return result


if __name__ == '__main__':
    import time
    from estalan.agent.graph.slide_generate_agent.tests.inputs import initial_msg

    logger.info("메인 실행 시작")
    s = time.time()

    list_user_inputs = [initial_msg, "아니야 10일 일정으로 부탁해", "슬라이드 개수는 12장이 좋겠어", "종아 슬라이드를 생성해줘"]
    list_user_inputs = [initial_msg, "종아 슬라이드를 생성해줘", "종아 슬라이드를 생성해줘"]
    list_user_inputs = ["떡볶이 맛집 전국 투어 계획해줘", "목차 생성해줘", "종아 슬라이드를 생성해줘"]
    list_user_inputs = [
        "떡볶이 맛집 전국 투어 계획해줘",
        "서울을 추가 시켜줘",
        "슬라이드는 12장으로 해줘",
        "목차 생성해줘",
        "슬라이드를 5장으로 줄여줘",
        "슬라이드를 3장으로 줄여줘",
        "목차에 서울 맛집 섹션을 추가해줘",
        "종아 슬라이드를 생성해줘"
    ]

    logger.info(f"테스트 실행: {len(list_user_inputs)}개 입력")
    result = asyncio.run(run_agent(list_user_inputs))

    logger.info("HTML 파일 생성 시작")
    for state in result['slides']:
        filename = f"{state['idx']}.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(state['html'])
        logger.debug(f"HTML 파일 생성: {filename}")

    e = time.time()
    execution_time = e - s
    logger.info(f"전체 실행 시간: {execution_time:.2f}초")

    logger.info("메인 실행 완료")
    print(result)
