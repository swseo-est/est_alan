from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, BaseMessage

from typing import List, Annotated, TypedDict, Sequence
from estalan.agent.graph.slide_generate_agent.planning_agent import create_planning_agent
from estalan.agent.graph.slide_generate_agent.research_agent import create_research_agent
from estalan.agent.graph.slide_generate_agent.slide_design_agent import create_slide_create_agent
from estalan.agent.graph.slide_generate_agent.state import ExecutorState, Section, SlideGenerateAgentState
from estalan.agent.base.node import create_alan_agent_start_node, alan_agent_finish_node
from estalan.agent.base.state import BaseAlanAgentState

from estalan.prebuilt.requirement_analysis_agent import create_requirement_analysis_agent

from estalan.llm.utils import create_chat_model
from estalan.messages.utils import create_ai_message

import asyncio
from langgraph.types import Send
from langgraph_supervisor import create_supervisor


class OutputState(TypedDict):
    topic: str
    requirements: str
    template_folder: str

LIST_TEMPLATE_FOLDER = {
    "general": "일반적인 주제에 대해 사용",
    # "travel": "여행과 관련된 주제에 사용",
    # "compare": "비교와 관련된 주제에 사용",
    # "education": "교육과 관련된 주제에 사용",
    # "data_analysis": "데이터 분석과 관련된 주제에 사용"
}


def msg_test_node(state):
    print(state)
    return {}


def preprocessing_node(state):
    print(state)
    llm = create_chat_model(provider="google_vertexai", model="gemini-2.5-flash").with_structured_output(OutputState)

    list_tempalte_folder = ""
    for key in LIST_TEMPLATE_FOLDER.keys():
        list_tempalte_folder += f"{key}: {LIST_TEMPLATE_FOLDER[key]}\n"

    msg = f"""
    슬라이드 topic과 유저 요구사항 requirement 그리고 template_folder를 추출하세요. topic을 한글로 추출하세요.
    
    template_folder는 아래 중 하나를 선택하세요
    {list_tempalte_folder}
    
    Output
        topic: str, 유저 메시지로 부터 추출한 슬라이드의 주제, ex) 제주도 여행
        requirements: str, 유저 메시지로 부터 추출한 유저의 요구사항, ex) 3박 4일 제주도 여행 일정
        template_folder: str, topic에 적합한 template 폴더 이름, ex) general

    """
    msg = HumanMessage(content=msg)

    num_retry = 10
    for i in range(num_retry):
        try:
            updated_state = llm.invoke([msg] + state["messages"])

            node_message = create_ai_message(content=f"{updated_state['topic']}을 주제로 슬라이드를 생성하도록 하겠습니다.",
                                             name="msg_planning_start")

            metadata = {
                "topic": updated_state["topic"],
                "requirements": updated_state["requirements"],
                "template_folder": updated_state["template_folder"],
                "num_sections": 5,
                "num_slides": 7,
                "status": "start"
            }
            print(metadata)
            break
        except Exception as e:
            print(e)

    return {"metadata": metadata, "messages": [node_message]}


class ExecutorOutput(TypedDict):
    # executor에서 출력되는 결과
    slides: List[Section]
    messages: Sequence[BaseMessage]


def post_processing_executor_node(state):
    # print("Post processing state:", state)

    # state에서 messages를 제거하고 반환
    state_without_messages = {k: v for k, v in state.items() if k != 'messages'}

    # executor의 output이 ExecutorOutput 형태이므로 slides 필드를 그대로 반환
    return {"slides": [state_without_messages]}


def post_processing_node(state):
    msg = create_ai_message(content="슬라이드 생성이 완료되었습니다.", name="msg_slide_generation_finish")
    print(msg)
    metadata = state["metadata"].copy()
    metadata["status"] = "finish"
    return {"messages": [msg], "metadata": metadata}


def create_slide_generate_graph(name="slide_generate_agent"):
    """슬라이드 생성 메인 그래프"""
    planning_agent = create_planning_agent(name="planning_agent")

    ## subroutine
    executor = StateGraph(ExecutorState, output_schema=ExecutorOutput)

    research_agent = create_research_agent()
    slide_create_agent = create_slide_create_agent()

    executor.add_node("research_agent", research_agent)
    executor.add_node("slide_create_agent", slide_create_agent)
    executor.add_node("post_processing_executor_node", post_processing_executor_node)

    executor.add_edge(START, "research_agent")
    executor.add_edge("research_agent", "slide_create_agent")
    executor.add_edge("slide_create_agent", "post_processing_executor_node")
    executor.add_edge("post_processing_executor_node", END)

    # main graph
    builder = StateGraph(SlideGenerateAgentState)
    builder.add_node("preprocessing_node", preprocessing_node)
    builder.add_node("planning_agent", planning_agent)
    builder.add_node("executor", executor.compile(name="executor"))
    builder.add_node("post_processing_node", post_processing_node)

    builder.add_edge(START, "preprocessing_node")
    builder.add_edge("preprocessing_node", "planning_agent")

    def generate_slide(state):
        return [
            Send(
                "executor",
                s  | {"template_folder": state["metadata"]["template_folder"]}
            ) for s in state["sections"]]

    builder.add_conditional_edges(
        "planning_agent",
        generate_slide,
        ["executor"]
    )
    builder.add_edge("executor", "post_processing_node")
    builder.add_edge("post_processing_node", END)

    return builder.compile(name=name)


def create_graph():
    # 슬라이드 생성 그래프 생성
    slide_generate_graph = create_slide_generate_graph()
    # msg_test_node와 slide_generate_graph를 연결하는 테스트 그래프 생성
    requirement_analysis_agent = create_requirement_analysis_agent()

    # # # Supervisor 생성
    workflow = create_supervisor(
        [requirement_analysis_agent, slide_generate_graph],
        model=create_chat_model(provider="azure_openai", model="gpt-4.1"),
        prompt= """
                사용자와 대화를 통해 슬라이드 생성에 필요한 정보들을 수집하세요.
                
                requirement_analysis_agent를 이용해서, 사용자 대화에서 요구사항을 추가/업데이트/삭제 등 관리하세요.
                요구사항과 관련된 모든 작업은 requirement_analysis_agent에게 위임하세요.
                
                수집된 요구사항이 충분하지 않다면 사용자에게 질문을 통해, 요구사항을 더 수집하세요.
    
                충분한 정보가 모이면 slide_generate_agent를 이용하여, 슬라이드를 생성하세요.
                
                마지막 메시지를 통해 다음 에이전트에 충분한 정보를 전달하세요.
                다음 에이전트는 마지막 메시지만을 참조합니다.
            """
        ,
        state_schema=SlideGenerateAgentState,
        output_mode="full_history",
    ).compile()


    builder = StateGraph(SlideGenerateAgentState)

    alan_agent_start_node = create_alan_agent_start_node([BaseAlanAgentState, SlideGenerateAgentState])

    # 노드 추가
    builder.add_node("start_node", alan_agent_start_node)
    builder.add_node("test_node", msg_test_node)
    builder.add_node("agent", workflow)
    builder.add_node("finish_node", alan_agent_finish_node)

    # 엣지 연결
    builder.add_edge(START, "start_node")
    builder.add_edge("start_node", "test_node")
    builder.add_edge("test_node", "agent")
    builder.add_edge("agent", "finish_node")
    builder.add_edge("finish_node", END)

    # Compile and run
    app = builder.compile()
    return app



if __name__ == '__main__':
    import time

    s = time.time()

    graph = create_graph()
    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": ["이스트소프트에 대한 투자 IR 자료 만들어줘. 추가 질문은 하지말고 만들어"]
            }
        )
    )
    print(result)
    for state in result['slides']:
        with open(f"{state['idx']}.html", "w", encoding="utf-8") as f:
            f.write(state['html'])

    e = time.time()
    print(e - s)

    print(result)
