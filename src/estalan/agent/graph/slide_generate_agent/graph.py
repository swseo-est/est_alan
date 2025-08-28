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

import asyncio
from langgraph.types import Send
from estalan.prebuilt.supervisor import create_supervisor


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
    print("Supervisor 결과:", state)
    
    # supervisor의 결과에서 요구사항 정보 추출
    requirements_docs = state.get("requirements_docs", "")
    
    # 요구사항 정보를 포함하여 다음 단계로 전달
    return {
        "requirements_docs": requirements_docs
    }


def preprocessing_node(state):
    llm = create_chat_model(provider="azure_openai", model="gpt-5-mini").with_structured_output(OutputState)

    list_tempalte_folder = ""
    for key in LIST_TEMPLATE_FOLDER.keys():
        list_tempalte_folder += f"{key}: {LIST_TEMPLATE_FOLDER[key]}\n"

    topic = state["metadata"]["topic"]
    requirements_docs = state.get("requirements_docs", "")

    msg = f"""
    슬라이드 topic 주제와 유저 요구사항 requirement을 고려해서 template_folder를 추출하세요.
    
    # topic
    {topic}
    
    # requirements
    {requirements_docs}
    
    template_folder는 아래 중 하나를 선택하세요
    {list_tempalte_folder}
    
    Output
        template_folder: str, topic에 적합한 template 폴더 이름, ex) general
    """
    msg = HumanMessage(content=msg)

    num_retry = 10
    for i in range(num_retry):
        try:
            updated_state = llm.invoke([msg])

            node_message = create_ai_message(content=f"{topic}을 주제로 슬라이드를 생성하도록 하겠습니다.",
                                             name="msg_planning_start")

            metadata = state["metadata"].copy()
            metadata["template_folder"] = updated_state["template_folder"]
            break
        except Exception as e:
            print(e)

    return {
        "metadata": metadata, 
        "messages": [node_message], 
    }


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
    msg = create_ai_message(content="슬라이드 생성이 완료되었습니다.", name="end_msg")
    print(msg)
    metadata = state["metadata"].copy()
    metadata["status"] = "finish"
    return {"messages": [msg], "metadata": metadata}


def create_slide_generate_agent(name="slide_generate_agent"):
    """슬라이드 생성 메인 그래프"""
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
    builder.add_node("executor", executor.compile(name="executor"))
    builder.add_node("post_processing_node", post_processing_node)

    builder.add_edge(START, "preprocessing_node")

    def generate_slide(state):
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

    return builder.compile(name=name)


def create_graph(in_memory=False):
    requirement_analysis_agent = create_requirement_analysis_agent()
    planning_agent = create_planning_agent(name="planning_agent")
    slide_generate_graph = create_slide_generate_agent(name="slide_generate_agent")

    # Supervisor 생성
    workflow = create_supervisor(
        [requirement_analysis_agent, planning_agent, slide_generate_graph],
        model=create_chat_model(provider="azure_openai", model="gpt-5-mini"),
        prompt=prompt_supervisor,
        state_schema=SlideGenerateAgentState,
        output_mode="full_history",
        add_handoff_messages=True,
        add_handoff_back_messages=True
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

    if in_memory:
        checkpointer = InMemorySaver()
    else:
        checkpointer = None

    # Compile and run
    app = builder.compile(checkpointer=checkpointer)
    return app


async def run_agent(list_user_inputs):
    graph = create_graph(in_memory=True)

    for msg in list_user_inputs:
        print("user input : ", msg)
        result = await graph.ainvoke(
                {"messages": [msg]},
                {"configurable": {"thread_id": "1"}}
            )
        print("updated state: ", result)
        # for msg in result["messages"]:
        #     print(msg.content)

    return result


if __name__ == '__main__':
    import time
    from estalan.agent.graph.slide_generate_agent.tests.inputs import initial_msg

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

    result = asyncio.run(run_agent(list_user_inputs))

    for state in result['slides']:
        with open(f"{state['idx']}.html", "w", encoding="utf-8") as f:
            f.write(state['html'])

    e = time.time()
    print(e - s)

    print(result)
