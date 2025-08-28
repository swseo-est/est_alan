import os
from typing import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.prebuilt import create_react_agent
from langgraph.graph import START, END, StateGraph

from estalan.agent.graph.slide_generate_agent.prompt.research_agent import *
from estalan.llm import create_chat_model
from estalan.tools.search import GoogleSerperSearchResult
from estalan.messages.utils import create_ai_message
from estalan.agent.graph.slide_generate_agent.state import ExecutorState
from estalan.logging.base import get_logger

# 로거 초기화
logger = get_logger(__name__)

class ResearchAgentState(ExecutorState):
    pass

class ResearchNodeOutput(TypedDict):
    research: bool
    content: str


def pre_processing_node(state):
    logger.info("연구 에이전트 전처리 노드 실행")
    return {}


def post_processing_node(state):
    logger.info("연구 에이전트 후처리 노드 실행")
    return {}


def pre_processing_research_node(state):
    logger.info("연구 노드 전처리 시작")
    
    content = f"""슬라이드 생성에 필요한 조사를 시작합니다.
"""

    msg = create_ai_message(
        content=content,
        name="msg_research_start",
        id="msg_research_start"
    )

    logger.info("연구 시작 메시지 생성 완료")
    return {"messages": [msg]}


def post_processing_research_node(state):
    logger.info("연구 노드 후처리 시작")
    
    name = state["name"]
    logger.debug(f"연구 완료 섹션: {name}")

    content = f"""{name} 페이지에 대한 조사를 완료하였습니다."""

    msg = create_ai_message(
        content=content,
        name="msg_research_end",
        id = "msg_research_end"
    )
    
    logger.info(f"연구 완료 메시지 생성: {name}")
    return {}


def create_research_node(llm):
    async def research_node(state: ResearchAgentState):
        logger.info("연구 노드 실행 시작")
        
        topic = state["topic"]
        name = state["name"]
        description = state["description"]
        
        logger.debug(f"연구 파라미터: topic='{topic}', name='{name}', description='{description[:100]}...'")

        section_writer_inputs_formatted = section_writer_inputs.format(topic=topic,
                                                                       section_name=name,
                                                                       section_topic=description,
                                                                       content="")
        
        logger.debug("섹션 작성자 입력 포맷팅 완료")
        
        # Format system instructions
        logger.info("LLM을 사용한 연구 시작")
        for i in range(10):
            try:
                logger.debug(f"연구 시도 {i+1}/10")
                
                results = await llm.ainvoke(
                    {
                        "messages":
                            [
                                SystemMessage(content=section_writer_instructions),
                                HumanMessage(content=section_writer_inputs_formatted),
                            ]
                    }
                )

                logger.info(f"연구 성공: {name} 섹션")
                logger.debug(f"연구 결과 키: {list(results['structured_response'].keys())}")
                
                return results['structured_response']
                
            except Exception as e:
                logger.error(f"연구 중 오류 발생 (시도 {i+1}/10): {e}")
                if i == 9:  # 마지막 시도에서도 실패
                    logger.critical(f"연구가 10번 시도 후에도 실패함: {name}")
                    raise

    return research_node


def create_research_agent(name=None):
    logger.info(f"연구 에이전트 생성 시작: name='{name}'")
    
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        logger.warning("SERPER_API_KEY 환경변수가 설정되지 않음")

    search_tool = GoogleSerperSearchResult.from_api_key(
        api_key=serper_api_key,
        k=15,
    )
    logger.debug("Google Serper 검색 도구 초기화 완료")

    research_node_llm = create_chat_model(provider="azure_openai", model="gpt-4o")
    logger.debug("연구 노드용 LLM 초기화 완료")

    research_node_agent = create_react_agent(
        research_node_llm,
        tools =[search_tool],
        response_format = ResearchNodeOutput,
    )
    logger.debug("연구 노드용 React 에이전트 생성 완료")
    
    research_node = create_research_node(
        research_node_agent
    )

    logger.debug("상태 그래프 빌더 생성")
    builder = StateGraph(ResearchAgentState)
    
    # 노드 추가
    logger.debug("연구 에이전트 노드 추가")
    builder.add_node("pre_processing_node", pre_processing_node)
    builder.add_node("post_processing_node", post_processing_node)
    builder.add_node("pre_processing_research_node", pre_processing_research_node)
    builder.add_node("research_node", research_node)
    builder.add_node("post_processing_research_node", post_processing_research_node)

    # 엣지 연결
    logger.debug("연구 에이전트 엣지 연결")
    builder.add_edge(START, "pre_processing_node")
    builder.add_edge("pre_processing_node", "pre_processing_research_node")
    builder.add_edge("pre_processing_research_node", "research_node")
    builder.add_edge("research_node", "post_processing_research_node")
    builder.add_edge("post_processing_research_node", "post_processing_node")
    builder.add_edge("post_processing_node", END)

    research_agent = builder.compile(name=name)
    logger.info(f"연구 에이전트 생성 완료: name='{name}'")
    
    return research_agent


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    import asyncio

    load_dotenv()
    logger.info("연구 에이전트 메인 실행 시작")

    agent = create_research_agent()
    
    test_state = {
        'topic': "이스트소프트",
        'name': '기업 개요 및 연혁',
        'description': '이스트소프트의 설립 배경, 주요 연혁, 대표자 및 기업의 주요 역사적 변화를 소개하는 섹션.'
    }
    
    logger.info("테스트 상태로 에이전트 실행")
    result = asyncio.run(agent.ainvoke(test_state))
    
    logger.info("에이전트 실행 완료")
    print(result)