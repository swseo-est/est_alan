from typing import Dict, Any

import langgraph.prebuilt
from langgraph.graph import START, END, StateGraph

from estalan.logging import get_logger

# 로거 생성
logger = get_logger(__name__)


def update_structured_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    상태에서 structured_response를 추출하여 반환하는 함수
    
    Args:
        state (Dict[str, Any]): 현재 상태 정보
        
    Returns:
        Dict[str, Any]: 구조화된 응답 데이터
    """
    logger.debug("structured_response 업데이트 시작", state_keys=list(state.keys()))
    
    structured_response = state.get("structured_response", {})
    logger.info("structured_response 추출됨", response_keys=list(structured_response.keys()))

    # reset structured_response
    updated_state = structured_response | {"structured_response": {}}
    logger.debug("structured_response 리셋 완료")
    
    return updated_state


def refresh_remaining_steps(state: Dict[str, Any]) -> Dict[str, Any]:
    """남은 단계 수를 새로고침하는 함수"""
    logger.debug("remaining_steps 새로고침", current_steps=state.get("remaining_steps"))
    
    new_steps = 25
    logger.info("remaining_steps 설정됨", new_steps=new_steps)
    
    return {"remaining_steps": new_steps}


def create_react_agent(*args, state_schema=None, pre_agent_node=None, post_agent_node=None, name=None, **kwargs):
    """
    LangGraph의 ReAct 에이전트를 확장하여 생성하는 함수
    
    Args:
        *args: 기본 ReAct 에이전트 생성에 필요한 인자들
        state_schema: 상태 스키마 정의
        pre_agent_node: 에이전트 실행 전에 실행될 노드 함수
        post_agent_node: 에이전트 실행 후에 실행될 노드 함수
        name: 그래프 이름
        **kwargs: 기본 ReAct 에이전트 생성에 필요한 키워드 인자들
        
    Returns:
        StateGraph: 컴파일된 상태 그래프
    """
    logger.info("ReAct 에이전트 생성 시작", 
                has_pre_node=pre_agent_node is not None,
                has_post_node=post_agent_node is not None,
                graph_name=name)
    
    try:
        # 기본 ReAct 에이전트 생성
        logger.debug("기본 ReAct 에이전트 생성 중...")
        react_agent = langgraph.prebuilt.create_react_agent(*args, **kwargs)
        logger.debug("기본 ReAct 에이전트 생성 완료")

        # 상태 그래프 빌더 초기화
        builder = StateGraph(state_schema)
        logger.debug("상태 그래프 빌더 초기화됨")

        # pre_agent_node가 제공된 경우 추가
        if pre_agent_node is not None:
            logger.debug("pre_agent_node 추가 중...")
            builder.add_node("pre_agent_node", pre_agent_node)
            
            # START -> pre_agent_node -> react_agent 순서로 연결
            builder.add_edge(START, "pre_agent_node")
            builder.add_edge("pre_agent_node", "react_agent")
            logger.debug("pre_agent_node 연결 완료")
        else:
            # pre_agent_node가 없는 경우 START에서 바로 react_agent로 연결
            logger.debug("START에서 react_agent로 직접 연결")
            builder.add_edge(START, "react_agent")

        # ReAct 에이전트와 structured_response 업데이트 노드 추가
        logger.debug("핵심 노드들 추가 중...")
        builder.add_node("react_agent", react_agent)
        builder.add_node("update_structured_response", update_structured_response)

        # refresh_remaining_steps 노드 추가
        builder.add_node("refresh_remaining_steps", refresh_remaining_steps)
        logger.debug("핵심 노드들 추가 완료")

        # post_agent_node가 제공된 경우 추가
        if post_agent_node is not None:
            logger.debug("post_agent_node 추가 중...")
            builder.add_node("post_agent_node", post_agent_node)
            
            # react_agent -> update_structured_response -> refresh_remaining_steps -> post_agent_node -> END 순서로 연결
            builder.add_edge("react_agent", "update_structured_response")
            builder.add_edge("update_structured_response", "refresh_remaining_steps")
            builder.add_edge("refresh_remaining_steps", "post_agent_node")
            builder.add_edge("post_agent_node", END)
            logger.debug("post_agent_node 연결 완료")
        else:
            # post_agent_node가 없는 경우 update_structured_response -> refresh_remaining_steps -> END 순서로 연결
            logger.debug("기본 엣지 연결 중...")
            builder.add_edge("react_agent", "update_structured_response")
            builder.add_edge("update_structured_response", "refresh_remaining_steps")
            builder.add_edge("refresh_remaining_steps", END)
            logger.debug("기본 엣지 연결 완료")

        # 그래프 컴파일하여 반환
        logger.info("그래프 컴파일 시작", graph_name=name)
        compiled_graph = builder.compile(name=name)
        logger.info("그래프 컴파일 완료", graph_name=name)
        
        return compiled_graph
        
    except Exception as e:
        logger.exception("ReAct 에이전트 생성 중 오류 발생", 
                        error_type=type(e).__name__,
                        error_message=str(e))
        raise

