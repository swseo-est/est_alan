from typing import Dict, Any

import langgraph.prebuilt
from langgraph.graph import START, END, StateGraph


def update_structured_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    상태에서 structured_response를 추출하여 반환하는 함수
    
    Args:
        state (Dict[str, Any]): 현재 상태 정보
        
    Returns:
        Dict[str, Any]: 구조화된 응답 데이터
    """
    structured_response = state.get("structured_response", {})

    # reset structured_response
    updated_state = structured_response | {"structured_response": {}}
    return updated_state


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
    # 기본 ReAct 에이전트 생성
    react_agent = langgraph.prebuilt.create_react_agent(*args, **kwargs)

    # 상태 그래프 빌더 초기화
    builder = StateGraph(state_schema)

    # pre_agent_node가 제공된 경우 추가
    if pre_agent_node is not None:
        builder.add_node("pre_agent_node", pre_agent_node)
        
        # START -> pre_agent_node -> react_agent 순서로 연결
        builder.add_edge(START, "pre_agent_node")
        builder.add_edge("pre_agent_node", "react_agent")
    else:
        # pre_agent_node가 없는 경우 START에서 바로 react_agent로 연결
        builder.add_edge(START, "react_agent")

    # ReAct 에이전트와 structured_response 업데이트 노드 추가
    builder.add_node("react_agent", react_agent)
    builder.add_node("update_structured_response", update_structured_response)

    # post_agent_node가 제공된 경우 추가
    if post_agent_node is not None:
        builder.add_node("post_agent_node", post_agent_node)
        
        # react_agent -> update_structured_response -> post_agent_node -> END 순서로 연결
        builder.add_edge("react_agent", "update_structured_response")
        builder.add_edge("update_structured_response", "post_agent_node")
        builder.add_edge("post_agent_node", END)
    else:
        # post_agent_node가 없는 경우 update_structured_response에서 바로 END로 연결
        builder.add_edge("react_agent", "update_structured_response")
        builder.add_edge("update_structured_response", END)

    # 그래프 컴파일하여 반환
    return builder.compile(name=name)

