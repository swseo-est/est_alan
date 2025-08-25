from typing import Dict, Any
from estalan.messages.utils import create_ai_message
from estalan.agent.base.state import AlanAgentMetaData, create_default_state, BaseAlanAgentState


def alan_agent_start_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Alan Agent의 시작 노드입니다.
    초기화 상태에 따라 적절한 기본 상태를 생성합니다.
    
    Args:
        state: 현재 상태
        
    Returns:
        초기화된 상태
    """
    # state에서 initialization 값을 올바르게 가져오기
    initialization = state.get("initialization", False)
    
    if not initialization:
        # 초기화되지 않은 경우: 전체 상태에 대한 기본값 생성
        state = create_default_state(BaseAlanAgentState)
        state["metadata"]["initialization"] = True
    else:
        # 기존 상태를 유지하면서 metadata만 업데이트
        state["metadata"] = dict()

    state["metadata"]["chat_status"] = "unavailable"

    return state


def alan_agent_finish_node(state: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict()
    metadata["chat_status"] = "available"
    dummy = create_ai_message(content="")

    return {"metadata": metadata, "messages": [dummy]}
