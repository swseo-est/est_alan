from typing import Dict, Any
from estalan.messages.utils import create_ai_message
from estalan.agent.base.state import AlanAgentMetaData, create_default_state, BaseAlanAgentState
from estalan.agent.base.reducer_function import update_metadata


def create_alan_agent_start_node(
        list_init_state=[BaseAlanAgentState]
):
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

        updated_state = {"metadata": dict()}
        if not initialization:
            for base_state in list_init_state:
                # 초기화되지 않은 경우: 전체 상태에 대한 기본값 생성
                added_state = create_default_state(base_state)
                updated_state = update_metadata(updated_state, added_state)

            updated_state["metadata"]["initialization"] = True

        updated_state["metadata"]["chat_status"] = "unavailable"
        return updated_state
    return alan_agent_start_node


def alan_agent_finish_node(state: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict()
    metadata["chat_status"] = "available"

    return {"metadata": metadata}
