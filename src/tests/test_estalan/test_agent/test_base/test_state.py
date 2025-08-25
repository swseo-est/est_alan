import pytest
from pydantic import ValidationError

from estalan.agent.base.state import (
    AlanAgentMetaData,
    BaseAlanAgentState,
    Canvas,
    AlanAgentStateWithCanvas
)
from estalan.agent.base.reducer_function import add_messages_for_alan
from langchain_core.messages import HumanMessage


def test_alan_agent_metadata_default_values():
    """AlanAgentMetaData의 기본값이 올바르게 설정되는지 테스트"""
    metadata = AlanAgentMetaData(
        chat_status="available",
        status="start"
    )
    assert metadata["chat_status"] == "available"
    assert metadata["status"] == "start"


def test_alan_agent_metadata_extra_fields():
    """AlanAgentMetaData에 추가 필드를 수동으로 추가할 수 있는지 테스트"""
    # 기본 필드와 함께 추가 필드 생성
    metadata = AlanAgentMetaData(
        chat_status="available",
        status="start"
    )
    # TypedDict는 딕셔너리이므로 추가 필드 가능
    metadata["user_id"] = 123
    metadata["timestamp"] = "2024-01-01"
    metadata["custom_field"] = "custom_value"
    
    # 기본 필드 확인
    assert metadata["chat_status"] == "available"
    assert metadata["status"] == "start"
    
    # 추가 필드 확인
    assert metadata["custom_field"] == "custom_value"
    assert metadata["user_id"] == 123
    assert metadata["timestamp"] == "2024-01-01"
    
    # 나중에 필드 추가
    metadata["new_field"] = "new_value"
    assert metadata["new_field"] == "new_value"


def test_base_alan_agent_state_metadata_type():
    """BaseAlanAgentState의 metadata가 AlanAgentMetaData 타입인지 테스트"""
    # 필수 필드들을 포함하여 상태 생성
    state = BaseAlanAgentState(
        messages=[],
        structured_response={},
        metadata=AlanAgentMetaData(
            chat_status="available",
            status="start"
        )
    )
    # TypedDict이므로 딕셔너리로 접근
    assert isinstance(state, dict)
    assert "metadata" in state
    assert isinstance(state["metadata"], dict)
    assert state["metadata"]["chat_status"] == "available"
    assert state["metadata"]["status"] == "start"


def test_canvas_required_fields():
    """Canvas 클래스의 필수 필드들이 올바르게 설정되는지 테스트"""
    canvas = Canvas(
        type="markdown",
        metadata={"key": "value"}
    )
    assert canvas["type"] == "markdown"
    assert canvas["metadata"] == {"key": "value"}


def test_alan_agent_state_with_canvas_inheritance():
    """AlanAgentStateWithCanvas가 BaseAlanAgentState를 올바르게 상속하는지 테스트"""
    # 필수 필드들을 포함하여 상태 생성
    state = AlanAgentStateWithCanvas(
        messages=[HumanMessage(content="test")],
        structured_response={},
        metadata=AlanAgentMetaData(
            chat_status="available",
            status="start"
        ),
        canvases=[]
    )
    # TypedDict이므로 딕셔너리로 접근
    assert isinstance(state, dict)
    assert "messages" in state
    assert "structured_response" in state
    assert "metadata" in state
    assert "canvases" in state
    assert state["canvases"] == []
