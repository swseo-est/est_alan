import pytest
from pydantic import ValidationError

from estalan.agent.base.state import (
    AlanAgentMetaData,
    BaseAlanAgentState,
    Canvas,
    AlanAgentStateWithCanvas,
    create_default_state
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


def test_create_default_state_basic():
    """create_default_state 함수의 기본 동작을 테스트합니다"""
    # TypedDict 클래스에 대해 기본 상태 생성
    default_state = create_default_state(BaseAlanAgentState)
    
    # 기본값이 올바르게 설정되었는지 확인
    assert isinstance(default_state, dict)
    
    # BaseAlanAgentState의 필수 필드들 확인
    assert "messages" in default_state
    assert "metadata" in default_state
    
    # messages 필드는 빈 리스트여야 함
    assert default_state["messages"] == []
    
    # metadata 필드는 AlanAgentMetaData의 기본값을 가져야 함
    assert isinstance(default_state["metadata"], dict)
    assert "chat_status" in default_state["metadata"]
    assert "status" in default_state["metadata"]
    assert "initialization" in default_state["metadata"]
    
    # Literal 타입의 실제 구조 확인
    from typing import get_type_hints, get_origin, get_args
    metadata_hints = get_type_hints(AlanAgentMetaData)
    
    print(f"생성된 기본 상태: {default_state}")
    print(f"metadata 내용: {default_state['metadata']}")
    print(f"chat_status 타입: {metadata_hints['chat_status']}")
    print(f"chat_status origin: {get_origin(metadata_hints['chat_status'])}")
    print(f"chat_status args: {get_args(metadata_hints['chat_status'])}")
    print(f"status 타입: {metadata_hints['status']}")
    print(f"status origin: {get_origin(metadata_hints['status'])}")
    print(f"status args: {get_args(metadata_hints['status'])}")
    print(f"initialization 타입: {metadata_hints['initialization']}")
    print(f"initialization origin: {get_origin(metadata_hints['initialization'])}")
    print(f"initialization args: {get_args(metadata_hints['initialization'])}")
