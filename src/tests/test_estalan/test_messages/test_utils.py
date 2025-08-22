import pytest
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from estalan.messages.utils import (
    create_message,
    create_ai_message,
    create_human_message,
    create_system_message,
    create_tool_message
)
from estalan.messages.base import (
    AlanHumanMessage,
    AlanSystemMessage,
    AlanAIMessage,
    AlanToolMessage,
    convert_to_alan_message
)


def test_create_message_human():
    """human 타입으로 메시지 생성 테스트"""
    content = "Hello, how are you?"
    message = create_message("human", content)
    
    assert isinstance(message, AlanHumanMessage)
    assert message.content == content
    assert hasattr(message, 'id')
    assert hasattr(message, 'metadata')


def test_create_message_system():
    """system 타입으로 메시지 생성 테스트"""
    content = "You are a helpful assistant."
    message = create_message("system", content)
    
    assert isinstance(message, AlanSystemMessage)
    assert message.content == content
    assert hasattr(message, 'id')
    assert hasattr(message, 'metadata')


def test_create_message_ai():
    """ai 타입으로 메시지 생성 테스트"""
    content = "I'm doing well, thank you!"
    message = create_message("ai", content)
    
    assert isinstance(message, AlanAIMessage)
    assert message.content == content
    assert hasattr(message, 'id')
    assert hasattr(message, 'metadata')


def test_create_message_tool():
    """tool 타입으로 메시지 생성 테스트"""
    content = "Tool execution result"
    tool_call_id = "tool_123"
    message = create_message("tool", content, tool_call_id=tool_call_id)
    
    assert isinstance(message, AlanToolMessage)
    assert message.content == content
    assert message.tool_call_id == tool_call_id
    assert hasattr(message, 'id')
    assert hasattr(message, 'metadata')


def test_create_message_with_additional_kwargs():
    """추가 키워드 인자와 함께 메시지 생성 테스트"""
    content = "Test message"
    additional_data = {"key": "value"}
    message = create_message("human", content, **additional_data)
    
    assert isinstance(message, AlanHumanMessage)
    assert message.content == content
    # 추가 키워드 인자가 메시지에 설정되었는지 확인
    for key, value in additional_data.items():
        assert hasattr(message, key)
        assert getattr(message, key) == value


def test_create_message_invalid_type():
    """잘못된 메시지 타입에 대한 에러 테스트"""
    with pytest.raises(ValueError, match="Invalid message type: invalid"):
        create_message("invalid", "content")


def test_create_ai_message_basic():
    """기본 AI 메시지 생성 테스트"""
    content = "This is an AI response"
    message = create_ai_message(content)
    
    assert isinstance(message, AlanAIMessage)
    assert message.content == content
    assert hasattr(message, 'id')
    assert hasattr(message, 'metadata')


def test_create_ai_message_with_kwargs():
    """키워드 인자와 함께 AI 메시지 생성 테스트"""
    content = "AI response with kwargs"
    additional_data = {"response_type": "informative"}
    message = create_ai_message(content, **additional_data)
    
    assert isinstance(message, AlanAIMessage)
    assert message.content == content
    assert message.response_type == additional_data["response_type"]


def test_create_human_message_basic():
    """기본 Human 메시지 생성 테스트"""
    content = "This is a human message"
    message = create_human_message(content)
    
    assert isinstance(message, AlanHumanMessage)
    assert message.content == content
    assert hasattr(message, 'id')
    assert hasattr(message, 'metadata')


def test_create_human_message_with_kwargs():
    """키워드 인자와 함께 Human 메시지 생성 테스트"""
    content = "Human message with kwargs"
    additional_data = {"user_id": "user123"}
    message = create_human_message(content, **additional_data)
    
    assert isinstance(message, AlanHumanMessage)
    assert message.content == content
    assert message.user_id == additional_data["user_id"]


def test_create_system_message_basic():
    """기본 System 메시지 생성 테스트"""
    content = "This is a system message"
    message = create_system_message(content)
    
    assert isinstance(message, AlanSystemMessage)
    assert message.content == content
    assert hasattr(message, 'id')
    assert hasattr(message, 'metadata')


def test_create_system_message_with_kwargs():
    """키워드 인자와 함께 System 메시지 생성 테스트"""
    content = "System message with kwargs"
    additional_data = {"system_type": "configuration"}
    message = create_system_message(content, **additional_data)
    
    assert isinstance(message, AlanSystemMessage)
    assert message.content == content
    assert message.system_type == additional_data["system_type"]


def test_create_tool_message_basic():
    """기본 Tool 메시지 생성 테스트"""
    content = "This is a tool message"
    tool_call_id = "tool_default"
    message = create_tool_message(content, tool_call_id=tool_call_id)
    
    assert isinstance(message, AlanToolMessage)
    assert message.content == content
    assert message.tool_call_id == tool_call_id
    assert hasattr(message, 'id')
    assert hasattr(message, 'metadata')


def test_create_tool_message_with_kwargs():
    """키워드 인자와 함께 Tool 메시지 생성 테스트"""
    content = "Tool message with kwargs"
    tool_call_id = "tool_456"
    additional_data = {"tool_name": "calculator", "tool_call_id": tool_call_id}
    message = create_tool_message(content, **additional_data)
    
    assert isinstance(message, AlanToolMessage)
    assert message.content == content
    assert message.tool_name == additional_data["tool_name"]
    assert message.tool_call_id == tool_call_id


def test_message_has_uuid():
    """모든 메시지가 UUID를 가지고 있는지 테스트"""
    message_types = ["human", "system", "ai"]
    
    for msg_type in message_types:
        message = create_message(msg_type, "Test content")
        assert hasattr(message, 'id')
        assert message.id is not None
        assert len(message.id) > 0
    
    # tool 타입은 별도로 테스트 (tool_call_id 필요)
    tool_message = create_message("tool", "Test content", tool_call_id="test_tool")
    assert hasattr(tool_message, 'id')
    assert tool_message.id is not None
    assert len(tool_message.id) > 0


def test_message_has_metadata():
    """모든 메시지가 metadata를 가지고 있는지 테스트"""
    message_types = ["human", "system", "ai"]
    
    for msg_type in message_types:
        message = create_message(msg_type, "Test content")
        assert hasattr(message, 'metadata')
        assert isinstance(message.metadata, dict)
        assert "rendering_option" in message.metadata
    
    # tool 타입은 별도로 테스트 (tool_call_id 필요)
    tool_message = create_message("tool", "Test content", tool_call_id="test_tool")
    assert hasattr(tool_message, 'metadata')
    assert isinstance(tool_message.metadata, dict)
    assert "rendering_option" in tool_message.metadata


def test_message_content_preservation():
    """메시지 내용이 올바르게 보존되는지 테스트"""
    test_content = "This is a test message with special characters: !@#$%^&*()"
    message_types = ["human", "system", "ai"]
    
    for msg_type in message_types:
        message = create_message(msg_type, test_content)
        assert message.content == test_content
    
    # tool 타입은 별도로 테스트 (tool_call_id 필요)
    tool_message = create_message("tool", test_content, tool_call_id="test_tool")
    assert tool_message.content == test_content
