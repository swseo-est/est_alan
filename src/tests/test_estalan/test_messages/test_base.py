import pytest

from estalan.messages.base import (
    AlanAIMessage,
    AlanHumanMessage,
    AlanSystemMessage,
    AlanToolMessage,
    convert_to_alan_message,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def print_message_info(message, title):
    """메시지 정보를 출력하는 헬퍼 함수"""
    print(f"\n=== {title} ===")
    print(f"Type: {type(message)}")
    print(f"Content: {message.content}")
    print(f"ID: {message.id}")
    print(f"Additional kwargs: {message.additional_kwargs}")
    print(f"Response metadata: {getattr(message, 'response_metadata', 'N/A')}")
    print(f"Tool calls: {getattr(message, 'tool_calls', 'N/A')}")
    print(f"Name: {message.name}")
    print(f"Metadata: {getattr(message, 'metadata', 'N/A')}")
    print(f"Dict keys: {list(message.__dict__.keys())}")


def message_conversion(original_msg, expected_alan_class, expected_attrs):
    """메시지 변환을 테스트하는 공통 함수"""
    # 변환 전 출력
    print_message_info(original_msg, f"변환 전 {original_msg.__class__.__name__}")
    
    # 변환
    converted_msg = convert_to_alan_message(original_msg)
    
    # 변환 후 출력
    print_message_info(converted_msg, f"변환 후 {converted_msg.__class__.__name__}")
    
    # 타입 검증
    assert isinstance(converted_msg, expected_alan_class)
    
    # 속성 검증
    for attr_name, expected_value in expected_attrs.items():
        actual_value = getattr(converted_msg, attr_name)
        assert actual_value == expected_value, f"{attr_name}: expected {expected_value}, got {actual_value}"
    
    # metadata 필드 검증
    assert hasattr(converted_msg, 'metadata'), "Alan 메시지는 metadata 필드를 가져야 함"
    assert isinstance(converted_msg.metadata, dict), "metadata는 dict 타입이어야 함"
    
    # metadata 기본값을 실제 팩토리에서 가져와서 비교
    from estalan.messages.base import default_metadata_factory
    expected_metadata = default_metadata_factory()
    
    # metadata의 키들이 예상된 키들과 일치하는지만 검증
    assert set(converted_msg.metadata.keys()) == set(expected_metadata.keys()), \
        f"metadata 키가 일치하지 않음. expected: {set(expected_metadata.keys())}, got: {set(converted_msg.metadata.keys())}"


def test_convert_aimessage_to_alan():
    """Test converting AIMessage to AlanAIMessage."""
    # Create original AIMessage with various attributes
    original_msg = AIMessage(
        content="Test AI content",
        id="original-ai-id",
        additional_kwargs={"key1": "value1", "key2": "value2"},
        response_metadata={"model": "gpt-4"},
        tool_calls=[{"id": "tool1", "name": "test_tool", "args": {"param1": "value1"}}],
        name="test_ai_agent"
    )
    
    expected_attrs = {
        'content': "Test AI content",
        'id': "original-ai-id",
        'additional_kwargs': {"key1": "value1", "key2": "value2"},
        'response_metadata': {"model": "gpt-4"},
        'name': "test_ai_agent"
    }
    
    message_conversion(original_msg, AlanAIMessage, expected_attrs)
    
    # tool_calls 별도 검증 (type 필드가 추가될 수 있음)
    converted_msg = convert_to_alan_message(original_msg)
    assert len(converted_msg.tool_calls) == 1
    tool_call = converted_msg.tool_calls[0]
    assert tool_call["id"] == "tool1"
    assert tool_call["name"] == "test_tool"
    assert tool_call["args"] == {"param1": "value1"}


def test_convert_humanmessage_to_alan():
    """Test converting HumanMessage to AlanHumanMessage."""
    original_msg = HumanMessage(
        content="Test human content",
        id="original-human-id",
        additional_kwargs={"user_id": "user123"},
        name="test_user"
    )
    
    expected_attrs = {
        'content': "Test human content",
        'id': "original-human-id",
        'additional_kwargs': {"user_id": "user123"},
        'name': "test_user"
    }
    
    message_conversion(original_msg, AlanHumanMessage, expected_attrs)


def test_convert_systemmessage_to_alan():
    """Test converting SystemMessage to AlanSystemMessage."""
    original_msg = SystemMessage(
        content="Test system content",
        id="original-system-id",
        additional_kwargs={"system_type": "assistant"}
    )
    
    expected_attrs = {
        'content': "Test system content",
        'id': "original-system-id",
        'additional_kwargs': {"system_type": "assistant"}
    }
    
    message_conversion(original_msg, AlanSystemMessage, expected_attrs)


def test_convert_toolmessage_to_alan():
    """Test converting ToolMessage to AlanToolMessage."""
    original_msg = ToolMessage(
        content="Tool execution result",
        tool_call_id="tool-call-123",
        id="original-tool-id",
        name="test_tool",
        additional_kwargs={"execution_time": 0.5}
    )
    
    expected_attrs = {
        'content': "Tool execution result",
        'tool_call_id': "tool-call-123",
        'id': "original-tool-id",
        'name': "test_tool",
        'additional_kwargs': {"execution_time": 0.5}
    }
    
    message_conversion(original_msg, AlanToolMessage, expected_attrs)


def test_convert_message_without_id():
    """Test converting message without explicit ID (should generate new UUID)."""
    original_msg = AIMessage(content="No ID message")
    
    # 변환 전 출력
    print_message_info(original_msg, "변환 전 AIMessage (No ID)")
    
    # 변환
    converted_msg = convert_to_alan_message(original_msg)
    
    # 변환 후 출력
    print_message_info(converted_msg, "변환 후 AlanAIMessage (Generated ID)")
    
    # 검증
    assert isinstance(converted_msg, AlanAIMessage)
    assert converted_msg.id is not None
    assert isinstance(converted_msg.id, str)
    assert len(converted_msg.id) == 36
    assert converted_msg.content == "No ID message"


def test_convert_message_with_private_attributes():
    """Test that private attributes (starting with _) are not copied."""
    original_msg = AIMessage(content="Private test")
    original_msg._private_attr = "private_value"
    original_msg.__internal_attr = "internal_value"
    
    # 변환 전 출력
    print_message_info(original_msg, "변환 전 AIMessage (with private attrs)")
    
    # 변환
    converted_msg = convert_to_alan_message(original_msg)
    
    # 변환 후 출력
    print_message_info(converted_msg, "변환 후 AlanAIMessage (private attrs filtered)")
    
    # 검증
    assert isinstance(converted_msg, AlanAIMessage)
    assert not hasattr(converted_msg, '_private_attr')
    assert not hasattr(converted_msg, '__internal_attr')
    assert converted_msg.content == "Private test"


def test_convert_message_with_slots():
    """Test converting message that uses __slots__ instead of __dict__."""
    class SlottedMessage(AIMessage):
        __slots__ = ('custom_attr', 'another_attr')
        
        def __init__(self, content: str, custom_attr: str, another_attr: str):
            super().__init__(content=content)
            self.custom_attr = custom_attr
            self.another_attr = another_attr
    
    original_msg = SlottedMessage(
        content="Slotted message",
        custom_attr="custom_value",
        another_attr="another_value"
    )
    
    # 변환 전 출력
    print_message_info(original_msg, "변환 전 SlottedMessage")
    
    # 변환
    converted_msg = convert_to_alan_message(original_msg)
    
    # 변환 후 출력
    print_message_info(converted_msg, "변환 후 AlanAIMessage (with slotted attrs)")
    
    # 검증
    assert isinstance(converted_msg, AlanAIMessage)
    assert converted_msg.custom_attr == "custom_value"
    assert converted_msg.another_attr == "another_value"
    assert converted_msg.content == "Slotted message"


def test_convert_message_with_nested_structures():
    """Test converting message with nested data structures."""
    original_msg = AIMessage(
        content="Nested test",
        additional_kwargs={
            "nested_dict": {
                "level1": {
                    "level2": ["item1", "item2"],
                    "level2_dict": {"key": "value"}
                }
            },
            "nested_list": [1, [2, 3], {"key": "value"}]
        }
    )
    
    expected_attrs = {
        'content': "Nested test",
        'additional_kwargs': {
            "nested_dict": {
                "level1": {
                    "level2": ["item1", "item2"],
                    "level2_dict": {"key": "value"}
                }
            },
            "nested_list": [1, [2, 3], {"key": "value"}]
        }
    }
    
    message_conversion(original_msg, AlanAIMessage, expected_attrs)


def test_convert_unsupported_message_type():
    """Test that unsupported message types raise appropriate exceptions."""
    class CustomMessage:
        def __init__(self, content: str):
            self.content = content
    
    custom_msg = CustomMessage("Custom content")
    
    # 변환 전 출력
    print(f"\n=== 변환 전 CustomMessage ===")
    print(f"Type: {type(custom_msg)}")
    print(f"Content: {custom_msg.content}")
    
    # 예외 발생 확인
    with pytest.raises(Exception) as exc_info:
        convert_to_alan_message(custom_msg)
    
    assert "Unsupported message type" in str(exc_info.value)
    assert "CustomMessage" in str(exc_info.value)


def test_convert_message_preserves_all_public_attributes():
    """Test that all public attributes are preserved during conversion."""
    original_msg = AIMessage(
        content="Comprehensive test",
        id="comprehensive-id",
        additional_kwargs={
            "attr1": "value1",
            "attr2": 42,
            "attr3": True,
            "attr4": None,
            "attr5": [1, 2, 3],
            "attr6": {"nested": "value"}
        },
        response_metadata={
            "model": "gpt-4",
            "usage": {"tokens": 100},
            "finish_reason": "stop"
        },
        tool_calls=[
            {"id": "call1", "name": "tool1", "args": {"arg1": "value1"}},
            {"id": "call2", "name": "tool2", "args": {"arg2": "value2"}}
        ],
        name="comprehensive_agent",
        example=True
    )
    
    expected_attrs = {
        'content': "Comprehensive test",
        'id': "comprehensive-id",
        'additional_kwargs': {
            "attr1": "value1",
            "attr2": 42,
            "attr3": True,
            "attr4": None,
            "attr5": [1, 2, 3],
            "attr6": {"nested": "value"}
        },
        'response_metadata': {
            "model": "gpt-4",
            "usage": {"tokens": 100},
            "finish_reason": "stop"
        },
        'name': "comprehensive_agent",
        'example': True
    }
    
    message_conversion(original_msg, AlanAIMessage, expected_attrs)
    
    # tool_calls 별도 검증
    converted_msg = convert_to_alan_message(original_msg)
    assert len(converted_msg.tool_calls) == 2
    
    tool_call1 = converted_msg.tool_calls[0]
    assert tool_call1["id"] == "call1"
    assert tool_call1["name"] == "tool1"
    assert tool_call1["args"] == {"arg1": "value1"}
    
    tool_call2 = converted_msg.tool_calls[1]
    assert tool_call2["id"] == "call2"
    assert tool_call2["name"] == "tool2"
    assert tool_call2["args"] == {"arg2": "value2"}
