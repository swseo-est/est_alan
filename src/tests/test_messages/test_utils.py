import pytest
from langchain_core.messages import BaseMessage
from estalan.messages.utils import (
    create_message, 
    create_ai_message, 
    create_human_message, 
    create_system_message, 
    create_tool_message, 
    create_block_message
)
from estalan.messages.base import (
    AlanHumanMessage, 
    AlanSystemMessage, 
    AlanAIMessage, 
    AlanToolMessage, 
    BaseAlanBlockMessage
)


def test_create_message_human():
    """create_message 함수의 human 타입을 테스트합니다."""
    
    # 함수 실행
    message = create_message("human", "Hello, how are you?")
    
    # 결과 검증
    assert isinstance(message, AlanHumanMessage)
    assert message.content == "Hello, how are you?"
    assert message.id is not None


def test_create_message_system():
    """create_message 함수의 system 타입을 테스트합니다."""
    
    # 함수 실행
    message = create_message("system", "You are a helpful assistant.")
    
    # 결과 검증
    assert isinstance(message, AlanSystemMessage)
    assert message.content == "You are a helpful assistant."
    assert message.id is not None


def test_create_message_ai():
    """create_message 함수의 ai 타입을 테스트합니다."""
    
    # 함수 실행
    message = create_message("ai", "I'm doing well, thank you!")
    
    # 결과 검증
    assert isinstance(message, AlanAIMessage)
    assert message.content == "I'm doing well, thank you!"
    assert message.id is not None


def test_create_message_tool():
    """create_message 함수의 tool 타입을 테스트합니다."""
    
    # 함수 실행
    message = create_message("tool", "Tool execution completed", tool_call_id="tool123")
    
    # 결과 검증
    assert isinstance(message, AlanToolMessage)
    assert message.content == "Tool execution completed"
    assert message.tool_call_id == "tool123"
    assert message.id is not None


def test_create_message_block():
    """create_message 함수의 block 타입을 테스트합니다."""
    
    # 함수 실행
    message = create_message("block", "print('Hello World')", block_tag="python")
    
    # 결과 검증
    assert isinstance(message, BaseAlanBlockMessage)
    assert message.content == "```python\nprint('Hello World')\n```"
    assert message.block_tag == "python"
    assert message.id is not None


def test_create_message_invalid_type():
    """create_message 함수에 잘못된 타입을 전달했을 때의 동작을 테스트합니다."""
    
    # 잘못된 타입으로 함수 실행 시 ValueError 발생해야 함
    with pytest.raises(ValueError, match="Invalid message type: invalid"):
        create_message("invalid", "Some content")


def test_create_ai_message():
    """create_ai_message 함수를 테스트합니다."""
    
    # 함수 실행
    message = create_ai_message("This is an AI response")
    
    # 결과 검증
    assert isinstance(message, AlanAIMessage)
    assert message.content == "This is an AI response"
    assert message.id is not None


def test_create_human_message():
    """create_human_message 함수를 테스트합니다."""
    
    # 함수 실행
    message = create_human_message("This is a human message")
    
    # 결과 검증
    assert isinstance(message, AlanHumanMessage)
    assert message.content == "This is a human message"
    assert message.id is not None


def test_create_system_message():
    """create_system_message 함수를 테스트합니다."""
    
    # 함수 실행
    message = create_system_message("This is a system message")
    
    # 결과 검증
    assert isinstance(message, AlanSystemMessage)
    assert message.content == "This is a system message"
    assert message.id is not None


def test_create_tool_message():
    """create_tool_message 함수를 테스트합니다."""
    
    # 함수 실행
    message = create_tool_message("Tool result", tool_call_id="tool456")
    
    # 결과 검증
    assert isinstance(message, AlanToolMessage)
    assert message.content == "Tool result"
    assert message.tool_call_id == "tool456"
    assert message.id is not None


def test_create_block_message():
    """create_block_message 함수를 테스트합니다."""
    
    # 함수 실행
    message = create_block_message("def hello():", "python")
    
    # 결과 검증
    assert isinstance(message, BaseAlanBlockMessage)
    assert message.content == "```python\ndef hello():\n```"
    assert message.block_tag == "python"
    assert message.id is not None


def test_create_block_message_without_block_tag():
    """create_block_message 함수에서 block_tag를 None으로 전달했을 때의 동작을 테스트합니다."""
    
    # 함수 실행 (block_tag를 None으로 전달)
    message = create_block_message("Some content", None)
    
    # 결과 검증
    assert isinstance(message, BaseAlanBlockMessage)
    assert message.content == "```\nSome content\n```"
    assert message.block_tag is None
    assert message.id is not None


def test_create_message_with_additional_kwargs():
    """create_message 함수에 추가 키워드 인자를 전달했을 때의 동작을 테스트합니다."""
    
    # 추가 키워드 인자와 함께 함수 실행
    message = create_message("human", "Hello", additional_info="test", priority="high")
    
    # 결과 검증
    assert isinstance(message, AlanHumanMessage)
    assert message.content == "Hello"
    assert message.id is not None
    # 추가 키워드 인자는 additional_kwargs에 저장됨
    assert hasattr(message, 'additional_kwargs')
