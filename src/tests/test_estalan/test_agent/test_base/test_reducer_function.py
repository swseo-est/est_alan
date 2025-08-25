import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from estalan.agent.base.reducer_function import (
    add_messages_for_alan,
    merge_message,
    update_metadata
)
from estalan.messages.base import AlanAIMessage, AlanHumanMessage, AlanSystemMessage


def test_add_messages_for_alan_with_single_messages():
    """단일 메시지로 add_messages_for_alan 함수 테스트"""
    left_message = HumanMessage(content="Hello", id="msg1")
    right_message = AIMessage(content="Hi there", id="msg2")
    
    with patch('estalan.agent.base.reducer_function.add_messages') as mock_add_messages:
        mock_add_messages.return_value = [right_message]
        
        result = add_messages_for_alan(left_message, right_message)
        
        # add_messages가 호출되었는지 확인
        mock_add_messages.assert_called_once_with([], [right_message])
        
        # 결과가 리스트인지 확인
        assert isinstance(result, list)
        assert len(result) == 2


def test_add_messages_for_alan_with_message_lists():
    """메시지 리스트로 add_messages_for_alan 함수 테스트"""
    left_messages = [
        HumanMessage(content="Hello", id="msg1"),
        SystemMessage(content="System info", id="msg2")
    ]
    right_messages = [
        AIMessage(content="Response 1", id="msg3"),
        AIMessage(content="Response 2", id="msg4")
    ]
    
    with patch('estalan.agent.base.reducer_function.add_messages') as mock_add_messages:
        mock_add_messages.return_value = right_messages
        
        result = add_messages_for_alan(left_messages, right_messages)
        
        # add_messages가 호출되었는지 확인
        mock_add_messages.assert_called_once_with([], right_messages)
        
        # 결과가 리스트인지 확인
        assert isinstance(result, list)
        assert len(result) == 4


def test_add_messages_for_alan_converts_to_alan_messages():
    """add_messages_for_alan이 right 메시지를 Alan 메시지로 변환하는지 테스트"""
    left_message = HumanMessage(content="Hello", id="msg1")
    right_message = AIMessage(content="Hi there", id="msg2")
    
    with patch('estalan.agent.base.reducer_function.add_messages') as mock_add_messages:
        mock_add_messages.return_value = [right_message]
        
        result = add_messages_for_alan(left_message, right_message)
        
        # left 메시지는 변환되지 않고, right 메시지만 Alan 메시지로 변환됨
        assert isinstance(result[0], HumanMessage)  # left는 변환되지 않음
        assert isinstance(result[1], AlanAIMessage)  # right는 Alan 메시지로 변환됨


def test_merge_message_basic_merge():
    """기본적인 메시지 병합 테스트"""
    left_messages = [
        HumanMessage(content="Hello", id="msg1"),
        AIMessage(content="Hi", id="msg2")
    ]
    right_messages = [
        AIMessage(content="How are you?", id="msg3")
    ]
    
    result = merge_message(left_messages, right_messages)
    
    assert len(result) == 3
    assert result[0].id == "msg1"
    assert result[1].id == "msg2"
    assert result[2].id == "msg3"


def test_merge_message_update_existing():
    """기존 메시지 업데이트 테스트"""
    left_messages = [
        HumanMessage(content="Hello", id="msg1"),
        AIMessage(content="Hi", id="msg2")
    ]
    right_messages = [
        AIMessage(content="Updated response", id="msg2")  # 기존 msg2 업데이트
    ]
    
    result = merge_message(left_messages, right_messages)
    
    assert len(result) == 2
    assert result[0].id == "msg1"
    assert result[1].id == "msg2"
    assert result[1].content == "Updated response"


def test_merge_message_remove_message():
    """RemoveMessage를 사용한 메시지 제거 테스트"""
    left_messages = [
        HumanMessage(content="Hello", id="msg1"),
        AIMessage(content="Hi", id="msg2"),
        SystemMessage(content="System", id="msg3")
    ]
    right_messages = [
        RemoveMessage(id="msg2")  # msg2 제거
    ]
    
    result = merge_message(left_messages, right_messages)
    
    assert len(result) == 2
    assert result[0].id == "msg1"
    assert result[1].id == "msg3"


def test_merge_message_remove_nonexistent_message():
    """존재하지 않는 메시지 제거 시도 시 예외 발생 테스트"""
    left_messages = [
        HumanMessage(content="Hello", id="msg1")
    ]
    right_messages = [
        RemoveMessage(id="nonexistent")  # 존재하지 않는 메시지 제거 시도
    ]
    
    with pytest.raises(ValueError, match="Attempting to delete a message with an ID that doesn't exist"):
        merge_message(left_messages, right_messages)


def test_merge_message_with_single_messages():
    """단일 메시지로 merge_message 함수 테스트"""
    left_message = HumanMessage(content="Hello", id="msg1")
    right_message = AIMessage(content="Hi", id="msg2")
    
    # 단일 메시지를 리스트로 변환하여 전달
    result = merge_message([left_message], [right_message])
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].id == "msg1"
    assert result[1].id == "msg2"


def test_update_metadata_basic_update():
    """기본적인 메타데이터 업데이트 테스트"""
    metadata = {"key1": "value1", "key2": "value2"}
    metadata_new = {"key3": "value3", "key4": "value4"}
    
    result = update_metadata(metadata_new, metadata)
    
    assert result == metadata  # 원본 딕셔너리가 수정됨
    assert metadata["key1"] == "value1"
    assert metadata["key2"] == "value2"
    assert metadata["key3"] == "value3"
    assert metadata["key4"] == "value4"


def test_update_metadata_overwrite_existing():
    """기존 키 덮어쓰기 테스트"""
    metadata = {"key1": "old_value", "key2": "value2"}
    metadata_new = {"key1": "new_value", "key3": "value3"}
    
    result = update_metadata(metadata_new, metadata)
    
    assert result == metadata
    assert metadata["key1"] == "new_value"  # 기존 값이 덮어써짐
    assert metadata["key2"] == "value2"
    assert metadata["key3"] == "value3"


def test_update_metadata_empty_new_metadata():
    """빈 새 메타데이터로 업데이트 테스트"""
    metadata = {"key1": "value1", "key2": "value2"}
    metadata_new = {}
    
    result = update_metadata(metadata_new, metadata)
    
    assert result == metadata
    assert metadata["key1"] == "value1"
    assert metadata["key2"] == "value2"


def test_update_metadata_empty_original_metadata():
    """빈 원본 메타데이터에 새 메타데이터 추가 테스트"""
    metadata = {}
    metadata_new = {"key1": "value1", "key2": "value2"}
    
    result = update_metadata(metadata_new, metadata)
    
    assert result == metadata
    assert metadata["key1"] == "value1"
    assert metadata["key2"] == "value2"


def test_update_metadata_both_empty():
    """둘 다 빈 딕셔너리인 경우 테스트"""
    metadata = {}
    metadata_new = {}
    
    result = update_metadata(metadata_new, metadata)
    
    assert result == metadata
    assert len(metadata) == 0
