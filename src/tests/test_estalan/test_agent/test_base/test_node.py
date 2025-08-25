import pytest
from unittest.mock import patch, MagicMock
from typing import TypedDict, Dict, Any
from estalan.agent.base.node import create_alan_agent_start_node, alan_agent_finish_node
from estalan.messages.utils import create_ai_message
from estalan.agent.base.state import BaseAlanAgentState


def test_create_alan_agent_start_node_sets_chat_status_unavailable():
    """create_alan_agent_start_node가 chat_status를 'unavailable'로 설정하는지 테스트"""
    # Given
    state = {"some": "data"}
    
    # When
    alan_agent_start_node = create_alan_agent_start_node()
    result = alan_agent_start_node(state)
    
    # Then
    assert result["metadata"]["chat_status"] == "unavailable"


def test_create_alan_agent_start_node_initializes_state_when_not_initialized():
    """create_alan_agent_start_node가 초기화되지 않은 상태에서 기본 상태를 생성하는지 테스트"""
    # Given
    state = {"some": "data"}
    
    # When
    alan_agent_start_node = create_alan_agent_start_node()
    result = alan_agent_start_node(state)
    
    # Then
    assert result["metadata"]["initialization"] == True
    assert result["metadata"]["chat_status"] == "unavailable"


def test_create_alan_agent_start_node_preserves_state_when_initialized():
    """create_alan_agent_start_node가 이미 초기화된 상태를 보존하는지 테스트"""
    # Given
    state = {"some": "data", "metadata": {"initialization": True}}
    
    # When
    alan_agent_start_node = create_alan_agent_start_node()
    result = alan_agent_start_node(state)
    
    # Then
    assert result["metadata"]["initialization"] == True
    assert result["metadata"]["chat_status"] == "unavailable"


def test_create_alan_agent_start_node_with_custom_state_list():
    """create_alan_agent_start_node가 사용자 정의 상태 리스트로 초기화하는지 테스트"""
    # Given
    state = {"some": "data"}
    custom_state_list = [BaseAlanAgentState]
    
    # When
    alan_agent_start_node = create_alan_agent_start_node(custom_state_list)
    result = alan_agent_start_node(state)
    
    # Then
    assert result["metadata"]["initialization"] == True
    assert result["metadata"]["chat_status"] == "unavailable"


def test_create_alan_agent_start_node_with_multiple_states():
    """create_alan_agent_start_node가 여러 상태를 병합하는지 테스트"""
    # Given
    state = {"some": "data"}
    multiple_states = [BaseAlanAgentState, BaseAlanAgentState]  # 같은 상태 클래스라도 여러 개
    
    # When
    alan_agent_start_node = create_alan_agent_start_node(multiple_states)
    result = alan_agent_start_node(state)
    
    # Then
    assert result["metadata"]["initialization"] == True
    assert result["metadata"]["chat_status"] == "unavailable"


def test_create_alan_agent_start_node_with_empty_state_list():
    """create_alan_agent_start_node가 빈 상태 리스트로도 동작하는지 테스트"""
    # Given
    state = {"some": "data"}
    empty_state_list = []
    
    # When
    alan_agent_start_node = create_alan_agent_start_node(empty_state_list)
    result = alan_agent_start_node(state)
    
    # Then
    assert result["metadata"]["initialization"] == True
    assert result["metadata"]["chat_status"] == "unavailable"


def test_create_alan_agent_start_node_with_different_state_types():
    """create_alan_agent_start_node가 다양한 형태의 상태들을 병합하는지 테스트"""
    # Given
    state = {"some": "data"}
    
    # 다양한 형태의 상태들을 TypedDict로 정의
    class MockState1(TypedDict):
        field1: str
        field2: str
    
    class MockState2(TypedDict):
        field3: str
        field4: str
    
    class MockState3(TypedDict):
        field5: str
        field6: str
    
    # TypedDict 인스턴스 생성
    different_states = [
        MockState1,
        MockState2,
        MockState3
    ]
    
    # When
    alan_agent_start_node = create_alan_agent_start_node(different_states)
    result = alan_agent_start_node(state)
    
    # Then
    assert result["metadata"]["initialization"] == True
    assert result["metadata"]["chat_status"] == "unavailable"
    # 각 상태의 필드들이 결과에 포함되어 있는지 확인
    assert "field1" in result
    assert "field2" in result
    assert "field3" in result
    assert "field4" in result
    assert "field5" in result
    assert "field6" in result


def test_create_alan_agent_start_node_with_duplicate_keys():
    """create_alan_agent_start_node가 중복 키를 후순위 상태 값으로 업데이트하는지 테스트"""
    # Given
    state = {"some": "data"}
    
    # 중복 키를 가진 상태들을 TypedDict로 정의
    class MockStateWithDuplicate1(TypedDict):
        common_field: str
        unique_field1: str
    
    class MockStateWithDuplicate2(TypedDict):
        common_field: str  # 중복 키, 다른 값
        unique_field2: str
    
    class MockStateWithDuplicate3(TypedDict):
        common_field: str   # 중복 키, 최종 값
        unique_field3: str
    
    # TypedDict 인스턴스 생성
    states_with_duplicates = [
        MockStateWithDuplicate1,
        MockStateWithDuplicate2,
        MockStateWithDuplicate3
    ]
    
    # When
    alan_agent_start_node = create_alan_agent_start_node(states_with_duplicates)
    result = alan_agent_start_node(state)
    
    # Then
    assert result["metadata"]["initialization"] == True
    assert result["metadata"]["chat_status"] == "unavailable"
    # 중복 키는 마지막 상태의 값으로 설정되어야 함
    assert "common_field" in result
    assert "unique_field1" in result
    assert "unique_field2" in result
    assert "unique_field3" in result


def test_create_alan_agent_start_node_with_nested_states():
    """create_alan_agent_start_node가 중첩된 구조의 상태들을 병합하는지 테스트"""
    # Given
    state = {"some": "data"}
    
    # 중첩된 구조를 가진 상태들을 TypedDict로 정의
    class MockNestedState1(TypedDict):
        nested: Dict[str, Dict[str, str]]
        simple: str
    
    class MockNestedState2(TypedDict):
        nested: Dict[str, Dict[str, str]]
        another: str
    
    # TypedDict 인스턴스 생성
    nested_states = [
        MockNestedState1,
        MockNestedState2
    ]
    
    # When
    alan_agent_start_node = create_alan_agent_start_node(nested_states)
    result = alan_agent_start_node(state)
    
    # Then
    assert result["metadata"]["initialization"] == True
    assert result["metadata"]["chat_status"] == "unavailable"
    # 중첩된 구조가 올바르게 병합되었는지 확인
    assert "nested" in result
    assert "simple" in result
    assert "another" in result


def test_alan_agent_finish_node_returns_correct_structure():
    """alan_agent_finish_node가 올바른 구조를 반환하는지 테스트"""
    # Given
    state = {"some": "data"}  # state는 사용되지 않지만 파라미터로 전달됨
    
    # When
    result = alan_agent_finish_node(state)
    
    # Then
    assert result["metadata"]["chat_status"] == "available"
    # state 파라미터가 결과에 영향을 주지 않는지 확인
    assert "some" not in result