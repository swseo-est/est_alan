import pytest
from unittest.mock import patch
from estalan.agent.base.node import alan_agent_start_node, alan_agent_finish_node
from estalan.messages.utils import create_ai_message


def test_alan_agent_start_node_sets_chat_status_unavailable():
    """alan_agent_start_node가 chat_status를 'unavailable'로 설정하는지 테스트"""
    # Given
    state = {"some": "data"}
    
    # When
    result = alan_agent_start_node(state)
    
    # Then
    assert result["metadata"]["chat_status"] == "unavailable"


def test_alan_agent_finish_node_sets_chat_status_available():
    """alan_agent_finish_node가 chat_status를 'available'로 설정하는지 테스트"""
    # Given
    state = {"some": "data"}
    
    # When
    result = alan_agent_finish_node(state)
    
    # Then
    assert result["metadata"]["chat_status"] == "available"
