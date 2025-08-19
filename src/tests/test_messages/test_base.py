"""Test automatic UUID generation in Alan messages."""

import pytest
import re
from typing import Set

from estalan.messages.base import (
    AlanAIMessage,
    AlanHumanMessage,
    AlanSystemMessage,
    AlanToolMessage,
)


class TestUUIDAutoGeneration:
    """Test automatic UUID generation functionality."""

    def test_basic_id_generation(self):
        """Test that all message types generate IDs automatically."""
        messages = [
            AlanAIMessage(content="AI message"),
            AlanHumanMessage(content="Human message"),
            AlanSystemMessage(content="System message"),
            AlanToolMessage(content="Tool message", tool_call_id="test-tool"),
        ]

        for msg in messages:
            assert msg.id is not None
            assert isinstance(msg.id, str)
            assert len(msg.id) == 36  # UUID length

    def test_uuid_format(self):
        """Test that generated IDs follow UUID format."""
        msg = AlanAIMessage(content="Test message")
        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        
        assert re.match(uuid_pattern, msg.id) is not None

    def test_id_uniqueness(self):
        """Test that generated IDs are unique."""
        ids: Set[str] = set()
        
        for i in range(20):
            msg = AlanAIMessage(content=f"Message {i}")
            ids.add(msg.id)
        
        assert len(ids) == 20, "All generated IDs should be unique"

    def test_explicit_id_assignment(self):
        """Test that explicit ID assignment works correctly."""
        custom_id = "custom-id-123"
        msg = AlanAIMessage(content="Test", id=custom_id)
        
        assert msg.id == custom_id

    def test_mixin_inheritance(self):
        """Test that the mixin pattern works correctly."""
        # Test that BaseAlanMessage is a mixin (not inheriting from BaseMessage)
        from estalan.messages.base import BaseAlanMessage
        
        # BaseAlanMessage should not inherit from BaseMessage
        from langchain_core.messages import BaseMessage
        assert not issubclass(BaseAlanMessage, BaseMessage)
        
        # But AlanAIMessage should inherit from both AIMessage and BaseAlanMessage
        from langchain_core.messages import AIMessage
        assert issubclass(AlanAIMessage, AIMessage)
        assert issubclass(AlanAIMessage, BaseAlanMessage)

    def test_all_message_types_have_ids(self):
        """Test that all Alan message types have automatic ID generation."""
        # Test regular message types
        regular_message_types = [
            AlanAIMessage,
            AlanHumanMessage,
            AlanSystemMessage,
        ]
        
        for msg_class in regular_message_types:
            msg = msg_class(content="Test content")
            assert msg.id is not None
            assert isinstance(msg.id, str)
            assert len(msg.id) == 36
        
        # Test ToolMessage separately since it requires tool_call_id
        tool_msg = AlanToolMessage(content="Test content", tool_call_id="test-tool")
        assert tool_msg.id is not None
        assert isinstance(tool_msg.id, str)
        assert len(tool_msg.id) == 36

    def test_id_persistence(self):
        """Test that ID remains the same after object creation."""
        msg = AlanAIMessage(content="Test message")
        original_id = msg.id
        
        # Access the ID multiple times
        assert msg.id == original_id
        assert msg.id == original_id
        assert msg.id == original_id

    def test_multiple_instances_different_ids(self):
        """Test that multiple instances have different IDs."""
        msg1 = AlanAIMessage(content="Message 1")
        msg2 = AlanAIMessage(content="Message 2")
        msg3 = AlanAIMessage(content="Message 3")
        
        ids = {msg1.id, msg2.id, msg3.id}
        assert len(ids) == 3, "Each message should have a unique ID"

    def test_tool_message_with_tool_call_id(self):
        """Test that ToolMessage works correctly with tool_call_id."""
        msg = AlanToolMessage(
            content="Tool result",
            tool_call_id="test-tool-call-123"
        )
        
        assert msg.id is not None
        assert isinstance(msg.id, str)
        assert len(msg.id) == 36
        assert msg.tool_call_id == "test-tool-call-123"

    def test_id_field_attributes(self):
        """Test that the id field has correct attributes."""
        msg = AlanAIMessage(content="Test")
        
        # Check that id is a string
        assert isinstance(msg.id, str)
        
        # Check that id is not empty
        assert msg.id.strip() != ""
        
        # Check that id contains only valid UUID characters
        assert all(c in "0123456789abcdef-" for c in msg.id.lower())

    @pytest.mark.parametrize("message_class,content", [
        (AlanAIMessage, "AI content"),
        (AlanHumanMessage, "Human content"),
        (AlanSystemMessage, "System content"),
        (AlanToolMessage, "Tool content"),
    ])
    def test_parametrized_id_generation(self, message_class, content):
        """Test ID generation with parametrized message types."""
        if message_class == AlanToolMessage:
            msg = message_class(content=content, tool_call_id="test-tool")
        else:
            msg = message_class(content=content)
        
        assert msg.id is not None
        assert isinstance(msg.id, str)
        assert len(msg.id) == 36
        
        # Verify UUID format
        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        assert re.match(uuid_pattern, msg.id) is not None

    def test_id_generation_performance(self):
        """Test that ID generation is fast enough for practical use."""
        import time
        
        start_time = time.time()
        messages = [AlanAIMessage(content=f"Message {i}") for i in range(100)]
        end_time = time.time()
        
        # Should complete within 1 second
        assert end_time - start_time < 1.0
        
        # All IDs should be unique
        ids = {msg.id for msg in messages}
        assert len(ids) == 100

    def test_id_with_special_characters(self):
        """Test ID generation with messages containing special characters."""
        special_content = "Message with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        msg = AlanAIMessage(content=special_content)
        
        assert msg.id is not None
        assert isinstance(msg.id, str)
        assert len(msg.id) == 36

    def test_id_with_unicode_content(self):
        """Test ID generation with Unicode content."""
        unicode_content = "한글 메시지 with emoji 🚀 and special chars ñáéíóú"
        msg = AlanAIMessage(content=unicode_content)
        
        assert msg.id is not None
        assert isinstance(msg.id, str)
        assert len(msg.id) == 36
