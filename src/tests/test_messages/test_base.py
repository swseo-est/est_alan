import pytest
import uuid
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from estalan.messages.base import (
    BaseAlanMessage,
    AlanAIMessage,
    AlanHumanMessage,
    AlanSystemMessage,
    AlanToolMessage,
    BaseAlanBlockMessage
)


class TestBaseAlanMessage:
    """BaseAlanMessage 클래스를 테스트합니다."""
    
    def test_id_field_exists(self):
        """BaseAlanMessage가 id 필드를 가지고 있는지 테스트합니다."""
        
        # BaseAlanMessage를 직접 테스트하기 위해 AlanAIMessage를 사용
        message = AlanAIMessage(content="test")
        
        # 결과 검증
        assert hasattr(message, 'id')
        assert message.id is not None
        assert isinstance(message.id, str)
    
    def test_id_is_uuid_format(self):
        """생성된 id가 UUID 형식인지 테스트합니다."""
        
        message = AlanAIMessage(content="test")
        
        # UUID 형식 검증 (예외가 발생하지 않으면 유효한 UUID)
        try:
            uuid.UUID(message.id)
            is_valid_uuid = True
        except ValueError:
            is_valid_uuid = False
        
        assert is_valid_uuid
    
    def test_each_instance_has_unique_id(self):
        """각 인스턴스가 고유한 id를 가지는지 테스트합니다."""
        
        message1 = AlanAIMessage(content="test1")
        message2 = AlanAIMessage(content="test2")
        
        # 결과 검증
        assert message1.id != message2.id


class TestAlanAIMessage:
    """AlanAIMessage 클래스를 테스트합니다."""
    
    def test_inherits_from_aimessage(self):
        """AlanAIMessage가 AIMessage를 상속받는지 테스트합니다."""
        
        message = AlanAIMessage(content="AI response")
        
        # 결과 검증
        assert isinstance(message, AIMessage)
        assert isinstance(message, AlanAIMessage)
    
    def test_inherits_from_base_alan_message(self):
        """AlanAIMessage가 BaseAlanMessage를 상속받는지 테스트합니다."""
        
        message = AlanAIMessage(content="AI response")
        
        # 결과 검증
        assert hasattr(message, 'id')
        assert message.id is not None
    
    def test_content_property(self):
        """AlanAIMessage의 content 속성이 올바르게 설정되는지 테스트합니다."""
        
        content_text = "This is an AI response"
        message = AlanAIMessage(content=content_text)
        
        # 결과 검증
        assert message.content == content_text
    
    def test_with_additional_kwargs(self):
        """AlanAIMessage가 추가 키워드 인자를 받을 수 있는지 테스트합니다."""
        
        message = AlanAIMessage(
            content="AI response",
            additional_kwargs={"model": "gpt-4", "temperature": 0.7}
        )
        
        # 결과 검증
        assert message.content == "AI response"
        assert message.additional_kwargs["model"] == "gpt-4"
        assert message.additional_kwargs["temperature"] == 0.7


class TestAlanHumanMessage:
    """AlanHumanMessage 클래스를 테스트합니다."""
    
    def test_inherits_from_humanmessage(self):
        """AlanHumanMessage가 HumanMessage를 상속받는지 테스트합니다."""
        
        message = AlanHumanMessage(content="Human input")
        
        # 결과 검증
        assert isinstance(message, HumanMessage)
        assert isinstance(message, AlanHumanMessage)
    
    def test_inherits_from_base_alan_message(self):
        """AlanHumanMessage가 BaseAlanMessage를 상속받는지 테스트합니다."""
        
        message = AlanHumanMessage(content="Human input")
        
        # 결과 검증
        assert hasattr(message, 'id')
        assert message.id is not None
    
    def test_content_property(self):
        """AlanHumanMessage의 content 속성이 올바르게 설정되는지 테스트합니다."""
        
        content_text = "Hello, how are you?"
        message = AlanHumanMessage(content=content_text)
        
        # 결과 검증
        assert message.content == content_text


class TestAlanSystemMessage:
    """AlanSystemMessage 클래스를 테스트합니다."""
    
    def test_inherits_from_systemmessage(self):
        """AlanSystemMessage가 SystemMessage를 상속받는지 테스트합니다."""
        
        message = AlanSystemMessage(content="System instruction")
        
        # 결과 검증
        assert isinstance(message, SystemMessage)
        assert isinstance(message, AlanSystemMessage)
    
    def test_inherits_from_base_alan_message(self):
        """AlanSystemMessage가 BaseAlanMessage를 상속받는지 테스트합니다."""
        
        message = AlanSystemMessage(content="System instruction")
        
        # 결과 검증
        assert hasattr(message, 'id')
        assert message.id is not None
    
    def test_content_property(self):
        """AlanSystemMessage의 content 속성이 올바르게 설정되는지 테스트합니다."""
        
        content_text = "You are a helpful assistant"
        message = AlanSystemMessage(content=content_text)
        
        # 결과 검증
        assert message.content == content_text


class TestAlanToolMessage:
    """AlanToolMessage 클래스를 테스트합니다."""
    
    def test_inherits_from_toolmessage(self):
        """AlanToolMessage가 ToolMessage를 상속받는지 테스트합니다."""
        
        message = AlanToolMessage(content="Tool result", tool_call_id="tool123")
        
        # 결과 검증
        assert isinstance(message, ToolMessage)
        assert isinstance(message, AlanToolMessage)
    
    def test_inherits_from_base_alan_message(self):
        """AlanToolMessage가 BaseAlanMessage를 상속받는지 테스트합니다."""
        
        message = AlanToolMessage(content="Tool result", tool_call_id="tool123")
        
        # 결과 검증
        assert hasattr(message, 'id')
        assert message.id is not None
    
    def test_content_and_tool_call_id(self):
        """AlanToolMessage의 content와 tool_call_id가 올바르게 설정되는지 테스트합니다."""
        
        content_text = "Tool execution completed"
        tool_id = "tool456"
        message = AlanToolMessage(content=content_text, tool_call_id=tool_id)
        
        # 결과 검증
        assert message.content == content_text
        assert message.tool_call_id == tool_id


class TestBaseAlanBlockMessage:
    """BaseAlanBlockMessage 클래스를 테스트합니다."""
    
    def test_inherits_from_alan_ai_message(self):
        """BaseAlanBlockMessage가 AlanAIMessage를 상속받는지 테스트합니다."""
        
        message = BaseAlanBlockMessage(content="code content")
        
        # 결과 검증
        assert isinstance(message, AlanAIMessage)
        assert isinstance(message, BaseAlanBlockMessage)
    
    def test_inherits_from_base_alan_message(self):
        """BaseAlanBlockMessage가 BaseAlanMessage를 상속받는지 테스트합니다."""
        
        message = BaseAlanBlockMessage(content="code content")
        
        # 결과 검증
        assert hasattr(message, 'id')
        assert message.id is not None
    
    def test_block_tag_field_exists(self):
        """BaseAlanBlockMessage가 block_tag 필드를 가지고 있는지 테스트합니다."""
        
        message = BaseAlanBlockMessage(content="code content")
        
        # 결과 검증
        assert hasattr(message, 'block_tag')
    
    def test_init_with_content_only(self):
        """BaseAlanBlockMessage.__init__()이 content만으로 올바르게 초기화되는지 테스트합니다."""
        
        content_text = "print('Hello World')"
        message = BaseAlanBlockMessage(content=content_text)
        
        # 결과 검증
        assert message.content == f"```\n{content_text}\n```"
        assert message.block_tag is None
        assert message.id is not None
    
    def test_init_with_content_and_block_tag(self):
        """BaseAlanBlockMessage.__init__()이 content와 block_tag로 올바르게 초기화되는지 테스트합니다."""
        
        content_text = "def hello():"
        block_tag = "python"
        message = BaseAlanBlockMessage(content=content_text, block_tag=block_tag)
        
        # 결과 검증
        assert message.content == f"```{block_tag}\n{content_text}\n```"
        assert message.block_tag == block_tag
        assert message.id is not None
    
    def test_init_with_none_content(self):
        """BaseAlanBlockMessage.__init__()이 None content로 올바르게 초기화되는지 테스트합니다."""
        
        message = BaseAlanBlockMessage(content=None)
        
        # 결과 검증
        assert message.content == "```\nNone\n```"
        assert message.block_tag is None
    
    def test_init_with_empty_content(self):
        """BaseAlanBlockMessage.__init__()이 빈 content로 올바르게 초기화되는지 테스트합니다."""
        
        message = BaseAlanBlockMessage(content="")
        
        # 결과 검증
        assert message.content == "```\n\n```"
        assert message.block_tag is None
    
    def test_init_with_additional_kwargs(self):
        """BaseAlanBlockMessage.__init__()이 추가 키워드 인자를 받을 수 있는지 테스트합니다."""
        
        message = BaseAlanBlockMessage(
            content="code",
            block_tag="python",
            additional_kwargs={"model": "gpt-4"}
        )
        
        # 결과 검증
        assert message.content == "```python\ncode\n```"
        assert message.block_tag == "python"
        assert message.additional_kwargs["model"] == "gpt-4"
    
    def test_process_content_without_block_tag(self):
        """BaseAlanBlockMessage._process_content()가 block_tag 없이 올바르게 작동하는지 테스트합니다."""
        
        message = BaseAlanBlockMessage(content="dummy")  # 초기화용
        result = message._process_content("test content")
        
        # 결과 검증
        assert result == "```\ntest content\n```"
    
    def test_process_content_with_block_tag(self):
        """BaseAlanBlockMessage._process_content()가 block_tag와 함께 올바르게 작동하는지 테스트합니다."""
        
        message = BaseAlanBlockMessage(content="dummy")  # 초기화용
        result = message._process_content("def main():", "python")
        
        # 결과 검증
        assert result == "```python\ndef main():\n```"
    
    def test_process_content_with_none_block_tag(self):
        """BaseAlanBlockMessage._process_content()가 None block_tag로 올바르게 작동하는지 테스트합니다."""
        
        message = BaseAlanBlockMessage(content="dummy")  # 초기화용
        result = message._process_content("content", None)
        
        # 결과 검증
        assert result == "```\ncontent\n```"
    
    def test_process_content_with_multiline_content(self):
        """BaseAlanBlockMessage._process_content()가 여러 줄 content를 올바르게 처리하는지 테스트합니다."""
        
        multiline_content = "def hello():\n    print('Hello')\n    return True"
        message = BaseAlanBlockMessage(content="dummy")  # 초기화용
        result = message._process_content(multiline_content, "python")
        
        # 결과 검증
        expected = f"```python\n{multiline_content}\n```"
        assert result == expected
    
    def test_process_content_with_various_types(self):
        """BaseAlanBlockMessage._process_content()가 다양한 타입의 content를 처리하는지 테스트합니다."""
        
        message = BaseAlanBlockMessage(content="dummy")  # 초기화용
        
        # 숫자 타입
        result_int = message._process_content(42, "number")
        assert result_int == "```number\n42\n```"
        
        # 리스트 타입
        result_list = message._process_content([1, 2, 3], "json")
        assert result_list == "```json\n[1, 2, 3]\n```"
        
        # 딕셔너리 타입
        result_dict = message._process_content({"key": "value"}, "json")
        assert result_dict == "```json\n{'key': 'value'}\n```"
