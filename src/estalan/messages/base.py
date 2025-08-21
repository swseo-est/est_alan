import uuid
from typing import Optional, Any
from pydantic import Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage


class BaseAlanMessage:
    """Mixin class that provides automatic UUID generation for message IDs."""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), coerce_numbers_to_str=True)


class AlanAIMessage(AIMessage, BaseAlanMessage):
    pass


class AlanHumanMessage(HumanMessage, BaseAlanMessage):
    pass


class AlanSystemMessage(SystemMessage, BaseAlanMessage):
    pass


class AlanToolMessage(ToolMessage, BaseAlanMessage):
    pass


class BaseAlanBlockMessage(AlanAIMessage, BaseAlanMessage):
    block_tag: Optional[str] = Field(default=None, coerce_numbers_to_str=True)
    
    def __init__(self, content: Any = None, block_tag: Optional[str] = None, **kwargs):
        # content를 후처리하는 메서드
        processed_content = self._process_content(content, block_tag)
        
        # AIMessage의 __init__을 호출하여 처리된 content를 전달
        super().__init__(content=processed_content, **kwargs)
    
    def _process_content(self, content: Any, block_tag: Optional[str] = None) -> Any:
        """
        content를 후처리하는 메서드입니다.
        하위 클래스에서 이 메서드를 오버라이드하여 원하는 후처리 로직을 구현할 수 있습니다.
        
        Args:
            content: 원본 content
            block_tag: 블록 태그 (선택사항)
            
        Returns:
            후처리된 content (코드블록으로 감싸짐)
        """
        # content를 코드블록으로 감싸기
        if block_tag:
            return f"```{block_tag}\n{content}\n```"
        else:
            return f"```\n{content}\n```"
