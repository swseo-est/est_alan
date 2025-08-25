import uuid
from typing import Optional, Any, Literal
from pydantic import Field, BaseModel, ConfigDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage
from logging import getLogger

class AlanMessageMetadata(BaseModel):
    """Alan 메시지의 메타데이터를 정의하는 모델"""
    model_config = ConfigDict(frozen=True)

    rendering_option: Literal["str", "json", "html"] = "str"
    log_level: Literal["info", "error", "debug", "warning"] = "info"


def default_metadata_factory() -> dict:
    """기본 메타데이터 팩토리 함수"""
    return {
        "rendering_option": "str",
        "log_level": "info",
    }


class BaseAlanMessage:
    """Alan 메시지의 기본 클래스 - UUID 자동 생성 및 메타데이터 제공"""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), coerce_numbers_to_str=True)
    metadata: AlanMessageMetadata = Field(default_factory=default_metadata_factory)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AlanAIMessage(AIMessage, BaseAlanMessage):
    """Alan AI 메시지 클래스"""
    pass


class AlanHumanMessage(HumanMessage, BaseAlanMessage):
    """Alan Human 메시지 클래스"""
    pass


class AlanSystemMessage(SystemMessage, BaseAlanMessage):
    """Alan System 메시지 클래스"""
    pass


class AlanToolMessage(ToolMessage, BaseAlanMessage):
    """Alan Tool 메시지 클래스"""
    pass


class BaseAlanBlockMessage(AlanAIMessage):
    """Alan 블록 메시지의 기본 클래스"""
    block_tag: Optional[str] = Field(default=None, coerce_numbers_to_str=True)

    def _process_content(self, content: Any, block_tag: Optional[str] = None) -> str:
        """
        content를 코드블록으로 후처리하는 메서드
        
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

    def __init__(self, content: Any = None, block_tag: Optional[str] = None, **kwargs):
        processed_content = self._process_content(content, block_tag)
        # Pydantic 모델과 호환되도록 super().__init__ 호출 후 block_tag 설정
        super().__init__(content=processed_content, **kwargs)
        # block_tag는 이미 Field로 정의되어 있으므로 kwargs를 통해 전달되어야 함
        if block_tag is not None:
            object.__setattr__(self, 'block_tag', block_tag)


def convert_to_alan_message(message: BaseMessage) -> BaseAlanMessage:
    """
    BaseMessage를 BaseAlanMessage로 변환하는 함수
    필요한 속성만 선택적으로 복사하여 안전하게 변환
    """

    if isinstance(message, BaseAlanMessage):
        return message

    if not isinstance(message, BaseMessage):
        raise ValueError(f"{message} is not BaseMessage")

    # 필요한 기본 속성들만 선택적으로 복사
    kwargs = {}

    # content는 필수 속성
    kwargs['content'] = message.content

    # 안전한 속성들만 복사 (명시적으로 허용된 속성들)
    safe_attributes = [
        'id', 'name', 'additional_kwargs', 'response_metadata',
        'tool_calls', 'tool_call_id', 'example'
    ]

    for attr_name in safe_attributes:
        if hasattr(message, attr_name):
            attr_value = getattr(message, attr_name)
            if attr_value is not None:
                kwargs[attr_name] = attr_value

    # ID가 None이거나 비어있는 경우 새로운 UUID 생성
    if not kwargs.get('id'):
        kwargs['id'] = str(uuid.uuid4())

    # AlanMessage 생성
    if isinstance(message, AIMessage):
        return AlanAIMessage(**kwargs)
    elif isinstance(message, HumanMessage):
        return AlanHumanMessage(**kwargs)
    elif isinstance(message, SystemMessage):
        return AlanSystemMessage(**kwargs)
    elif isinstance(message, ToolMessage):
        return AlanToolMessage(**kwargs)
    else:
        raise ValueError(f"Unsupported message type: {type(message)}. Please Add Message Type to convert_to_alan_message function")
