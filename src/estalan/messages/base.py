import uuid
from typing import Optional, Any, Literal
from typing_extensions import TypedDict
from pydantic import Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage
from logging import getLogger

class AlanMessageMetadata(TypedDict):
    """Alan 메시지의 메타데이터를 정의하는 모델"""
    rendering_option: Literal["str", "json", "html"]
    log_level: Literal["info", "error", "debug", "warning"]


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
        # 이미 코드블록으로 감싸져 있는지 확인 (더 정확한 감지)
        content_str = str(content)
        stripped_content = content_str.strip()
        
        # 여러 가지 중복 감싸기 패턴 확인
        is_already_wrapped = (
            (stripped_content.startswith('```') and stripped_content.endswith('```')) or
            (stripped_content.count('```') >= 4) or  # ```가 4개 이상 있으면 중복 감싸기 의심
            ('\n```\n' in stripped_content and stripped_content.count('```') >= 2)  # 내부에 ```가 있고 전체적으로도 감싸져 있음
        )
        
        logger = getLogger(__name__)
        logger.debug("BaseAlanBlockMessage._process_content 실행",
                    content_length=len(content_str),
                    block_tag=block_tag,
                    is_already_wrapped=is_already_wrapped,
                    backtick_count=stripped_content.count('```'),
                    has_internal_backticks='\n```\n' in stripped_content,
                    content_preview=content_str[:100] + "..." if len(content_str) > 100 else content_str)
        
        # 이미 코드블록으로 감싸져 있으면 그대로 반환
        if is_already_wrapped:
            logger.debug("이미 코드블록으로 감싸져 있음 - 중복 감싸기 방지")
            return content_str
        
        # content를 코드블록으로 감싸기
        if block_tag:
            result = f"```{block_tag}\n{content}\n```"
        else:
            result = f"```\n{content}\n```"
        
        logger.debug("코드블록 감싸기 완료", 
                    result_length=len(result),
                    result_preview=result[:100] + "..." if len(result) > 100 else result)
        
        return result

    def __init__(self, content: Any = None, block_tag: Optional[str] = None, **kwargs):
        # block_tag를 kwargs에서 제거하여 중복 처리 방지
        kwargs_without_block_tag = {k: v for k, v in kwargs.items() if k != 'block_tag'}
        
        processed_content = self._process_content(content, block_tag)
        # Pydantic 모델과 호환되도록 super().__init__ 호출
        super().__init__(content=processed_content, **kwargs_without_block_tag)
        
        # block_tag는 이미 Field로 정의되어 있으므로 kwargs를 통해 전달되어야 함
        if block_tag is not None:
            object.__setattr__(self, 'block_tag', block_tag)


def convert_to_alan_message(message: BaseMessage) -> BaseAlanMessage:
    """
    BaseMessage를 BaseAlanMessage로 변환하는 함수
    필요한 속성만 선택적으로 복사하여 안전하게 변환
    """

    if isinstance(message, BaseAlanMessage) or isinstance(message, ToolMessage) or isinstance(message, SystemMessage):
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

    # tool_calls가 있고 실제 내용이 있는 AI 메시지의 경우 log_level을 debug로 설정
    if isinstance(message, AIMessage) and kwargs.get('tool_calls'):
        tool_calls = kwargs['tool_calls']
        # tool_calls가 리스트이고 실제 내용이 있는지 확인
        if isinstance(tool_calls, list) and len(tool_calls) > 0:
            # metadata가 이미 있는 경우 기존 값 유지, 없는 경우 새로 생성
            if 'metadata' not in kwargs:
                kwargs['metadata'] = default_metadata_factory()
            # log_level만 debug로 변경하고 나머지는 기본값 유지
            kwargs['metadata']['log_level'] = 'debug'

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
