import uuid
from datetime import datetime
from typing import Optional
from pydantic import Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage


def default_metadata_factory() -> dict:
    return {
        "rendering_option": "str",
    }



class BaseAlanMessage:
    """Mixin class that provides automatic UUID generation for message IDs."""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), coerce_numbers_to_str=True)
    metadata: dict = Field(default_factory=default_metadata_factory)


class AlanAIMessage(AIMessage, BaseAlanMessage):
    pass


class AlanHumanMessage(HumanMessage, BaseAlanMessage):
    pass


class AlanSystemMessage(SystemMessage, BaseAlanMessage):
    pass


class AlanToolMessage(ToolMessage, BaseAlanMessage):
    pass


def convert_to_alan_message(message: BaseMessage) -> BaseAlanMessage:
    """
    BaseMessage를 BaseAlanMessage로 변환하는 함수
    Pydantic 내부 속성은 제외하고 필요한 속성만 동적으로 복사
    """
    # AIMessage 인스턴스의 모든 속성을 동적으로 탐색하여 복사
    kwargs = {}
    
    # __dict__를 통해 모든 인스턴스 속성 접근
    for attr_name, attr_value in message.__dict__.items():
        # private 속성과 Pydantic 내부 속성 제외
        if (not attr_name.startswith('_') and 
            not attr_name.startswith('__pydantic') and
            attr_name not in ['__dict__', '__slots__']):
            kwargs[attr_name] = attr_value
    
    # __slots__를 통해 모든 슬롯 속성도 접근 (일부 클래스는 __slots__ 사용)
    if hasattr(message.__class__, '__slots__'):
        for slot_name in message.__class__.__slots__:
            if hasattr(message, slot_name):
                slot_value = getattr(message, slot_name)
                # Pydantic 내부 속성 제외
                if not slot_name.startswith('__pydantic'):
                    kwargs[slot_name] = slot_value
    
    # content는 필수 속성이므로 별도로 확인
    if 'content' not in kwargs:
        kwargs['content'] = message.content
    
    # ID가 None이거나 비어있는 경우 새로운 UUID 생성
    if not kwargs.get('id'):
        kwargs['id'] = str(uuid.uuid4())
    
    # AlanMessage 생성 (동적으로 수집된 모든 속성 사용)
    if isinstance(message, AIMessage):
        return AlanAIMessage(**kwargs)
    elif isinstance(message, HumanMessage):
        return AlanHumanMessage(**kwargs)
    elif isinstance(message, SystemMessage):
        return AlanSystemMessage(**kwargs)
    elif isinstance(message, ToolMessage):
        return AlanToolMessage(**kwargs)
    else:
        raise Exception(f"Unsupported message type: {type(message)}. Please Add Message Type to convert_to_alan_message function")

