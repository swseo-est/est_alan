import re
from typing import Optional
from langchain_core.messages import BaseMessage
from estalan.messages.base import AlanHumanMessage, AlanSystemMessage, AlanAIMessage, AlanToolMessage, BaseAlanBlockMessage
from estalan.messages.format.chat_html import create_img_grid

def create_message(message_type: str, content: str, metadata: Optional[dict] = None, name: Optional[str] = None, *args, **kwargs) -> BaseMessage:
    """
    메시지를 생성하는 함수
    
    Args:
        message_type: 메시지 타입 ("human", "system", "ai", "tool", "block")
        content: 메시지 내용
        metadata: 메시지 메타데이터 (선택사항)
            - rendering_option: "str", "json", "html" 중 하나
            - log_level: "info", "error", "debug", "warning" 중 하나
            - 기타 사용자 정의 필드들
        name: 메시지 이름 (공백, <, |, \, /, > 문자를 포함할 수 없음)
        *args, **kwargs: 추가 매개변수들
    """
    # name 검증 로직 추가
    if name is not None:
        # OpenAI에서 정한 정규식 규칙: 공백, <, |, \, /, > 문자를 포함하지 않는지 검증
        # 해당 형식을 지키지 않으면 OpenAI에서 에러가 발생함
        invalid_pattern = r'[\s<|\\/>]'
        if re.search(invalid_pattern, name):
            raise ValueError(f"잘못된 이름 '{name}': 이름에는 공백, <, |, \\, /, > 문자가 포함될 수 없습니다")
    
    # metadata가 정의되지 않으면 기본값 설정
    if metadata is None:
        metadata = {}
    
    # 필수 키가 누락된 경우 기본값으로 채움
    default_metadata = {
        "rendering_option": "str",
        "log_level": "info"
    }
    
    # 사용자 metadata와 기본값 병합 (사용자 값이 우선)
    for key, default_value in default_metadata.items():
        if key not in metadata:
            metadata[key] = default_value
    
    # metadata를 kwargs에 명시적으로 추가
    kwargs['metadata'] = metadata
    
    if message_type == "human":
        return AlanHumanMessage(content=content, **kwargs)
    elif message_type == "system":
        return AlanSystemMessage(content=content, **kwargs)
    elif message_type == "ai":
        return AlanAIMessage(content=content, **kwargs)
    elif message_type == "tool":
        return AlanToolMessage(content=content, **kwargs)
    elif message_type == "block":
        return BaseAlanBlockMessage(content=content, **kwargs)
    else:
        raise ValueError(f"Invalid message type: {message_type}")   


def create_ai_message(content: str, metadata: Optional[dict] = None, *args, **kwargs) -> AlanAIMessage:
    message = create_message("ai", content, metadata=metadata, **kwargs)
    return message


def create_human_message(content: str, metadata: Optional[dict] = None, *args, **kwargs) -> AlanHumanMessage:
    message = create_message("human", content, metadata=metadata, **kwargs)
    return message


def create_system_message(content: str, metadata: Optional[dict] = None, *args, **kwargs) -> AlanSystemMessage:
    message = create_message("system", content, metadata=metadata, **kwargs)
    return message


def create_tool_message(content: str, metadata: Optional[dict] = None, *args, **kwargs) -> AlanToolMessage:
    message = create_message("tool", content, metadata=metadata, **kwargs)
    return message


def create_block_message(content: str, block_tag: Optional[str] = None, metadata: Optional[dict] = None, *args, **kwargs) -> BaseAlanBlockMessage:
    message = create_message("block", content, block_tag=block_tag, metadata=metadata, **kwargs)
    return message


def create_image_grid_message(list_image, num_cols=3, metadata: Optional[dict] = None, *args, **kwargs):
    html_code = create_img_grid(list_image, num_cols)
    # 기본 metadata와 사용자 제공 metadata 병합
    default_metadata = {"rendering_option": "html"}
    if metadata:
        default_metadata.update(metadata)
    msg = create_ai_message(content=html_code, metadata=default_metadata, **kwargs)
    return msg