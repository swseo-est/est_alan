from typing import Optional
from langchain_core.messages import BaseMessage
from estalan.messages.base import AlanHumanMessage, AlanSystemMessage, AlanAIMessage, AlanToolMessage, BaseAlanBlockMessage
from estalan.messages.format.chat_html import create_img_grid

def create_message(message_type: str, content: str, allow_duplicate: bool = False, *args, **kwargs) -> BaseMessage:
    # allow_duplicate를 kwargs에 포함
    kwargs['allow_duplicate'] = allow_duplicate
    
    # name이 있고 allow_duplicate가 True인 경우 name에 id를 추가
    if 'name' in kwargs and kwargs['name'] and allow_duplicate:
        import uuid
        original_name = kwargs['name']
        # id가 kwargs에 있으면 사용하고, 없으면 새로 생성
        message_id = kwargs.get('id', str(uuid.uuid4()))
        kwargs['name'] = f"{original_name}-{message_id}"
        # id가 kwargs에 없었다면 추가
        if 'id' not in kwargs:
            kwargs['id'] = message_id
    
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


def create_ai_message(content: str, *args, **kwargs) -> AlanAIMessage:
    message = create_message("ai", content, **kwargs)
    return message


def create_human_message(content: str, *args, **kwargs) -> AlanHumanMessage:
    message = create_message("human", content, **kwargs)
    return message


def create_system_message(content: str, *args, **kwargs) -> AlanSystemMessage:
    message = create_message("system", content, **kwargs)
    return message


def create_tool_message(content: str, *args, **kwargs) -> AlanToolMessage:
    message = create_message("tool", content, **kwargs)
    return message


def create_block_message(content: str, block_tag: Optional[str] = None, *args, **kwargs) -> BaseAlanBlockMessage:
    message = create_message("block", content, block_tag=block_tag, **kwargs)
    return message


def create_image_grid_message(list_image, num_cols=3, *args, **kwargs):
    html_code = create_img_grid(list_image, num_cols)
    msg = create_ai_message(content=html_code, metadata={"rendering_option": "html"}, **kwargs)
    return msg