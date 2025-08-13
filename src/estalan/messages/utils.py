from langchain_core.messages import BaseMessage
from estalan.messages.base import AlanHumanMessage, AlanSystemMessage, AlanAIMessage, AlanToolMessage


def create_message(message_type: str, content: str, *args, **kwargs) -> BaseMessage:
    if message_type == "human":
        return AlanHumanMessage(content=content, *args, **kwargs)
    elif message_type == "system":
        return AlanSystemMessage(content=content, *args, **kwargs)
    elif message_type == "ai":
        return AlanAIMessage(content=content, *args, **kwargs)
    elif message_type == "tool":
        return AlanToolMessage(content=content, *args, **kwargs)
    else:
        raise ValueError(f"Invalid message type: {message_type}")   


def create_ai_message(content: str, *args, **kwargs) -> AlanAIMessage:
    message = create_message("ai", content, *args, **kwargs)
    return message


def create_human_message(content: str, *args, **kwargs) -> AlanHumanMessage:
    message = create_message("human", content, *args, **kwargs)
    return message


def create_system_message(content: str, *args, **kwargs) -> AlanSystemMessage:
    message = create_message("system", content, *args, **kwargs)
    return message


def create_tool_message(content: str, *args, **kwargs) -> AlanToolMessage:
    message = create_message("tool", content, *args, **kwargs)
    return message
