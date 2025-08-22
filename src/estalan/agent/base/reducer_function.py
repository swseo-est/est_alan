from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage
from typing import Union, List, Any

from estalan.messages.base import convert_to_alan_message

Messages = Union[List[BaseMessage], BaseMessage]

def add_messages_for_alan(left: Messages, right: Messages) -> Messages:
    """
    메시지를 Alan 메시지로 변환한 후 add_messages를 호출하는 wrapping 함수
    
    Args:
        left: 기존 메시지들 (Messages 타입)
        right: 새로 추가할 메시지들 (Messages 타입)
    
    Returns:
        Alan 메시지로 변환된 후 add_messages로 처리된 결과
    """
    # left와 right를 리스트로 변환
    left_list = left if isinstance(left, list) else [left]
    right_list = right if isinstance(right, list) else [right]
    
    # right에 있는 메시지들을 Alan 메시지로 변환
    processed_right = []
    for message in right_list:
        # 메시지 타입에 따라 적절한 Alan 메시지로 변환
        alan_message = convert_to_alan_message(message)
        processed_right.append(alan_message)
    
    # Alan 메시지로 변환된 메시지들을 기존 add_messages로 처리
    return add_messages(left_list, processed_right)