from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage
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
    right_list = add_messages([], right_list)
    
    # right에 있는 메시지들을 Alan 메시지로 변환
    processed_right = []
    for message in right_list:
        # 메시지 타입에 따라 적절한 Alan 메시지로 변환
        alan_message = convert_to_alan_message(message)
        processed_right.append(alan_message)

    merged = merge_message(left_list, processed_right)
    return merged

def merge_message(left: Messages, right: Messages) -> Messages:
    # merge
    merged = left.copy()
    merged_by_id = {m.id: i for i, m in enumerate(merged)}
    ids_to_remove = set()
    for m in right:
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            if isinstance(m, RemoveMessage):
                ids_to_remove.add(m.id)
            else:
                ids_to_remove.discard(m.id)
                merged[existing_idx] = m
        else:
            if isinstance(m, RemoveMessage):
                raise ValueError(
                    f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
                )

            merged_by_id[m.id] = len(merged)
            merged.append(m)
    merged = [m for m in merged if m.id not in ids_to_remove]
    return merged



def update_metadata(metadata_new: dict, metadata: dict):
    for key, value in metadata_new.items():
        metadata[key] = value

    return metadata