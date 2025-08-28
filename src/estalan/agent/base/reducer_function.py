from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage
from typing import Union, List, Any

from estalan.messages.base import convert_to_alan_message
from estalan.logging.base import get_logger

# 로거 초기화
logger = get_logger(__name__)

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
    logger.debug("add_messages_for_alan 실행 시작", 
                left_count=len(left if isinstance(left, list) else [left]),
                right_count=len(right if isinstance(right, list) else [right]))
    
    # left와 right를 리스트로 변환
    left_list = left if isinstance(left, list) else [left]
    right_list = right if isinstance(right, list) else [right]
    right_list = add_messages([], right_list)
    
    # 새로 들어온 메시지들을 로깅
    logger.info("새로 추가될 메시지들", message_count=len(right_list))
    for i, message in enumerate(right_list):
        message_type = message.__class__.__name__
        content = getattr(message, 'content', '내용 없음')
        
        # Alan 메시지인지 확인
        is_alan_message = hasattr(message, 'metadata') and hasattr(message.metadata, 'get')
        alan_info = ""
        if is_alan_message:
            rendering_option = message.metadata.get('rendering_option', 'N/A')
            log_level = message.metadata.get('log_level', 'N/A')
            alan_info = f" (Alan: {rendering_option}/{log_level})"
        
        # content가 너무 길면 잘라서 로깅
        if isinstance(content, str) and len(content) > 100:
            content = content[:100] + "..."
        
        logger.info(f"[{message_type}]{alan_info} {content}", 
                   message_index=i, 
                   message_id=getattr(message, 'id', 'ID 없음'),
                   is_alan_message=is_alan_message,
                   rendering_option=message.metadata.get('rendering_option', 'N/A') if is_alan_message else 'N/A',
                   log_level=message.metadata.get('log_level', 'N/A') if is_alan_message else 'N/A')
    
    # right에 있는 메시지들을 Alan 메시지로 변환
    processed_right = []
    for message in right_list:
        # 메시지 타입에 따라 적절한 Alan 메시지로 변환
        alan_message = convert_to_alan_message(message)
        processed_right.append(alan_message)
        
        # Alan 메시지 변환 결과 상세 로깅
        logger.debug("Alan 메시지로 변환 완료", 
                    original_type=message.__class__.__name__,
                    alan_type=alan_message.__class__.__name__,
                    original_id=getattr(message, 'id', 'ID 없음'),
                    alan_id=getattr(alan_message, 'id', 'ID 없음'),
                    rendering_option=alan_message.metadata.get('rendering_option', 'N/A'),
                    log_level=alan_message.metadata.get('log_level', 'N/A'))

    merged = merge_message(left_list, processed_right)
    logger.info("메시지 병합 완료", 
               original_count=len(left_list), 
               new_count=len(processed_right), 
               final_count=len(merged))
    
    return merged

def merge_message(left: Messages, right: Messages) -> Messages:
    # merge
    logger.debug("merge_message 실행 시작", left_count=len(left), right_count=len(right))
    
    merged = left.copy()
    merged_by_id = {m.id: i for i, m in enumerate(merged)}
    ids_to_remove = set()
    
    for m in right:
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            if isinstance(m, RemoveMessage):
                ids_to_remove.add(m.id)
                logger.debug("메시지 제거 예정", message_id=m.id, message_type="RemoveMessage")
            else:
                ids_to_remove.discard(m.id)
                merged[existing_idx] = m
                logger.debug("기존 메시지 업데이트", message_id=m.id, existing_index=existing_idx)
        else:
            if isinstance(m, RemoveMessage):
                error_msg = f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
                logger.error("존재하지 않는 메시지 삭제 시도", message_id=m.id)
                raise ValueError(error_msg)

            merged_by_id[m.id] = len(merged)
            merged.append(m)
            logger.debug("새 메시지 추가", message_id=m.id, message_type=m.__class__.__name__)
    
    merged = [m for m in merged if m.id not in ids_to_remove]
    logger.debug("merge_message 완료", final_count=len(merged), removed_count=len(ids_to_remove))
    return merged



def update_metadata(metadata_old: dict, metadata_new: dict):
    logger.debug("메타데이터 업데이트 시작", 
                old_keys=list(metadata_old.keys()), 
                new_keys=list(metadata_new.keys()))
    
    for key, value in metadata_new.items():
        metadata_old[key] = value
        logger.debug("메타데이터 키 업데이트", key=key, value=value)

    logger.debug("메타데이터 업데이트 완료", final_keys=list(metadata_old.keys()))
    return metadata_old