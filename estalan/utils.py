import json

from langchain_core.messages import ToolMessage


def load_config_json(filename: str) -> dict:
    """JSON 파일을 읽어 dict 로 반환"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_last_tool_message(llm_response) -> ToolMessage:
    for msg in llm_response['messages'][::-1]:
        if type(msg) == ToolMessage:
            return msg

    return None
