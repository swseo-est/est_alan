import json


def load_config_json(filename: str) -> dict:
    """JSON 파일을 읽어 dict 로 반환"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)