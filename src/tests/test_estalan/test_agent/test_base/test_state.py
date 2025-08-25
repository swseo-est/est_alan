import pytest
from pydantic import ValidationError
from typing import TypedDict, List

from estalan.agent.base.state import (
    AlanAgentMetaData,
    BaseAlanAgentState,
    Canvas,
    AlanAgentStateWithCanvas,
    create_default_state,
    state_to_json,
    state_to_json_pretty,
    state_to_json_compact
)
from estalan.agent.base.reducer_function import add_messages_for_alan
from langchain_core.messages import HumanMessage


def test_alan_agent_metadata_default_values():
    """AlanAgentMetaData의 기본값이 올바르게 설정되는지 테스트"""
    metadata = AlanAgentMetaData(
        chat_status="available",
        status="start"
    )
    assert metadata["chat_status"] == "available"
    assert metadata["status"] == "start"


def test_alan_agent_metadata_extra_fields():
    """AlanAgentMetaData에 추가 필드를 수동으로 추가할 수 있는지 테스트"""
    # 기본 필드와 함께 추가 필드 생성
    metadata = AlanAgentMetaData(
        chat_status="available",
        status="start"
    )
    # TypedDict는 딕셔너리이므로 추가 필드 가능
    metadata["user_id"] = 123
    metadata["timestamp"] = "2024-01-01"
    metadata["custom_field"] = "custom_value"
    
    # 기본 필드 확인
    assert metadata["chat_status"] == "available"
    assert metadata["status"] == "start"
    
    # 추가 필드 확인
    assert metadata["custom_field"] == "custom_value"
    assert metadata["user_id"] == 123
    assert metadata["timestamp"] == "2024-01-01"
    
    # 나중에 필드 추가
    metadata["new_field"] = "new_value"
    assert metadata["new_field"] == "new_value"


def test_base_alan_agent_state_metadata_type():
    """BaseAlanAgentState의 metadata가 AlanAgentMetaData 타입인지 테스트"""
    # 필수 필드들을 포함하여 상태 생성
    state = BaseAlanAgentState(
        messages=[],
        structured_response={},
        metadata=AlanAgentMetaData(
            chat_status="available",
            status="start"
        )
    )
    # TypedDict이므로 딕셔너리로 접근
    assert isinstance(state, dict)
    assert "metadata" in state
    assert isinstance(state["metadata"], dict)
    assert state["metadata"]["chat_status"] == "available"
    assert state["metadata"]["status"] == "start"


def test_canvas_required_fields():
    """Canvas 클래스의 필수 필드들이 올바르게 설정되는지 테스트"""
    canvas = Canvas(
        type="markdown",
        metadata={"key": "value"}
    )
    assert canvas["type"] == "markdown"
    assert canvas["metadata"] == {"key": "value"}


def test_alan_agent_state_with_canvas_inheritance():
    """AlanAgentStateWithCanvas가 BaseAlanAgentState를 올바르게 상속하는지 테스트"""
    # 필수 필드들을 포함하여 상태 생성
    state = AlanAgentStateWithCanvas(
        messages=[HumanMessage(content="test")],
        structured_response={},
        metadata=AlanAgentMetaData(
            chat_status="available",
            status="start"
        ),
        canvases=[]
    )
    # TypedDict이므로 딕셔너리로 접근
    assert isinstance(state, dict)
    assert "messages" in state
    assert "structured_response" in state
    assert "metadata" in state
    assert "canvases" in state
    assert state["canvases"] == []


def test_create_default_state_basic():
    """create_default_state 함수의 기본 동작을 테스트합니다"""
    # TypedDict 클래스에 대해 기본 상태 생성
    default_state = create_default_state(BaseAlanAgentState)
    
    # 기본값이 올바르게 설정되었는지 확인
    assert isinstance(default_state, dict)
    
    # BaseAlanAgentState의 필수 필드들 확인
    assert "messages" in default_state
    assert "metadata" in default_state
    
    # messages 필드는 빈 리스트여야 함
    assert default_state["messages"] == []
    
    # metadata 필드는 AlanAgentMetaData의 기본값을 가져야 함
    assert isinstance(default_state["metadata"], dict)
    assert "chat_status" in default_state["metadata"]
    assert "status" in default_state["metadata"]
    assert "initialization" in default_state["metadata"]
    
    # Literal 타입의 실제 구조 확인
    from typing import get_type_hints, get_origin, get_args
    metadata_hints = get_type_hints(AlanAgentMetaData)
    
    print(f"생성된 기본 상태: {default_state}")
    print(f"metadata 내용: {default_state['metadata']}")
    print(f"chat_status 타입: {metadata_hints['chat_status']}")
    print(f"chat_status origin: {get_origin(metadata_hints['chat_status'])}")
    print(f"chat_status args: {get_args(metadata_hints['chat_status'])}")
    print(f"status 타입: {metadata_hints['status']}")
    print(f"status origin: {get_origin(metadata_hints['status'])}")
    print(f"status args: {get_args(metadata_hints['status'])}")
    print(f"initialization 타입: {metadata_hints['initialization']}")
    print(f"initialization origin: {get_origin(metadata_hints['initialization'])}")
    print(f"initialization args: {get_args(metadata_hints['initialization'])}")


# JSON 변환 함수 테스트
def test_state_to_json_basic():
    """기본적인 상태 객체를 JSON으로 변환하는 테스트"""
    metadata = AlanAgentMetaData(
        chat_status="available",
        status="start",
        initialization=True
    )
    
    json_str = state_to_json(metadata)
    assert isinstance(json_str, str)
    assert "available" in json_str
    assert "start" in json_str
    assert "true" in json_str  # JSON에서 boolean은 소문자


def test_state_to_json_pretty():
    """보기 좋게 포맷된 JSON 변환 테스트"""
    metadata = AlanAgentMetaData(
        chat_status="available",
        status="start"
    )
    
    json_str = state_to_json_pretty(metadata)
    assert isinstance(json_str, str)
    assert "\n" in json_str  # 들여쓰기가 있으면 줄바꿈이 포함됨
    assert "  " in json_str  # 들여쓰기 공백이 포함됨
    assert "available" in json_str


def test_state_to_json_compact():
    """압축된 JSON 변환 테스트"""
    metadata = AlanAgentMetaData(
        chat_status="available",
        status="start"
    )
    
    json_str = state_to_json_compact(metadata)
    assert isinstance(json_str, str)
    assert "\n" not in json_str  # 줄바꿈이 없어야 함
    assert "  " not in json_str  # 들여쓰기 공백이 없어야 함
    assert "available" in json_str


def test_state_to_json_with_nested_typeddict():
    """중첩된 TypedDict를 JSON으로 변환하는 테스트"""
    
    class UserInfo(TypedDict):
        name: str
        age: int
    
    class ProjectInfo(TypedDict):
        id: str
        title: str
        tags: List[str]
    
    class ComplexState(TypedDict):
        user: UserInfo
        projects: List[ProjectInfo]
        metadata: AlanAgentMetaData
    
    complex_state = ComplexState(
        user=UserInfo(name="홍길동", age=30),
        projects=[
            ProjectInfo(id="proj-001", title="AI 프로젝트", tags=["AI", "ML"]),
            ProjectInfo(id="proj-002", title="웹 프로젝트", tags=["React", "TypeScript"])
        ],
        metadata=AlanAgentMetaData(
            chat_status="available",
            status="start"
        )
    )
    
    json_str = state_to_json_pretty(complex_state)
    assert "홍길동" in json_str
    assert "AI 프로젝트" in json_str
    assert "React" in json_str
    assert "available" in json_str


def test_state_to_json_with_base_alan_agent_state():
    """BaseAlanAgentState를 JSON으로 변환하는 테스트"""
    state = BaseAlanAgentState(
        messages=[HumanMessage(content="안녕하세요!")],
        structured_response={"response": "테스트 응답"},
        metadata=AlanAgentMetaData(
            chat_status="available",
            status="start",
            initialization=True
        )
    )
    
    json_str = state_to_json_pretty(state)
    assert "안녕하세요!" in json_str
    assert "테스트 응답" in json_str
    assert "available" in json_str
    assert "human" in json_str  # HumanMessage 타입


def test_state_to_json_with_canvas_state():
    """AlanAgentStateWithCanvas를 JSON으로 변환하는 테스트"""
    state = AlanAgentStateWithCanvas(
        messages=[HumanMessage(content="테스트 메시지")],
        structured_response={},
        metadata=AlanAgentMetaData(
            chat_status="available",
            status="start"
        ),
        canvases=[
            Canvas(type="markdown", metadata={"title": "마크다운 캔버스"}),
            Canvas(type="html", metadata={"title": "HTML 캔버스"})
        ]
    )
    
    json_str = state_to_json_pretty(state)
    assert "테스트 메시지" in json_str
    assert "markdown" in json_str
    assert "HTML 캔버스" in json_str
    assert "available" in json_str


def test_state_to_json_with_special_types():
    """특수 타입들을 포함한 객체를 JSON으로 변환하는 테스트"""
    import json
    from datetime import datetime, date
    from decimal import Decimal
    from uuid import uuid4
    
    # 일반 딕셔너리로 테스트 (TypedDict는 런타임에 특수 타입을 직접 포함할 수 없음)
    special_state = {
        "string_val": "테스트 문자열",
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "list_val": [1, 2, 3],
        "dict_val": {"key": "value"},
        "none_val": None,
        "datetime_val": datetime(2024, 1, 1, 12, 0, 0),
        "date_val": date(2024, 1, 1),
        "decimal_val": Decimal("123.45"),
        "uuid_val": uuid4(),
        "bytes_val": b"test bytes"
    }
    
    json_str = state_to_json_pretty(special_state)
    assert "테스트 문자열" in json_str
    assert "42" in json_str
    assert "3.14" in json_str
    assert "true" in json_str
    assert "1" in json_str and "2" in json_str and "3" in json_str  # 리스트 내용 확인
    assert "null" in json_str  # None은 JSON에서 null로 변환


def test_state_to_json_with_unicode():
    """유니코드 문자(한글 등)를 포함한 JSON 변환 테스트"""
    metadata = AlanAgentMetaData(
        chat_status="available",
        status="start"
    )
    
    # 한글과 특수 문자가 포함된 추가 필드
    metadata["korean_text"] = "안녕하세요! 반갑습니다."
    metadata["special_chars"] = "🚀✨🎉"
    metadata["emoji_text"] = "이모지 테스트 🎯🎲🎮"
    
    json_str = state_to_json_pretty(metadata)
    assert "안녕하세요" in json_str
    assert "반갑습니다" in json_str
    assert "🚀" in json_str
    assert "🎯" in json_str


def test_state_to_json_error_handling():
    """JSON 변환 중 오류 처리 테스트"""
    # 직렬화할 수 없는 객체를 포함한 딕셔너리
    class NonSerializableObject:
        def __init__(self):
            self.data = "test"
    
    problematic_state = {
        "normal_field": "정상 필드",
        "problematic_field": NonSerializableObject()
    }
    
    # 오류가 발생하지 않고 __dict__를 통해 변환되어야 함
    json_str = state_to_json_pretty(problematic_state)
    assert "정상 필드" in json_str
    assert "data" in json_str  # __dict__의 내용이 포함됨
    assert "test" in json_str  # __dict__의 값이 포함됨


def test_state_to_json_with_empty_objects():
    """빈 객체들을 JSON으로 변환하는 테스트"""
    empty_state = {
        "empty_list": [],
        "empty_dict": {},
        "empty_string": "",
        "null_value": None
    }
    
    json_str = state_to_json_pretty(empty_state)
    assert "[]" in json_str
    assert "{}" in json_str
    assert '""' in json_str
    assert "null" in json_str


def test_state_to_json_indent_options():
    """다양한 들여쓰기 옵션 테스트"""
    metadata = AlanAgentMetaData(
        chat_status="available",
        status="start"
    )
    
    # 기본 들여쓰기 (2)
    json_default = state_to_json(metadata, indent=2)
    assert "  " in json_default
    
    # 4칸 들여쓰기
    json_indent_4 = state_to_json(metadata, indent=4)
    assert "    " in json_indent_4
    
    # 압축 (들여쓰기 없음)
    json_compact = state_to_json(metadata, indent=None)
    assert "  " not in json_compact
    assert "\n" not in json_compact


def test_state_to_json_ensure_ascii():
    """ensure_ascii 옵션 테스트"""
    metadata = AlanAgentMetaData(
        chat_status="available",
        status="start"
    )
    metadata["korean"] = "한글"
    
    # ensure_ascii=False (기본값, 한글 지원)
    json_unicode = state_to_json(metadata, ensure_ascii=False)
    assert "한글" in json_unicode
    
    # ensure_ascii=True (ASCII만 사용)
    json_ascii = state_to_json(metadata, ensure_ascii=True)
    assert "한글" not in json_ascii
    assert "\\u" in json_ascii  # 유니코드 이스케이프 시퀀스
