from typing import Dict, Sequence, Literal, Type, TypeVar, Union, get_type_hints, get_origin, get_args, Any
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.prebuilt.chat_agent_executor import AgentStateWithStructuredResponse

from estalan.agent.base.reducer_function import add_messages_for_alan, update_metadata

# 상태 클래스용 타입 변수
T = TypeVar('T')

class AlanAgentMetaData(TypedDict):
    chat_status: Literal["available", "unavailable"]
    status: Literal["start", "finish"]
    initialization: Literal[False, True]


def update_remaining_step(old, new):
    return 25


class BaseAlanAgentState(AgentStateWithStructuredResponse):
    messages: Annotated[Sequence[BaseMessage], add_messages_for_alan]
    metadata: Annotated[AlanAgentMetaData, update_metadata]
    private_state: Annotated[Dict, update_metadata]
    shared_state: Annotated[Dict, update_metadata]
    # remaining_steps: Annotated[int, update_remaining_step]


class Canvas(TypedDict):
    type: Literal["markdown", "slide", "html", "txt", "image"]
    metadata: dict


class AlanAgentStateWithCanvas(BaseAlanAgentState):
    canvases: list[Canvas]


def create_default_state(state_class: Type[T]) -> T:
    """
    모든 필드에 기본값이 채워진 기본 상태 인스턴스를 생성합니다.
    
    Args:
        state_class: 상태 클래스 (TypedDict 또는 Pydantic BaseModel)
        
    Returns:
        기본값이 채워진 상태 클래스의 인스턴스
        
    Raises:
        ValueError: 지원되지 않는 상태 클래스 타입인 경우
    """
    # Pydantic BaseModel인지 확인
    if hasattr(state_class, 'model_construct'):
        return _create_pydantic_default(state_class)
    
    # TypedDict인지 확인
    elif hasattr(state_class, '__annotations__'):
        return _create_typeddict_default(state_class)
    
    # 일반 클래스인 경우 직접 인스턴스 생성 시도
    else:
        try:
            return state_class()
        except Exception as e:
            raise ValueError(f"지원되지 않는 상태 클래스 타입: {state_class}. 오류: {e}")


def _create_pydantic_default(state_class: Type[T]) -> T:
    """
    Pydantic BaseModel 필드에 대한 기본값을 생성합니다.
    
    Args:
        state_class: Pydantic BaseModel 클래스
        
    Returns:
        기본값이 채워진 인스턴스
    """
    # Pydantic 모델에서 필드 정보 가져오기
    model_fields = state_class.model_fields
    default_values = {}
    
    for field_name, field_info in model_fields.items():
        if field_info.default is not None:
            default_values[field_name] = field_info.default
        elif field_info.default_factory is not None:
            default_values[field_name] = field_info.default_factory()
        else:
            # 필드 타입에 기반하여 기본값 생성
            default_values[field_name] = _get_default_value_for_type(field_info.annotation)
    
    # 더 나은 성능을 위해 model_construct 사용
    return state_class.model_construct(**default_values)


def _create_typeddict_default(state_class: Type[T]) -> T:
    """
    TypedDict 필드에 대한 기본값을 생성합니다.
    재귀적으로 중첩된 TypedDict와 복잡한 타입들을 처리합니다.
    
    Args:
        state_class: TypedDict 클래스
        
    Returns:
        기본값이 채워진 인스턴스
    """
    try:
        annotations = get_type_hints(state_class)
        default_values = {}
        
        for field_name, field_type in annotations.items():
            try:
                # 재귀적으로 각 필드의 기본값 생성
                default_values[field_name] = _get_default_value_for_type(field_type)
            except Exception as e:
                # 특정 필드에서 오류가 발생하면 기본값으로 None 사용
                print(f"경고: {field_name} 필드의 기본값 생성 실패: {e}")
                default_values[field_name] = None
        
        return state_class(**default_values)
    except Exception as e:
        # 전체 생성 과정에서 오류가 발생하면 빈 인스턴스 반환 시도
        print(f"경고: {state_class.__name__}의 기본값 생성 실패: {e}")
        try:
            return state_class()
        except:
            # 마지막 수단으로 빈 딕셔너리 반환
            return {}


def _get_default_value_for_type(field_type: Any) -> Any:
    """
    주어진 타입에 대한 적절한 기본값을 반환합니다.
    재귀적으로 하위 타입들을 처리하여 계층적 구조의 기본값을 생성합니다.
    
    Args:
        field_type: 타입 어노테이션
        
    Returns:
        타입에 대한 기본값
    """
    origin = get_origin(field_type)
    
    # Literal 타입 처리 (예: Literal["available", "unavailable"])
    if origin is not None and hasattr(origin, '__name__') and origin.__name__ == 'Literal':
        args = get_args(field_type)
        if args:
            # Literal의 첫 번째 값을 기본값으로 사용
            return args[0]
        else:
            return None
    
    # Check if it's Optional[T] or Union[T, None]
    if origin is Union or origin == Union.__class__:
        args = get_args(field_type)
        if type(None) in args:
            # 이는 Optional[T] 또는 Union[T, None]입니다
            non_none_types = [arg for arg in args if arg != type(None)]
            if non_none_types:
                # Optional 타입에 대해 None 반환
                return None
            else:
                return None
        else:
            # Union[T1, T2, ...]에서 None이 없는 경우, 첫 번째 타입의 기본값 사용
            non_none_types = [arg for arg in args if arg != type(None)]
            if non_none_types:
                # 첫 번째 non-None 타입의 기본값을 시도
                first_type = non_none_types[0]
                try:
                    return _get_default_value_for_type(first_type)
                except Exception:
                    # 첫 번째 타입이 실패하면 다음 타입 시도
                    for other_type in non_none_types[1:]:
                        try:
                            return _get_default_value_for_type(other_type)
                        except Exception:
                            continue
                    # 모든 타입이 실패하면 빈 딕셔너리 반환 (dict가 포함된 경우)
                    if dict in non_none_types:
                        return {}
                    # 마지막 수단으로 None 반환
                    return None
            else:
                return None
    
    # Annotated 타입 처리 (예: Annotated[Sequence[BaseMessage], add_messages_for_alan])
    if origin is Annotated:
        args = get_args(field_type)
        if args:
            # Annotated의 첫 번째 인자는 실제 타입
            actual_type = args[0]
            return _get_default_value_for_type(actual_type)
    
    if origin is None:
        # Handle primitive types
        if field_type == str:
            return ""
        elif field_type == int:
            return 0
        elif field_type == float:
            return 0.0
        elif field_type == bool:
            return False
        elif field_type == dict:
            return {}
        elif field_type == list:
            return []
        elif field_type == tuple:
            return ()
        elif field_type == set:
            return set()
        elif field_type == bytes:
            return b""
        elif field_type == type(None):
            return None
        else:
            # 사용자 정의 타입의 경우 재귀적으로 기본값 생성 시도
            if hasattr(field_type, '__annotations__'):
                # TypedDict나 dataclass인 경우
                return _create_typeddict_default(field_type)
            else:
                # 일반 클래스의 경우 인스턴스 생성 시도
                try:
                    return field_type()
                except:
                    return None
    
    elif origin is list:
        args = get_args(field_type)
        if args:
            # List[T]의 경우 T에 대한 기본값을 생성하여 리스트에 포함
            # 하지만 빈 리스트가 더 안전함
            return []
        return []
    elif origin is dict:
        args = get_args(field_type)
        if args:
            # Dict[K, V]의 경우 빈 딕셔너리 반환
            return {}
        return {}
    elif origin is tuple:
        args = get_args(field_type)
        if args:
            # Tuple[T1, T2, ...]의 경우 각 타입에 대한 기본값을 생성
            return tuple(_get_default_value_for_type(arg) for arg in args)
        return ()
    elif origin is set:
        args = get_args(field_type)
        if args:
            # Set[T]의 경우 빈 셋 반환
            return set()
        return set()
    elif hasattr(origin, '__name__') and origin.__name__ == 'Sequence':
        # typing.Sequence[T] 처리
        args = get_args(field_type)
        if args:
            # Sequence[T]의 경우 빈 리스트 반환
            return []
        return []
    else:
        # For other generic types, try to create instance
        try:
            return field_type()
        except:
            return None


def state_to_json(state_obj: Any, indent: int = 2, ensure_ascii: bool = False) -> str:
    """
    계층적으로 구성된 TypedDict 객체를 JSON 문자열로 변환합니다.
    
    Args:
        state_obj: 변환할 상태 객체 (TypedDict, Pydantic BaseModel, 일반 객체 등)
        indent: JSON 들여쓰기 크기 (기본값: 2)
        ensure_ascii: ASCII 문자만 사용할지 여부 (기본값: False, 한글 지원)
        
    Returns:
        JSON 문자열
        
    Raises:
        TypeError: JSON으로 직렬화할 수 없는 객체가 포함된 경우
        ValueError: 객체 변환 중 오류가 발생한 경우
    """
    import json
    from datetime import datetime, date
    from decimal import Decimal
    from uuid import UUID
    
    def _convert_to_serializable(obj: Any) -> Any:
        """
        객체를 JSON 직렬화 가능한 형태로 변환합니다.
        
        Args:
            obj: 변환할 객체
            
        Returns:
            JSON 직렬화 가능한 객체
        """
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        elif isinstance(obj, (list, tuple)):
            return [_convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): _convert_to_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            # 일반 객체의 경우 __dict__ 사용
            return _convert_to_serializable(obj.__dict__)
        elif hasattr(obj, 'model_dump'):
            # Pydantic BaseModel의 경우
            return _convert_to_serializable(obj.model_dump())
        elif hasattr(obj, 'dict'):
            # Pydantic v1 스타일
            return _convert_to_serializable(obj.dict())
        elif hasattr(obj, '__annotations__'):
            # TypedDict나 dataclass의 경우
            result = {}
            for attr_name in dir(obj):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(obj, attr_name)
                        if not callable(attr_value):
                            result[attr_name] = _convert_to_serializable(attr_value)
                    except Exception:
                        continue
            return result
        else:
            # 기타 객체는 문자열로 변환 시도
            try:
                return str(obj)
            except Exception:
                return f"<non-serializable: {type(obj).__name__}>"
    
    try:
        # 객체를 JSON 직렬화 가능한 형태로 변환
        serializable_obj = _convert_to_serializable(state_obj)
        
        # JSON 문자열로 변환
        return json.dumps(
            serializable_obj, 
            indent=indent, 
            ensure_ascii=ensure_ascii,
            default=str  # 마지막 수단으로 str() 사용
        )
    except Exception as e:
        raise ValueError(f"상태 객체를 JSON으로 변환하는 중 오류 발생: {e}")


def state_to_json_pretty(state_obj: Any) -> str:
    """
    계층적으로 구성된 TypedDict 객체를 보기 좋게 포맷된 JSON 문자열로 변환합니다.
    
    Args:
        state_obj: 변환할 상태 객체
        
    Returns:
        보기 좋게 포맷된 JSON 문자열
    """
    return state_to_json(state_obj, indent=2, ensure_ascii=False)


def state_to_json_compact(state_obj: Any) -> str:
    """
    계층적으로 구성된 TypedDict 객체를 압축된 JSON 문자열로 변환합니다.
    
    Args:
        state_obj: 변환할 상태 객체
        
    Returns:
        압축된 JSON 문자열 (공백 없음)
    """
    return state_to_json(state_obj, indent=None, ensure_ascii=False)


