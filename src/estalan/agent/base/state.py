from typing import Annotated, Sequence, Literal, TypedDict, Type, TypeVar, Union, get_type_hints, get_origin, get_args, Any
from typing_extensions import get_type_hints as get_type_hints_ext

from langchain_core.messages import BaseMessage
from langgraph.prebuilt.chat_agent_executor import AgentStateWithStructuredResponse

from estalan.agent.base.reducer_function import add_messages_for_alan, update_metadata

# 상태 클래스용 타입 변수
T = TypeVar('T')

class AlanAgentMetaData(TypedDict, total=False):
    chat_status: Literal["available", "unavailable"]
    status: Literal["start", "finish"]
    initialization: Literal[False, True]


class BaseAlanAgentState(AgentStateWithStructuredResponse):
    messages: Annotated[Sequence[BaseMessage], add_messages_for_alan]
    metadata: Annotated[AlanAgentMetaData, update_metadata]


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