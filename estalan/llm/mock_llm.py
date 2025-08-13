from typing import Any, Dict, List, Optional, Type, get_type_hints, Union
from langchain.schema import BaseMessage, AIMessage, HumanMessage
from langchain.chat_models.base import BaseChatModel
from estalan.llm.base import AlanBaseChatModelWrapper
import json
import random


class MockChatModel(BaseChatModel):
    """테스트용 Mock ChatModel 클래스"""
    
    def __init__(self, responses: Optional[List[str]] = None, default_response: str = "Mock response", **kwargs):
        # Pydantic 필드 설정 전에 일반 속성으로 초기화
        self._responses = responses or [default_response]
        self._default_response = default_response
        self._response_index = 0
        self._call_count = 0
        self._structured_schema = None
        super().__init__(**kwargs)
    
    @property
    def responses(self) -> List[str]:
        return self._responses
    
    @property
    def default_response(self) -> str:
        return self._default_response
    
    @property
    def response_index(self) -> int:
        return self._response_index
    
    @property
    def call_count(self) -> int:
        return self._call_count
    
    @property
    def structured_schema(self) -> Optional[Type]:
        return self._structured_schema
    
    @structured_schema.setter
    def structured_schema(self, value: Optional[Type]):
        self._structured_schema = value
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> AIMessage:
        """동기 생성 메서드"""
        self._call_count += 1
        response = self._get_structured_response()
        return AIMessage(content=response)
    
    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> AIMessage:
        """비동기 생성 메서드"""
        self._call_count += 1
        response = self._get_structured_response()
        return AIMessage(content=response)
    
    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any):
        """스트리밍 생성 메서드"""
        self._call_count += 1
        response = self._get_structured_response()
        # 응답을 단어 단위로 나누어 청크로 반환
        words = response.split()
        for word in words:
            yield AIMessage(content=word + " ")
    
    def with_structured_output(self, schema, **kwargs):
        """구조화된 출력을 지원하도록 래핑"""
        from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser
        from langchain.schema.output_parser import BaseOutputParser
        
        # Mock 모델에 structured output 지원 추가
        if hasattr(self, '_structured_output_parser'):
            # 이미 structured output이 설정된 경우
            return self
        
        # 스키마 저장
        self.structured_schema = schema
        
        # Pydantic 스키마를 파서로 변환
        if hasattr(schema, '__fields__'):  # Pydantic 모델인 경우
            parser = PydanticOutputFunctionsParser(pydantic_schema=schema)
        elif isinstance(schema, BaseOutputParser):
            parser = schema
        else:
            # 기본 파서 사용
            parser = BaseOutputParser()
        
        # Mock 모델에 파서 저장
        self._structured_output_parser = parser
        return self
    
    def _llm_type(self) -> str:
        """LLM 타입 반환"""
        return "mock"
    
    def _get_structured_response(self) -> str:
        """구조화된 응답 또는 일반 응답을 반환"""
        if self.structured_schema is not None:
            return self._generate_structured_response()
        else:
            return self._get_next_response()
    
    def _generate_structured_response(self) -> str:
        """스키마에 맞는 구조화된 응답 생성"""
        if hasattr(self.structured_schema, '__fields__'):  # Pydantic 모델
            return self._generate_pydantic_response()
        elif hasattr(self.structured_schema, '__annotations__'):  # TypedDict
            return self._generate_typeddict_response()
        else:
            return self._get_next_response()
    
    def _generate_pydantic_response(self) -> str:
        """Pydantic 모델에 맞는 응답 생성"""
        mock_data = {}
        fields = self.structured_schema.__fields__
        
        for field_name, field_info in fields.items():
            field_type = field_info.type_
            mock_data[field_name] = self._generate_mock_value(field_type)
        
        return json.dumps(mock_data, ensure_ascii=False, indent=2)
    
    def _generate_typeddict_response(self) -> str:
        """TypedDict에 맞는 응답 생성"""
        mock_data = {}
        annotations = self.structured_schema.__annotations__
        
        for field_name, field_type in annotations.items():
            mock_data[field_name] = self._generate_mock_value(field_type)
        
        return json.dumps(mock_data, ensure_ascii=False, indent=2)
    
    def _generate_mock_value(self, field_type) -> Any:
        """필드 타입에 맞는 Mock 값 생성"""
        # Union 타입 처리 (Optional 포함)
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            # Optional인 경우 None이 아닌 타입 선택
            args = [arg for arg in field_type.__args__ if arg is not type(None)]
            if args:
                field_type = args[0]
            else:
                field_type = str
        
        # List 타입 처리
        if hasattr(field_type, '__origin__') and field_type.__origin__ is list:
            element_type = field_type.__args__[0] if field_type.__args__ else str
            return [self._generate_mock_value(element_type) for _ in range(random.randint(1, 3))]
        
        # Dict 타입 처리
        if hasattr(field_type, '__origin__') and field_type.__origin__ is dict:
            return {"key": "value"}
        
        # 기본 타입들
        if field_type == str:
            return f"Mock String {self._call_count}"
        elif field_type == int:
            return random.randint(1, 100)
        elif field_type == float:
            return round(random.uniform(0.0, 100.0), 2)
        elif field_type == bool:
            return random.choice([True, False])
        elif field_type == list:
            return ["item1", "item2", "item3"]
        elif field_type == dict:
            return {"key": "value"}
        else:
            return f"Mock {field_type.__name__} {self._call_count}"
    
    def _get_next_response(self) -> str:
        """다음 응답을 가져오고 인덱스를 순환"""
        response = self._responses[self._response_index % len(self._responses)]
        self._response_index += 1
        return response
    



class AlanMockLLM(AlanBaseChatModelWrapper):
    """테스트 및 개발 환경용 Mock LLM 래퍼"""
    
    def __init__(self, responses: Optional[List[str]] = None, default_response: str = "Mock response", **kwargs):
        mock_model = MockChatModel(responses=responses, default_response=default_response)
        super().__init__(mock_model, **kwargs)
    
    def with_structured_output(self, schema, **kwargs):
        """구조화된 출력을 지원하도록 래핑"""
        # 내부 모델에 structured output 설정
        self._model = self._model.with_structured_output(schema, **kwargs)
        return self
    
    def _pre_hook(self, *args: Any, **kwargs: Any) -> None:
        """Mock 호출 전 로깅"""
        print(f"[Mock LLM] 호출됨 - 총 호출 횟수: {self._model._call_count + 1}")
    
    def _post_hook(self, result: Any):
        """Mock 호출 후 처리"""
        print(f"[Mock LLM] 응답 생성 완료")
        return result
