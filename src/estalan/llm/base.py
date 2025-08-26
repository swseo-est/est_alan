# LLM 기본 래퍼 모듈
# LangChain ChatModel을 감싸서 재시도 로직, 훅, 확장 기능을 제공하는 기본 클래스를 정의합니다.

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Iterable

from langchain.chat_models.base import BaseChatModel
import time


class AlanBaseChatModelWrapper(ABC):
    """
    LangChain ChatModel을 감싸 동작을 확장·오버라이드하기 위한 기본 래퍼.
    
    이 클래스는 다양한 언어 모델에 공통적으로 적용되는 기능들을 제공합니다:
    - 자동 재시도 로직
    - 호출 전/후 훅 실행
    - 스트리밍 JSON 지원
    - 에러 처리 및 로깅
    """

    def __init__(self, model: BaseChatModel, *, name: str | None = None):
        """
        기본 래퍼를 초기화합니다.
        
        Args:
            model: 감쌀 LangChain ChatModel 인스턴스
            name: 래퍼의 이름 (기본값: 원본 모델의 클래스명)
        """
        self._model = model
        self._max_retry = 10  # 최대 재시도 횟수
        # 래퍼 이름: 전달받지 않으면 원본 클래스명을 사용
        self.name = name or model.__class__.__name__

    # ------------------------------------------------------------------
    # 위임: 존재하지 않는 속성은 내부 모델로 전달
    # ------------------------------------------------------------------
    def __getattr__(self, item):
        """
        존재하지 않는 속성이나 메서드는 내부 모델로 위임합니다.
        이를 통해 원본 모델의 모든 기능을 그대로 사용할 수 있습니다.
        """
        return getattr(self._model, item)

    # ------------------------------------------------------------------
    # 공개 API: 훅이 포함된 비동기 / 동기 호출 래핑
    # ------------------------------------------------------------------
    async def ainvoke(self, *args: Any, **kwargs: Any):
        """
        비동기 호출 전/후에 훅을 실행합니다.
        
        자동 재시도 로직이 포함되어 있어 일시적인 네트워크 오류나 API 제한에 대응합니다.
        
        Args:
            *args: 모델 호출에 전달할 위치 인수
            **kwargs: 모델 호출에 전달할 키워드 인수
            
        Returns:
            모델의 응답 결과 (훅 처리 후)
            
        Raises:
            Exception: 모든 재시도 실패 시 원본 예외를 발생시킵니다
        """
        self._pre_hook(*args, **kwargs)
        last_exception = None

        for i in range(self._max_retry):
            try:
                result = await self._model.ainvoke(*args, **kwargs)
                return self._post_hook(result)
            except Exception as e:
                print(f"retry {i + 1}/{self._max_retry}")
                print(f"Error: {e}")
                last_exception = e
                if i < self._max_retry - 1:  # 마지막 시도가 아니면 대기
                    time.sleep(1)
        
        # 모든 retry 시도 실패 시 원본 에러를 그대로 발생
        raise last_exception from last_exception

    def invoke(self, *args: Any, **kwargs: Any):
        """
        동기 호출 전/후에 훅을 실행합니다.
        
        자동 재시도 로직이 포함되어 있어 일시적인 네트워크 오류나 API 제한에 대응합니다.
        
        Args:
            *args: 모델 호출에 전달할 위치 인수
            **kwargs: 모델 호출에 전달할 키워드 인수
            
        Returns:
            모델의 응답 결과 (훅 처리 후)
            
        Raises:
            Exception: 모든 재시도 실패 시 원본 예외를 발생시킵니다
        """
        self._pre_hook(*args, **kwargs)
        last_exception = None

        for i in range(self._max_retry):
            try:
                result = self._model.invoke(*args, **kwargs)
                return self._post_hook(result)
            except Exception as e:
                print(f"retry {i + 1}/{self._max_retry}")
                print(f"Error: {e}")
                last_exception = e
                if i < self._max_retry - 1:  # 마지막 시도가 아니면 대기
                    time.sleep(1)

        # 모든 retry 시도 실패 시 원본 에러를 그대로 발생
        raise last_exception from last_exception

    # ------------------------------------------------------------------
    # 확장 포인트: 공통 유틸리티 메서드 예시
    # ------------------------------------------------------------------
    def stream_json(self, *args: Any, **kwargs: Any) -> Iterable[Dict[str, Any]]:
        """
        텍스트 대신 JSON 청크를 스트리밍하여 후속 파이프라인을 일관되게 구성합니다.
        
        각 청크는 모델 이름과 내용을 포함하는 딕셔너리 형태로 반환됩니다.
        
        Args:
            *args: 스트리밍 호출에 전달할 위치 인수
            **kwargs: 스트리밍 호출에 전달할 키워드 인수
            
        Returns:
            JSON 형태의 스트리밍 데이터 이터레이터
        """
        for chunk in self._model.stream(*args, **kwargs):
            yield {"model": self.name, "content": chunk}

    # ------------------------------------------------------------------
    # 자식 클래스가 필요에 따라 오버라이드할 훅
    # ------------------------------------------------------------------
    def _pre_hook(self, *args: Any, **kwargs: Any) -> None:
        """
        모델 호출 직전에 실행되는 훅입니다.
        
        자식 클래스에서 오버라이드하여 프롬프트 전처리, 로깅, 검증 등을 수행할 수 있습니다.
        
        Args:
            *args: 호출 인수
            **kwargs: 호출 키워드 인수
        """
        pass

    def _post_hook(self, result: Any):
        """
        모델 호출 후 실행되는 훅입니다.
        
        자식 클래스에서 오버라이드하여 결과 후처리, 트리밍, 변환 등을 수행할 수 있습니다.
        
        Args:
            result: 모델의 원본 응답 결과
            
        Returns:
            처리된 결과 (기본적으로는 원본 결과를 그대로 반환)
        """
        return result




