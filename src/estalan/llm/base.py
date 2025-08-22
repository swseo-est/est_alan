from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Iterable

from langchain.chat_models.base import BaseChatModel
import time


class AlanBaseChatModelWrapper(ABC):
    """LangChain ChatModel을 감싸 동작을 확장·오버라이드하기 위한 기본 래퍼."""

    def __init__(self, model: BaseChatModel, *, name: str | None = None):
        self._model = model
        self._max_retry = 10
        # 래퍼 이름: 전달받지 않으면 원본 클래스명을 사용
        self.name = name or model.__class__.__name__

    # ------------------------------------------------------------------
    # 위임: 존재하지 않는 속성은 내부 모델로 전달
    # ------------------------------------------------------------------
    def __getattr__(self, item):
        return getattr(self._model, item)

    # ------------------------------------------------------------------
    # 공개 API: 훅이 포함된 비동기 / 동기 호출 래핑
    # ------------------------------------------------------------------
    async def ainvoke(self, *args: Any, **kwargs: Any):
        """비동기 호출 전/후에 훅을 실행."""
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
        """동기 호출 전/후에 훅을 실행."""
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
        """텍스트 대신 JSON 청크를 스트리밍하여 후속 파이프라인을 일관되게 구성."""
        for chunk in self._model.stream(*args, **kwargs):
            yield {"model": self.name, "content": chunk}

    # ------------------------------------------------------------------
    # 자식 클래스가 필요에 따라 오버라이드할 훅
    # ------------------------------------------------------------------
    def _pre_hook(self, *args: Any, **kwargs: Any) -> None:
        """모델 호출 직전에 실행(예: 프롬프트 전처리, 로깅)."""
        pass

    def _post_hook(self, result: Any):
        """모델 호출 후 실행(예: 결과 후처리, 트리밍)."""
        return result




