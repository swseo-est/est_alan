"""
아키텍처 개요
--------------
- **BaseChatModelWrapper**: LangChain ChatModel 인스턴스를 합성(composition) 방식으로 감싸는 범용 래퍼.
- **주요 기능**
    * pre/post 훅을 통한 동기·비동기 호출 인터셉터(예: 로깅, 메트릭 수집).
    * 예시용 공통 메서드 `stream_json` 제공.
- 원본 ChatModel의 메서드·프로퍼티는 `__getattr__`로 투명하게 위임.
- 구체 래퍼(`ChatOpenAIWrapper`, `ChatAnthropicWrapper` 등)는 상속하여 훅을 재정의해 모델별 맞춤 동작을 구현.
- 팩토리 함수 `wrap_chat_model`로 일관된 방식으로 래핑 인스턴스를 생성할 수 있음.

사용 예시
---------
```python
from chatmodel_wrapper import ChatOpenAIWrapper

chat = ChatOpenAIWrapper(api_key="...", temperature=0.7)
response = chat.invoke("Hello!")

for chunk in chat.stream_json("Hello, stream!"):
    print(chunk)
```
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Iterable

from langchain.chat_models.base import BaseChatModel


class AlanBaseChatModelWrapper(ABC):
    """LangChain ChatModel을 감싸 동작을 확장·오버라이드하기 위한 기본 래퍼."""

    def __init__(self, model: BaseChatModel, *, name: str | None = None):
        self._model = model
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
        result = await self._model.ainvoke(*args, **kwargs)
        return self._post_hook(result)

    def invoke(self, *args: Any, **kwargs: Any):
        """동기 호출 전/후에 훅을 실행."""
        self._pre_hook(*args, **kwargs)
        result = self._model.invoke(*args, **kwargs)
        return self._post_hook(result)

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


