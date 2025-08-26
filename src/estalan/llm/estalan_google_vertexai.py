# Google VertexAI LLM 래퍼 모듈
# Google Cloud VertexAI 서비스를 위한 AlanBaseChatModelWrapper 구현체를 제공합니다.

import os

from google.oauth2 import service_account
from langchain_google_vertexai import ChatVertexAI
from estalan.llm.base import AlanBaseChatModelWrapper


class AlanChatVertexAI(AlanBaseChatModelWrapper):
    """
    Google Cloud VertexAI 서비스를 위한 Alan 래퍼 클래스
    
    Google Cloud의 VertexAI 서비스를 통해 Gemini, PaLM 등의 모델을 사용할 수 있습니다.
    Google Cloud 서비스 계정 인증을 사용하여 VertexAI에 접근합니다.
    재시도 로직, 훅, 에러 처리 등 AlanBaseChatModelWrapper의 모든 기능을 포함합니다.
    """
    def __init__(self, **kwargs):
        """
        Google VertexAI 채팅 모델 래퍼를 초기화합니다.
        
        Google Cloud 서비스 계정 인증 정보를 사용하여
        VertexAI 채팅 모델 인스턴스를 생성합니다.
        
        Args:
            **kwargs: ChatVertexAI 생성자에 전달할 매개변수들
                     (model, temperature, max_tokens 등)
        """
        super().__init__(
            ChatVertexAI(
                **kwargs,
                credentials=service_account.Credentials.from_service_account_file(
                    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                ),
            )
        )