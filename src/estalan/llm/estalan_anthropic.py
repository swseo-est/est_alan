# Anthropic LLM 래퍼 모듈
# Anthropic Claude 모델과 Google VertexAI를 통한 Claude 모델을 위한 AlanBaseChatModelWrapper 구현체를 제공합니다.

import os
from google.oauth2 import service_account

from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from estalan.llm.base import AlanBaseChatModelWrapper


class AlanChatAnthropic(AlanBaseChatModelWrapper):
    """
    Anthropic Claude 모델을 위한 Alan 래퍼 클래스
    
    Anthropic의 Claude 모델을 직접 사용하여 대화형 AI와 상호작용할 수 있습니다.
    재시도 로직, 훅, 에러 처리 등 AlanBaseChatModelWrapper의 모든 기능을 포함합니다.
    """
    def __init__(self, **kwargs):
        """
        Anthropic Claude 채팅 모델 래퍼를 초기화합니다.
        
        Args:
            **kwargs: ChatAnthropic 생성자에 전달할 매개변수들
                     (model, temperature, max_tokens 등)
        """
        super().__init__(ChatAnthropic(**kwargs))


class AlanChatAnthropicVertex(AlanBaseChatModelWrapper):
    """
    Google VertexAI를 통한 Anthropic Claude 모델을 위한 Alan 래퍼 클래스
    
    Google Cloud의 VertexAI 서비스를 통해 Claude 모델을 사용할 수 있습니다.
    Google Cloud 서비스 계정 인증을 사용하여 Claude 모델에 접근합니다.
    """
    def __init__(self, **kwargs):
        """
        VertexAI Claude 채팅 모델 래퍼를 초기화합니다.
        
        Google Cloud 서비스 계정 인증 정보를 사용하여
        VertexAI를 통한 Claude 모델 인스턴스를 생성합니다.
        
        Args:
            **kwargs: ChatAnthropicVertex 생성자에 전달할 매개변수들
                     (model, temperature, max_tokens 등)
        """
        super().__init__(
            ChatAnthropicVertex(
                **kwargs,
                credentials=service_account.Credentials.from_service_account_file(
                    os.getenv("ANTHROPIC_VERTEXAI_CREDENTIALS")
                ).with_scopes(["https://www.googleapis.com/auth/cloud-platform"]),
                location=os.getenv("ANTHROPIC_VERTEXAI_LOCATION")
            )
        )


if __name__ == '__main__':
    # 테스트 실행 코드
    from dotenv import load_dotenv

    load_dotenv()

    # VertexAI를 통한 Claude 모델 테스트
    llm = AlanChatAnthropicVertex(
        model="claude-sonnet-4@20250514",
    )
    result = llm.invoke("hi")
    print(result)