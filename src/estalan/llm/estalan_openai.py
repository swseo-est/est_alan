# OpenAI LLM 래퍼 모듈
# OpenAI와 Azure OpenAI 서비스를 위한 AlanBaseChatModelWrapper 구현체를 제공합니다.

import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from estalan.llm.base import AlanBaseChatModelWrapper
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI 기본 설정값
# 환경 변수에서 Azure OpenAI 서비스 설정을 가져옵니다.
DEFAULT_AZUREOPENAI_KWARGS = {
    "azure_endpoint" : os.getenv("AZURE_ENDPOINT"),
    "openai_api_version" : os.getenv("AZURE_OPENAI_API_VERSION"),
    "openai_api_key" : os.getenv("AZURE_OPENAI_API_KEY"),
    "deployment_name" : os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
}


class AlanChatOpenAI(AlanBaseChatModelWrapper):
    """
    OpenAI Chat API를 위한 Alan 래퍼 클래스
    
    OpenAI의 Chat API를 사용하여 대화형 AI 모델과 상호작용할 수 있습니다.
    재시도 로직, 훅, 에러 처리 등 AlanBaseChatModelWrapper의 모든 기능을 포함합니다.
    """
    def __init__(self, **kwargs):
        """
        OpenAI 채팅 모델 래퍼를 초기화합니다.
        
        Args:
            **kwargs: ChatOpenAI 생성자에 전달할 매개변수들
                     (model, temperature, max_tokens 등)
        """
        super().__init__(ChatOpenAI(**kwargs))


class AlanAzureChatOpenAI(AlanBaseChatModelWrapper):
    """
    Azure OpenAI 서비스를 위한 Alan 래퍼 클래스
    
    Azure OpenAI 서비스를 통해 OpenAI 모델을 사용할 수 있습니다.
    환경 변수에서 기본 설정을 가져오며, 추가 설정은 kwargs로 전달할 수 있습니다.
    """
    def __init__(self, **kwargs):
        """
        Azure OpenAI 채팅 모델 래퍼를 초기화합니다.
        
        기본 설정(DEFAULT_AZUREOPENAI_KWARGS)과 전달받은 kwargs를 병합하여
        AzureChatOpenAI 인스턴스를 생성합니다.
        
        Args:
            **kwargs: AzureChatOpenAI 생성자에 전달할 추가 매개변수들
                     기본 설정을 덮어쓸 수 있습니다.
        """
        # kwargs와 DEFAULT_AZUREOPENAI_KWARGS를 병합할 때,
        # kwargs에 없는 값은 DEFAULT_AZUREOPENAI_KWARGS에서 가져오고,
        # kwargs에 있는 값은 DEFAULT_AZUREOPENAI_KWARGS의 값을 덮어씁니다.
        super().__init__(
            AzureChatOpenAI(
                **{**DEFAULT_AZUREOPENAI_KWARGS, **kwargs}
            )
        )

if __name__ == '__main__':
    # 테스트 실행 코드
    from typing import TypedDict

    # 테스트용 구조화된 출력 스키마
    class TestOutput(TypedDict):
        msg: str
        response: str

    # Azure OpenAI 모델을 사용하여 구조화된 출력 테스트
    llm = AlanAzureChatOpenAI(**DEFAULT_AZUREOPENAI_KWARGS).with_structured_output(TestOutput)
    result = llm.invoke("hi")
    print(result)