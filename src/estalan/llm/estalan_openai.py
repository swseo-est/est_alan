# OpenAI LLM 래퍼 모듈
# OpenAI와 Azure OpenAI 서비스를 위한 AlanBaseChatModelWrapper 구현체를 제공합니다.

import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from estalan.llm.base import AlanBaseChatModelWrapper
from dotenv import load_dotenv

load_dotenv()

def get_azure_openai_config(model: str = None) -> dict:
    """
    모델 이름에 따라 Azure OpenAI 설정을 반환합니다.
    
    Args:
        model (str): 모델 이름 (예: "gpt-5-mini", "gpt-4o", "gpt-4", "gpt-5")
        
    Returns:
        dict: Azure OpenAI 설정 딕셔너리
    """
    if not model:
        # 기본 설정 (기존 환경변수 사용)
        return {
            "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
            "openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
            "openai_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        }
    
    # 모델 이름을 소문자로 변환하고 파싱
    model_lower = model.lower()
    
    # GPT-5 계열 모델들 (gpt-5, gpt-5-mini, gpt-5o, gpt-5o-mini 등)
    if model_lower.startswith("gpt-5"):
        return {
            "azure_endpoint": os.getenv("AZURE_ENDPOINT_5"),
            "openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION_5"),
            "openai_api_key": os.getenv("AZURE_OPENAI_API_KEY_5"),
            "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_5"),
        }
    
    # GPT-4 계열 모델들 (gpt-4, gpt-4o, gpt-4-turbo, gpt-4o-mini 등)
    elif model_lower.startswith("gpt-4"):
        return {
            "azure_endpoint": os.getenv("AZURE_ENDPOINT_4"),
            "openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION_4"),
            "openai_api_key": os.getenv("AZURE_OPENAI_API_KEY_4"),
            "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_4"),
        }
    
    # 기타 모델들은 기본 설정 사용
    else:
        return {
            "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
            "openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
            "openai_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
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
    모델 이름에 따라 다른 환경변수를 사용하며, 추가 설정은 kwargs로 전달할 수 있습니다.
    """
    def __init__(self, **kwargs):
        """
        Azure OpenAI 채팅 모델 래퍼를 초기화합니다.
        
        모델 이름에 따라 적절한 환경변수를 선택하고, 전달받은 kwargs와 병합하여
        AzureChatOpenAI 인스턴스를 생성합니다.
        
        Args:
            **kwargs: AzureChatOpenAI 생성자에 전달할 추가 매개변수들
                     기본 설정을 덮어쓸 수 있습니다.
        """
        # 모델 이름 추출
        model = kwargs.get('model')
        
        # 모델별 설정 가져오기
        azure_config = get_azure_openai_config(model)
        
        # kwargs와 azure_config를 병합할 때,
        # kwargs에 없는 값은 azure_config에서 가져오고,
        # kwargs에 있는 값은 azure_config의 값을 덮어씁니다.
        super().__init__(
            AzureChatOpenAI(
                **{**azure_config, **kwargs}
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
    # GPT-5 계열 모델들 테스트
    gpt5_models = ["gpt-5", "gpt-5-mini", "gpt-5o", "gpt-5o-mini"]
    for model_name in gpt5_models:
        print(f"\n=== Testing {model_name} ===")
        llm = AlanAzureChatOpenAI(model=model_name)
        print(f"Endpoint: {llm.llm.azure_endpoint}")
        print(f"Deployment: {llm.llm.deployment_name}")
    
    # GPT-4 계열 모델들 테스트
    gpt4_models = ["gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-4o-mini"]
    for model_name in gpt4_models:
        print(f"\n=== Testing {model_name} ===")
        llm = AlanAzureChatOpenAI(model=model_name)
        print(f"Endpoint: {llm.llm.azure_endpoint}")
        print(f"Deployment: {llm.llm.deployment_name}")
    
    # 기타 모델 테스트
    other_models = ["gpt-3.5-turbo", "claude-3"]
    for model_name in other_models:
        print(f"\n=== Testing {model_name} ===")
        llm = AlanAzureChatOpenAI(model=model_name)
        print(f"Endpoint: {llm.llm.azure_endpoint}")
        print(f"Deployment: {llm.llm.deployment_name}")