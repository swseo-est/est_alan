import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from estalan.llm.base import AlanBaseChatModelWrapper
from dotenv import load_dotenv

load_dotenv()

DEFAULT_AZUREOPENAI_KWARGS = {
    "azure_endpoint" : os.getenv("AZURE_ENDPOINT"),
    "openai_api_version" : os.getenv("AZURE_OPENAI_API_VERSION"),
    "openai_api_key" : os.getenv("AZURE_OPENAI_API_KEY"),
    "deployment_name" : os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
}


class AlanChatOpenAI(AlanBaseChatModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(ChatOpenAI(**kwargs))


class AlanAzureChatOpenAI(AlanBaseChatModelWrapper):
    def __init__(self, **kwargs):
        # kwargs와 DEFAULT_AZUREOPENAI_KWARGS를 병합할 때,
        # kwargs에 없는 값은 DEFAULT_AZUREOPENAI_KWARGS에서 가져오고,
        # kwargs에 있는 값은 DEFAULT_AZUREOPENAI_KWARGS의 값을 덮어씁니다.
        super().__init__(
            AzureChatOpenAI(
                **{**DEFAULT_AZUREOPENAI_KWARGS, **kwargs}
            )
        )

if __name__ == '__main__':
    llm = AlanAzureChatOpenAI(model="gpt-4.1")
    print(llm.invoke("hi"))