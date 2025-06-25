import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from estalan.llm.base import AlanBaseChatModelWrapper


DEFAULT_AZUREOPENAI_KWARGS = {
    "azure_endpoint" : os.getenv("AZURE_ENDPOINT"),
    "openai_api_type" : os.getenv("AZURE_OPENAI_API_TYPE"),
    "openai_api_version" : os.getenv("AZURE_OPENAI_API_VERSION"),
    "openai_api_key" : os.getenv("AZURE_OPENAI_API_KEY"),
    "model_name" : os.getenv("AZURE_OPENAI_MODEL_NAME"),
    "deployment_name" : os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
}


class AlanChatOpenAI(AlanBaseChatModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(ChatOpenAI(**kwargs))


class AlanAzureChatOpenAI(AlanBaseChatModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            AzureChatOpenAI(
                **kwargs,
                ** DEFAULT_AZUREOPENAI_KWARGS
            )
        )

