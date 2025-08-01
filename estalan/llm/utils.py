from estalan.llm.estalan_openai import AlanChatOpenAI, AlanAzureChatOpenAI
from estalan.llm.estalan_google_vertexai import AlanChatVertexAI
from estalan.llm.estalan_anthropic import AlanChatAnthropic


def create_chat_model(provider=None, model=None):
    available_providers = [
        "openai",
        "azure_openai",
        "google_vertaxai",
        "anthropic"
    ]
    if provider not in available_providers:
        raise Exception()
    elif provider == "openai":
        chat_model = AlanChatOpenAI(model=model)
    elif provider == "azure_openai":
        chat_model = AlanAzureChatOpenAI(model=model)
    elif provider == "google_vertaxai":
        chat_model = AlanChatVertexAI(model=model)
    elif provider == "anthropic":
        chat_model = AlanChatAnthropic(model=model)
    else:
        raise Exception()

    return chat_model
