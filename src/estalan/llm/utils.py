# 조건부 import
try:
    from estalan.llm.estalan_openai import AlanChatOpenAI, AlanAzureChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from estalan.llm.estalan_anthropic import AlanChatAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from estalan.llm.estalan_google_vertexai import AlanChatVertexAI
    HAS_GOOGLE_VERTEXAI = True
except ImportError:
    HAS_GOOGLE_VERTEXAI = False

from estalan.llm.mock_llm import AlanMockLLM


def create_chat_model(provider=None, model=None, structured_output=None):
    available_providers = [
        "openai",
        "azure_openai",
        "google_vertexai",
        "anthropic",
        "mock"
    ]
    if provider not in available_providers:
        raise Exception(f"Unsupported provider: {provider}. Available providers: {available_providers}")
    
    if provider == "openai":
        if not HAS_OPENAI:
            raise ImportError("OpenAI support is not available. Please install langchain_openai.")
        chat_model = AlanChatOpenAI(model=model)
    elif provider == "azure_openai":
        if not HAS_OPENAI:
            raise ImportError("Azure OpenAI support is not available. Please install langchain_openai.")
        chat_model = AlanAzureChatOpenAI(model=model)
    elif provider == "google_vertexai":
        if not HAS_GOOGLE_VERTEXAI:
            raise ImportError("Google VertexAI support is not available. Please install langchain_google_vertexai.")
        chat_model = AlanChatVertexAI(model=model)
    elif provider == "anthropic":
        if not HAS_ANTHROPIC:
            raise ImportError("Anthropic support is not available. Please install langchain_anthropic.")
        chat_model = AlanChatAnthropic(model=model)
    elif provider == "mock":
        chat_model = AlanMockLLM(model=model)
    else:
        raise Exception(f"Unsupported provider: {provider}")

    if structured_output is not None:
        chat_model = chat_model.with_structured_output(structured_output)

    return chat_model
