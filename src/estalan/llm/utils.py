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

try:
    from estalan.llm.estalan_anthropic import AlanChatAnthropicVertex
    HAS_ANTHROPIC_VERTEXAI = True
except ImportError:
    HAS_ANTHROPIC_VERTEXAI = False  


def create_chat_model(provider=None, model=None, structured_output=None, lazy=True):
    available_providers = [
        "openai",
        "azure_openai",
        "google_vertexai",
        "anthropic",
        "anthropic_vertexai",
    ]
    if provider not in available_providers:
        raise Exception(f"Unsupported provider: {provider}. Available providers: {available_providers}")
    
    def create_instance():
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
        elif provider == "anthropic_vertexai":
            if not HAS_ANTHROPIC_VERTEXAI:
                raise ImportError("VertexAI support is not available. Please install langchain_anthropic_vertexai.")
            chat_model = AlanChatAnthropicVertex(model=model)
        else:
            raise Exception(f"Unsupported provider: {provider}")

        if structured_output is not None:
            chat_model = chat_model.with_structured_output(structured_output)

        return chat_model
    
    if lazy:
        # Lazy proxy class that creates new instance on each method call
        class LazyChatModel:
            def __init__(self, create_func):
                self._create_func = create_func
            
            def __getattr__(self, name):
                # Create new instance and delegate the method call
                instance = self._create_func()
                return getattr(instance, name)
            
            def invoke(self, *args, **kwargs):
                instance = self._create_func()
                return instance.invoke(*args, **kwargs)
            
            def ainvoke(self, *args, **kwargs):
                instance = self._create_func()
                return instance.ainvoke(*args, **kwargs)
            
            def stream(self, *args, **kwargs):
                instance = self._create_func()
                return instance.stream(*args, **kwargs)
            
            def astream(self, *args, **kwargs):
                instance = self._create_func()
                return instance.astream(*args, **kwargs)
            
            def with_structured_output(self, *args, **kwargs):
                instance = self._create_func()
                return instance.with_structured_output(*args, **kwargs)
        
        return LazyChatModel(create_instance)
    else:
        return create_instance()

if __name__ == '__main__':
    llm = create_chat_model(provider="google_vertexai", model="gemini-2.5-flash", lazy=True)
    result = llm.invoke("테스트")
    print(result)