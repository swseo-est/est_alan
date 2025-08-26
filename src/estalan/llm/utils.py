# LLM 유틸리티 모듈
# 다양한 언어 모델 제공자를 통합하여 일관된 인터페이스로 접근할 수 있도록 하는 팩토리 함수를 제공합니다.

# 조건부 import - 각 제공자별 모듈이 설치되어 있는지 확인
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
    """
    지정된 제공자와 모델을 사용하여 채팅 모델을 생성합니다.
    
    이 함수는 다양한 언어 모델 제공자(OpenAI, Anthropic, Google VertexAI)를
    통합된 인터페이스로 접근할 수 있도록 하는 팩토리 함수입니다.
    
    Args:
        provider (str): 모델 제공자 ("openai", "azure_openai", "google_vertexai", "anthropic", "anthropic_vertexai")
        model (str): 사용할 모델 이름 (예: "gpt-4", "claude-3-sonnet", "gemini-2.0-flash")
        structured_output: 구조화된 출력을 위한 스키마 (선택사항)
        lazy (bool): 지연 초기화 사용 여부 (기본값: True)
                    True인 경우 실제 호출 시점에 모델 인스턴스가 생성됩니다.
    
    Returns:
        채팅 모델 인스턴스 또는 지연 초기화 프록시
        
    Raises:
        Exception: 지원하지 않는 제공자인 경우
        ImportError: 해당 제공자의 의존성이 설치되지 않은 경우
    """
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
        """
        실제 모델 인스턴스를 생성하는 내부 함수
        """
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

        # 구조화된 출력이 지정된 경우 적용
        if structured_output is not None:
            chat_model = chat_model.with_structured_output(structured_output)

        return chat_model
    
    if lazy:
        # 지연 초기화 프록시 클래스 - 각 메서드 호출 시 새로운 인스턴스 생성
        class LazyChatModel:
            """
            지연 초기화를 위한 프록시 클래스
            
            이 클래스는 실제 모델 인스턴스를 즉시 생성하지 않고,
            메서드가 호출될 때마다 새로운 인스턴스를 생성합니다.
            이를 통해 메모리 사용량을 줄이고 초기화 시간을 단축할 수 있습니다.
            """
            def __init__(self, create_func):
                """
                프록시를 초기화합니다.
                
                Args:
                    create_func: 실제 모델 인스턴스를 생성하는 함수
                """
                self._create_func = create_func
            
            def __getattr__(self, name):
                """
                속성 접근 시 새로운 인스턴스를 생성하고 해당 속성을 반환합니다.
                """
                # Create new instance and delegate the method call
                instance = self._create_func()
                return getattr(instance, name)
            
            def invoke(self, *args, **kwargs):
                """
                동기 호출 - 새로운 인스턴스를 생성하여 호출을 위임합니다.
                """
                instance = self._create_func()
                return instance.invoke(*args, **kwargs)
            
            def ainvoke(self, *args, **kwargs):
                """
                비동기 호출 - 새로운 인스턴스를 생성하여 호출을 위임합니다.
                """
                instance = self._create_func()
                return instance.ainvoke(*args, **kwargs)
            
            def stream(self, *args, **kwargs):
                """
                스트리밍 호출 - 새로운 인스턴스를 생성하여 호출을 위임합니다.
                """
                instance = self._create_func()
                return instance.stream(*args, **kwargs)
            
            def astream(self, *args, **kwargs):
                """
                비동기 스트리밍 호출 - 새로운 인스턴스를 생성하여 호출을 위임합니다.
                """
                instance = self._create_func()
                return instance.astream(*args, **kwargs)
            
            def with_structured_output(self, *args, **kwargs):
                """
                구조화된 출력 설정 - 새로운 인스턴스를 생성하여 설정을 위임합니다.
                """
                instance = self._create_func()
                return instance.with_structured_output(*args, **kwargs)
        
        return LazyChatModel(create_instance)
    else:
        # 즉시 초기화 - 모델 인스턴스를 바로 생성하여 반환
        return create_instance()

if __name__ == '__main__':
    # 테스트 실행 코드
    llm = create_chat_model(provider="google_vertexai", model="gemini-2.5-flash", lazy=True)
    result = llm.invoke("테스트")
    print(result)