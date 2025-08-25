import pytest
import os
from estalan.llm.utils import create_chat_model
from dotenv import load_dotenv

load_dotenv()


def test_create_chat_model_invalid_provider():
    """잘못된 프로바이더로 create_chat_model 호출 시 예외 발생 테스트"""
    with pytest.raises(Exception, match="Unsupported provider"):
        create_chat_model(provider="invalid_provider")


# 프로바이더별 테스트 데이터 정의
PROVIDER_TEST_CASES = [
    pytest.param(
        "openai", 
        "gpt-4o-mini", 
        marks=pytest.mark.skipif(
            not os.getenv("OPENAI_API_KEY"),
            reason="OPENAI_API_KEY 환경변수가 설정되지 않음"
        )
    ),
    pytest.param(
        "azure_openai", 
        None, 
        marks=pytest.mark.skipif(
            not all([
                os.getenv("AZURE_ENDPOINT"),
                os.getenv("AZURE_OPENAI_API_VERSION"),
                os.getenv("AZURE_OPENAI_API_KEY"),
                os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            ]),
            reason="Azure OpenAI 관련 환경변수가 모두 설정되지 않음"
        )
    ),
    pytest.param(
        "google_vertexai", 
        "gemini-2.5-flash",
        marks=pytest.mark.skipif(
            not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            reason="GOOGLE_APPLICATION_CREDENTIALS 환경변수가 설정되지 않음"
        )
    ),
    pytest.param(
        "anthropic", 
        "claude-3-7-sonnet-20250219", 
        marks=pytest.mark.skipif(
            not os.getenv("ANTHROPIC_API_KEY"),
            reason="ANTHROPIC_API_KEY 환경변수가 설정되지 않음"
        )
    ),
    pytest.param(
        "anthropic_vertexai", 
        "claude-sonnet-4@20250514", 
        marks=pytest.mark.skipif(
            not os.getenv("ANTHROPIC_VERTEXAI_CREDENTIALS") or not os.getenv("ANTHROPIC_VERTEXAI_LOCATION"),
            reason="ANTHROPIC_VERTEXAI_CREDENTIALS 또는 ANTHROPIC_VERTEXAI_LOCATION 환경변수가 설정되지 않음"
        )
    ),
]


@pytest.mark.parametrize("provider,model", PROVIDER_TEST_CASES)
def test_create_chat_model_invoke(provider, model):
    """프로바이더별 invoke 메서드 테스트"""
    llm = create_chat_model(provider=provider, model=model, lazy=False)
    result = llm.invoke("hi")
    
    assert result is not None
    assert hasattr(result, 'content')
    assert isinstance(result.content, str)


@pytest.mark.parametrize("provider,model", PROVIDER_TEST_CASES)
@pytest.mark.asyncio
async def test_create_chat_model_ainvoke(provider, model):
    """프로바이더별 ainvoke 메서드 테스트"""
    llm = create_chat_model(provider=provider, model=model, lazy=False)
    result = await llm.ainvoke("hi")
    
    assert result is not None
    assert hasattr(result, 'content')
    assert isinstance(result.content, str)


# 특수 기능 테스트
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY 환경변수가 설정되지 않음"
)
def test_create_chat_model_lazy():
    """Lazy 모드로 create_chat_model 테스트"""
    llm = create_chat_model(provider="openai", model="gpt-4o-mini", lazy=True)
    result = llm.invoke("hi")
    
    assert result is not None
    assert hasattr(result, 'content')
    assert isinstance(result.content, str)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY 환경변수가 설정되지 않음"
)
def test_create_chat_model_with_structured_output():
    """구조화된 출력과 함께 create_chat_model 테스트"""
    from typing import TypedDict
    
    class TestOutput(TypedDict):
        message: str
        response: str
    
    llm = create_chat_model(provider="openai", model="gpt-4o-mini", structured_output=TestOutput, lazy=False)
    result = llm.invoke("hi")
    
    assert result is not None
    assert isinstance(result, dict)
    assert "message" in result
    assert "response" in result
