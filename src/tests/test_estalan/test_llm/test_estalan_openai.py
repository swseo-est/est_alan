import pytest
import asyncio
import os
from estalan.llm.estalan_openai import AlanChatOpenAI, AlanAzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY 환경변수가 설정되지 않음"
)
def test_alan_chat_openai_invoke():
    """AlanChatOpenAI의 invoke 메서드 테스트"""
    # 테스트 실행
    llm = AlanChatOpenAI(model="gpt-4o-mini")
    result = llm.invoke("hi")
    
    # 검증
    assert result is not None
    assert hasattr(result, 'content')
    assert isinstance(result.content, str)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY 환경변수가 설정되지 않음"
)
@pytest.mark.asyncio
async def test_alan_chat_openai_ainvoke():
    """AlanChatOpenAI의 ainvoke 메서드 테스트"""
    # 테스트 실행
    llm = AlanChatOpenAI(model="gpt-4o-mini")
    result = await llm.ainvoke("hi")
    
    # 검증
    assert result is not None
    assert hasattr(result, 'content')
    assert isinstance(result.content, str)


@pytest.mark.skipif(
    not all([
        os.getenv("AZURE_ENDPOINT"),
        os.getenv("AZURE_OPENAI_API_VERSION"),
        os.getenv("AZURE_OPENAI_API_KEY"),
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    ]),
    reason="Azure OpenAI 관련 환경변수가 모두 설정되지 않음"
)
def test_alan_azure_chat_openai_invoke():
    """AlanAzureChatOpenAI의 invoke 메서드 테스트"""
    # 테스트 실행
    llm = AlanAzureChatOpenAI()
    result = llm.invoke("hi")
    
    # 검증
    assert result is not None
    assert hasattr(result, 'content')
    assert isinstance(result.content, str)


@pytest.mark.skipif(
    not all([
        os.getenv("AZURE_ENDPOINT"),
        os.getenv("AZURE_OPENAI_API_VERSION"),
        os.getenv("AZURE_OPENAI_API_KEY"),
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    ]),
    reason="Azure OpenAI 관련 환경변수가 모두 설정되지 않음"
)
@pytest.mark.asyncio
async def test_alan_azure_chat_openai_ainvoke():
    """AlanAzureChatOpenAI의 ainvoke 메서드 테스트"""
    # 테스트 실행
    llm = AlanAzureChatOpenAI()
    result = await llm.ainvoke("hi")
    
    # 검증
    assert result is not None
    assert hasattr(result, 'content')
    assert isinstance(result.content, str)
