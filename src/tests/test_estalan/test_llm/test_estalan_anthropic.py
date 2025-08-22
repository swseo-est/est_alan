import pytest
import asyncio
import os
from estalan.llm.estalan_anthropic import AlanChatAnthropic, AlanChatAnthropicVertex
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY 환경변수가 설정되지 않음"
)
def test_alan_chat_anthropic_invoke():
    """AlanChatAnthropic의 invoke 메서드 테스트"""
    # 테스트 실행
    llm = AlanChatAnthropic(model="claude-3-7-sonnet-20250219")
    result = llm.invoke("hi")
    
    # 검증
    assert result is not None
    assert hasattr(result, 'content')
    assert isinstance(result.content, str)


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY 환경변수가 설정되지 않음"
)
@pytest.mark.asyncio
async def test_alan_chat_anthropic_ainvoke():
    """AlanChatAnthropic의 ainvoke 메서드 테스트"""
    # 테스트 실행
    llm = AlanChatAnthropic(model="claude-3-7-sonnet-20250219")
    result = await llm.ainvoke("hi")
    
    # 검증
    assert result is not None
    assert hasattr(result, 'content')
    assert isinstance(result.content, str)


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_VERTEXAI_CREDENTIALS") or not os.getenv("ANTHROPIC_VERTEXAI_LOCATION"),
    reason="ANTHROPIC_VERTEXAI_CREDENTIALS 또는 ANTHROPIC_VERTEXAI_LOCATION 환경변수가 설정되지 않음"
)
def test_alan_chat_anthropic_vertex_invoke():
    """AlanChatAnthropicVertex의 invoke 메서드 테스트"""
    # 테스트 실행
    llm = AlanChatAnthropicVertex(model="claude-sonnet-4@20250514")
    result = llm.invoke("hi")
    
    # 검증
    assert result is not None
    assert hasattr(result, 'content')
    assert isinstance(result.content, str)


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_VERTEXAI_CREDENTIALS") or not os.getenv("ANTHROPIC_VERTEXAI_LOCATION"),
    reason="ANTHROPIC_VERTEXAI_CREDENTIALS 또는 ANTHROPIC_VERTEXAI_LOCATION 환경변수가 설정되지 않음"
)
@pytest.mark.asyncio
async def test_alan_chat_anthropic_vertex_ainvoke():
    """AlanChatAnthropicVertex의 ainvoke 메서드 테스트"""
    # 테스트 실행
    llm = AlanChatAnthropicVertex(model="claude-sonnet-4@20250514")
    result = await llm.ainvoke("hi")
    
    # 검증
    assert result is not None
    assert hasattr(result, 'content')
    assert isinstance(result.content, str)
