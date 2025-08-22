import pytest
import asyncio
import os
from estalan.llm.estalan_google_vertexai import AlanChatVertexAI
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(
    not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    reason="GOOGLE_APPLICATION_CREDENTIALS 환경변수가 설정되지 않음"
)
def test_alan_chat_vertexai_invoke():
    """AlanChatVertexAI의 invoke 메서드 테스트"""
    # 테스트 실행
    llm = AlanChatVertexAI(model="gemini-2.5-flash")
    result = llm.invoke("hi")
    
    # 검증
    assert result is not None
    assert hasattr(result, 'content')
    assert isinstance(result.content, str)


@pytest.mark.skipif(
    not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    reason="GOOGLE_APPLICATION_CREDENTIALS 환경변수가 설정되지 않음"
)
@pytest.mark.asyncio
async def test_alan_chat_vertexai_ainvoke():
    """AlanChatVertexAI의 ainvoke 메서드 테스트"""
    # 테스트 실행
    llm = AlanChatVertexAI(model="gemini-2.5-flash")
    result = await llm.ainvoke("hi")
    
    # 검증
    assert result is not None
    assert hasattr(result, 'content')
    assert isinstance(result.content, str)
