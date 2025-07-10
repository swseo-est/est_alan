from __future__ import annotations
import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


@dataclass(slots=True)
class AlanMCPClient:
    """
    Life-cycle
    ----------
    _sessions와 _client 인스턴스가 유지되어야, mcp 도구를 외부에서 불러와 사용가능
    
    1. `await MCPAgent.create(...)`   →  클래스메서드로 비동기 인스턴스 생성
    2. `await agent.ainvoke(text)`    →  대화 호출
    3. 필요 시 `await agent.aclose()` →  세션 정리
    """
    tools: Any # langgraph tool
    _stack: AsyncExitStack                            # 세션/툴 생명주기 관리
    _client: MultiServerMCPClient                      # MCP 클라이언트
    _sessions: Any                   # 유지되는 MCP 세션
    # _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


    @classmethod
    async def create(
        cls,
        server_configs,
        *args,
        **kwargs,
    ) -> "AlanMCPClient":
        """
        비동기적으로 AlanMCPClient 인스턴스를 생성합니다.
        """
        stack = AsyncExitStack()
        await stack.__aenter__()                 # 세션 관리용 컨텍스트 열기

        # 1) MCP 세션 한 번만 열고 보관
        client = MultiServerMCPClient(server_configs)
        sessions = [
            await stack.enter_async_context(client.session(name))
            for name in server_configs
        ]

        # 2) MCP 툴 로드
        tools = sum(await asyncio.gather(*map(load_mcp_tools, sessions)), [])

        return cls(
            tools=tools,
            _stack=stack,
            _sessions=sessions,
            _client=client,
        )

    def get_tools(self) -> list:
        return self._tools

    async def aclose(self):
        """세션 및 자원 해제(옵션)."""
        await self._stack.aclose()

    def __getattr__(self, name: str):
        """
        self 에서 못 찾은 속성·메서드는 내부 _agent 로 위임.
        예) agent.invoke(), agent.astream(), agent.tools 등 전부 사용 가능.
        """
        return getattr(self._agent, name)


async def create_mcp_tools(
        server_configs,
        *args,
        **kwargs,
    ):
    client = await AlanMCPClient.create(server_configs, *args, **kwargs)
    tools = client.get_tools()
    return client, tools


if __name__ == '__main__':
    # run npx @playwright/mcp@latest --port 8931
    from estalan.utils import load_config_json

    configs = load_config_json("test_config.json")

    model_name = configs["model"]
    server_configs = configs["mcpServers"]
    print(server_configs)

    client = asyncio.run(AlanMCPClient.create(server_configs))
    tools = client.tools
    print(tools)
