from __future__ import annotations
import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.graph import CompiledGraph

from estalan.utils import load_config_json

@dataclass(slots=True)
class AlanMCPHost:
    """
    Life-cycle
    ----------
    _sessions와 _client 인스턴스가 유지되어야, mcp 도구를 외부에서 불러와 사용가능
    
    1. `await MCPAgent.create(...)`   →  클래스메서드로 비동기 인스턴스 생성
    2. `await agent.ainvoke(text)`    →  대화 호출
    3. 필요 시 `await agent.aclose()` →  세션 정리
    """
    _agent: Any                                       # LangGraph Agent 객체
    _stack: AsyncExitStack                            # 세션/툴 생명주기 관리
    _client: MultiServerMCPClient                      # MCP 클라이언트
    _sessions: Any                   # 유지되는 MCP 세션
    # _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @classmethod
    async def create_from_config_file(cls,
                                      filename,
                                        *args,
                                        tools: list = None,
                                        **kwargs,
                                    ) -> AlanMCPHost:

        server_configs = load_config_json(filename)
        result = await cls.create(server_configs, *args, tools=tools, **kwargs)

        return result

    @classmethod
    async def create(
        cls,
        server_configs,
        *args,
        tools: list = None,
        **kwargs,
    ) -> AlanMCPHost:
        if tools is None:
            tools = list()

        stack = AsyncExitStack()
        await stack.__aenter__()                 # 세션 관리용 컨텍스트 열기

        # 1) MCP 세션 한 번만 열고 보관
        client = MultiServerMCPClient(server_configs)
        sessions = [
            await stack.enter_async_context(client.session(name))
            for name in server_configs
        ]

        # 2) MCP 툴 로드
        tools_mcp = sum(await asyncio.gather(*map(load_mcp_tools, sessions)), [])
        tools = tools + tools_mcp

        # 3) LangGraph Agent 생성
        if 'checkpointer' not in kwargs:
            kwargs['checkpointer'] = InMemorySaver()
        
        agent = create_react_agent(
            *args,
            tools=tools,
            **kwargs,
        )

        return cls(
            _agent=agent,
            _stack=stack,
            _sessions=sessions,
            _client=client,
        )

    def get_graph(self) -> CompiledGraph:
        return self._agent

    async def aclose(self):
        """세션 및 자원 해제(옵션)."""
        await self._stack.aclose()

    def __getattr__(self, name: str):
        """
        self 에서 못 찾은 속성·메서드는 내부 _agent 로 위임.
        예) agent.invoke(), agent.astream(), agent.tools 등 전부 사용 가능.
        """
        return getattr(self._agent, name)


async def create_mcp_agent(
        server_configs,
        *args,
        tools: list = None,
        **kwargs,
    ):
    host = await AlanMCPHost.create(server_configs, *args, tools=tools, **kwargs)
    graph = host.get_graph()
    return graph, host