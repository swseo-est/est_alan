import asyncio
from estalan.agent.graph.slide_generate_agent.graph import create_graph

_graph = None
_lock = None


async def get_graph():
    global _graph, _lock

    if _lock is None:
        _lock = asyncio.Lock()  # 현재 이벤트 루프에서 Lock 생성

    async with _lock:
        if _graph is None:
            _graph = create_graph()

    return _graph
