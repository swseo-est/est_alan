import asyncio
from estalan.agent.graph.slide_generate_agent import get_

_graph = None
_lock = asyncio.Lock()


async def get_graph():
    global _graph

    async with _lock:
        if _graph is None:
            _graph = graph.create_graph()

    return _graph
