import asyncio
from estalan.agent.graph.slide_generate_agent.graph import create_graph

_graph = None
_lock = asyncio.Lock()


async def get_graph():
    global _graph

    async with _lock:
        if _graph is None:
            _graph = create_graph()

    return _graph
