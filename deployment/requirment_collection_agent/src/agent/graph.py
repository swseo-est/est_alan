from estalan.agent.graph.requirement_collection_agent import create_requirement_collection_agent
from dotenv import load_dotenv
import asyncio


load_dotenv()

_graph = None
_lock = asyncio.Lock()


async def get_graph():
    global _graph

    async with _lock:
        if _graph is None:
            _graph = create_requirement_collection_agent()

    return _graph