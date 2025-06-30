import asyncio
from estalan.tools.mcp.host import create_mcp_agent
from dotenv import load_dotenv


load_dotenv()


SERVER_CONFIGS = {
    "playwright": {
        "url": "http://localhost:8931/mcp",
        "transport": "streamable_http",
    },
    "notion":{
        "url": "https://server.smithery.ai/@smithery/notion/mcp?profile=scary-turtle-Ljf5sY&api_key=a457b5a4-cd03-4a13-b2ac-cf99c04f6fc4",
        "transport": "streamable_http",
    },
    "sequential_thinking":{
        "url": "https://server.smithery.ai/@kiennd/reference-servers/mcp?api_key=a457b5a4-cd03-4a13-b2ac-cf99c04f6fc4",
        "transport":  "streamable_http",
    },

}


_graph = None          # CompiledGraph (lazy)
_host  = None
_lock  = asyncio.Lock()

async def get_graph():
    """LangGraph CLI가 불러갈 callable. 최초 호출 때만 세션을 연다."""
    global _graph, _host
    async with _lock:
        if _graph is None:
            _graph, _host = await create_mcp_agent(
                SERVER_CONFIGS,
                model="openai:gpt-4.1",
            )
    return _graph
