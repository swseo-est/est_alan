import asyncio
from estalan.tools.mcp.host import create_mcp_agent
from estalan.utils import load_config_json

from dotenv import load_dotenv

load_dotenv()


configs = load_config_json("config.json")

model_name = configs["model"]
server_configs = configs["mcpServers"]

_graph = None          # CompiledGraph (lazy)
_host  = None
_lock  = asyncio.Lock()

async def get_graph():
    """LangGraph CLI가 불러갈 callable. 최초 호출 때만 세션을 연다."""
    global _graph, _host

    async with _lock:
        if _graph is None:
            _graph, _host = await create_mcp_agent(
                server_configs,
                model=model_name
            )
    return _graph
