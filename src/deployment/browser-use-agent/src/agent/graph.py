from estalan.tools.mcp.client import AlanMCPClient
from langgraph.prebuilt import create_react_agent
from estalan.agent.graph.browser_use_agent.subgraph.navigater_agent import NavigatorOutput, create_browser_use_agent, SupervisorOutput
import asyncio

from estalan.llm import create_chat_model

from dotenv import load_dotenv

load_dotenv()

server_configs = {
    "playwright": {
        "url": "http://localhost:8931/mcp",
        "transport": "streamable_http"
    }
}

client = None
tools = None
graph = None
_lock = asyncio.Lock()


async def get_graph():
    global client, tools, graph

    async with _lock:
        if graph is None:
            client = await AlanMCPClient.create(server_configs)
            tools = client.tools

            planer_llm = create_chat_model(provider="azure_openai", model="gpt-4.1").with_structured_output(SupervisorOutput)
            navigator_llm = create_chat_model(provider="azure_openai", model="gpt-4.1")

            navigator_agent = create_react_agent(
                model=navigator_llm,
                tools=tools,
                response_format=NavigatorOutput
            )

            graph = create_browser_use_agent(planer_llm, navigator_agent)

    return graph
