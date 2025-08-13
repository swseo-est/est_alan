from estalan.tools.mcp.client import AlanMCPClient
from langgraph.prebuilt import create_react_agent
from estalan.agent.graph.browser_use_agent.subgraph.navigater_agent import NavigatorOutput, create_navigator_agent, navigator_prompt
import asyncio
from estalan.agent.graph.browser_use_agent.state import BrowserUseAgentState


from langgraph_supervisor import create_supervisor


if __name__ == "__main__":
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
    _lock = asyncio.Lock()


    async def main():
        global client, tools

        async with _lock:
            client = await AlanMCPClient.create(server_configs)
            tools = client.tools

        navigator_agent = create_react_agent(
            model="openai:gpt-4.1",
            tools=tools,
            response_format=NavigatorOutput,
            prompt = navigator_prompt
        )

        state = BrowserUseAgentState()
        #
        # navigator_agent = create_navigator_agent(navigator_llm)
        workflow = create_supervisor(
            [navigator_agent],
            model="openai:gpt-4.1",
            prompt=(
                "당신은 웹 브라우저 자동화 도구를 사용하는 browser-use agent를 관리하는 감독자입니다."
                "유저의 요구사항을 수행하기 위한 명령을 browser-use agent에게 전달하세요"
                "browser-use agent에게 단위 작업 형태로 명령을 전달해야합니다."
                "navigator_goal에 단기 목표를 넣어서 명령을 전달하세요"
                "navigator_result이 실패라면 navigator_error를 참고해서 다음 계획을 수립하고 작업을 진행하세요"
            ),
            state_schema=state
        )

        # Compile and run
        app = workflow.compile()
        return app
