from langchain.agents.chat.base import ChatAgent
from langchain_openai import ChatOpenAI

from estalan.tools.mcp.client import AlanMCPClient
from estalan.agent.graph.browser_use_agent.state import BrowserUseAgentState
from langgraph.prebuilt import create_react_agent
from typing import List, TypedDict, Literal, Annotated
from langchain_core.tools import tool
import asyncio
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage




# 1. Define structured output schema as Pydantic model
class NavigatorOutput(TypedDict):
    navigator_result: bool
    navigator_error: str
    navigator_message: str # 현재 작업과 관련된 ai 메시지, 다음 작업에 필요한 정보가 있을 경우 충분한 정보를 담아 메시지를 출력

class SupervisorOutput(TypedDict):
    agent_goal: str
    supervisor_messages: str # 현재 작업과 관련된 ai 메시지, 다음 작업에 필요한 정보가 있을 경우 충분한 정보를 담아 메시지를 출력

    current_plan: str
    plans: list

    navigator_goal: str
    next_node: Literal["end", "navigator_node", "supervisor_node"]

class PlanerOutput(TypedDict):
    next_plans: list

def create_browser_use_agent(supervisor_llm, navigator_llm, name=None):
    from typing import Dict
    async def navigator_node(state: BrowserUseAgentState) -> Dict:
        goal = state["navigator_goal"]

        prompt = f"""
        [역할]
        당신은 웹 브라우저를 제어할 수 있는 MCP 기반 에이전트입니다.
        
        당신은 다음과 같은 최종 목표를 수행하기 위해 단위 작업들을 수행하고 있습니다.
        - 최종목표 : {state['agent_goal']}
        - 현재까지 수행한 작업: {state['navigation_history']}
        
        이제 다음으로 아래와 같은 작업을 수행하세요.
        - {goal}        
        
        작업 수행 결과를 아래와 같이 출력하세요.
        - navigator_result: bool = Field(..., description="작업 성공/실패 결과")
        - navigator_error: str = Field(..., description="작업 실패 시 실패 사유를 기술")
        - navigator_message: str # 현재 작업과 관련된 ai 메시지, 다음 작업에 필요한 정보가 있을 경우 충분한 정보를 담아 메시지를 출력

        """

        response = await navigator_llm.ainvoke({"messages": prompt})
        result = dict(response["structured_response"])

        navigation_history = {
            "navigator_goal": goal,
            "navigator_result": result["navigator_result"],
            "navigator_error": result["navigator_error"],
        }
        return {
            "navigation_history": [navigation_history],
            "messages": [AIMessage(content=result['navigator_message'])]
        }

    def supervisor_node(state: BrowserUseAgentState) -> Dict:
        if state.get("plans") is None:
            system_instructions = f"""
            [역할]
            
            당신은 웹 브라우저 자동화 에이전트를 관리하는 supervisor 입니다.
            
            1. 사용자의 요구사항을 정리해서 agent_goal을 업데이트하세요        
            2. 요구사항을 달성하기 위한 계획을 5단계로 정리해서 plans를 업데이트하세요.
                ex)
                    ["1. ~~~~", "2. ~~~~", "3. ~~~~", "4. ~~~~", "5. ~~~~"]
            3. 1단계 계획을 current_plan에 업데이트하세요.
            4. next_node를 "supervisor_node"로 업데이트하세요
            """
        else:
            system_instructions = f"""
            [역할]

            당신은 웹 브라우저 자동화 에이전트를 관리하는 supervisor 입니다.
            
            1. {state.get("navigation_history")}을 참고하여, current_plan: {state.get("current_plan")}을 모두 완료했을 경우 current_plan을 다음 계획으로 업데이트하세요.
            2. current_plan를 달성하기 위한 단위 작업을 navigator_goal 에 입력하세요.
                - navigator_goal의 작업은 navigator_node에서 수행합니다.
            3. navigator_goal이 웹 브라우저를 통한 작업이 필요한 경우 next_node를 "navigator_node"로 업데이트하세요.
            4. 모든 계획을 완료한 경우 next_node를 end로 업데이트하세요.
            
            - supervisor_messages: str # 현재 작업과 관련된 ai 메시지, 다음 작업에 필요한 정보가 있을 경우 충분한 정보를 담아 메시지를 출력

            """

        response = supervisor_llm.invoke(
                    state["messages"] + [SystemMessage(content=system_instructions)]
            )

        if state.get("plans") is not None:
            response["plans"] = state["plans"]
            response["messages"] = [AIMessage(content=response['supervisor_messages'])]

        print(response)
        print(state.get("navigation_history"))
        return response

    builder = StateGraph(BrowserUseAgentState)
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("navigator_node", navigator_node)

    # Add edges
    builder.add_edge(START, "supervisor_node")
    builder.add_edge("navigator_node", "supervisor_node")


    def router(state):
        return END if state["next_node"] == "end" else state["next_node"]

    builder.add_conditional_edges(
        "supervisor_node",
        router,
        ["navigator_node", "supervisor_node", END],
    )

    browser_use_agent = builder.compile(name=name)

    return browser_use_agent


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

        planer_llm = ChatOpenAI(model="gpt-4.1").with_structured_output(SupervisorOutput)

        navigator_llm = create_react_agent(
            model="openai:gpt-4.1",
            tools=tools,
            response_format=NavigatorOutput
        )

        browser_use_agent = create_browser_use_agent(planer_llm, navigator_llm)
        await browser_use_agent.ainvoke({"messages": "최신 엔비디아 gpu 모델명을 알아내고, amazon에서 최저가 링크를 찾아줘"})


    asyncio.run(main())


