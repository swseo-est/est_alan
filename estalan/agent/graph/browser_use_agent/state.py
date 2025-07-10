from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import Literal
from typing import List, TypedDict, Literal, Annotated
import operator


# ---------------------------------------------------------------------------
# 공유 구조체(Shared Structures)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 상위 에이전트 공유 상태
# ---------------------------------------------------------------------------
class BrowserUseAgentState(AgentState):
    agent_goal: str
    plans: list
    current_plan: str
    navigator_goal: str

    next_node: Literal["end", "navigator_node", "supervisor_node"]
    navigation_history: Annotated[list, operator.add]
