from langgraph.prebuilt.chat_agent_executor import AgentState

class BaseAlanAgentState(AgentState):
    next_step: str
    last_step: str