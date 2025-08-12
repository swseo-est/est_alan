from langgraph.prebuilt.chat_agent_executor import AgentState

class BaseAlanAgentState(AgentState):
    next_step: str
    last_step: str


def private_state_updater(state1, state2):
    for key, value in state2.items():
        state1[key] = value
    
    return state1
