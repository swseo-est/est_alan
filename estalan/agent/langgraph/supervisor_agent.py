import langgraph_supervisor
from langgraph.graph import StateGraph


def create_supervisor(*args, **kwargs) -> StateGraph:
    supervisor_agent = langgraph_supervisor.create_supervisor(*args, **kwargs)
    return supervisor_agent
