import langgraph.prebuilt
from langgraph.graph import CompiledGraph


def create_react_agent(*args, **kwargs) -> CompiledGraph:
    react_agent = langgraph.prebuilt.create_react_agent(*args, **kwargs)
    return react_agent