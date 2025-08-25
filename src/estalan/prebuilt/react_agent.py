from typing import Dict, Any

import langgraph.prebuilt
from langgraph.graph import START, END, StateGraph


def update_structured_response(state: Dict[str, Any]) -> Dict[str, Any]:
    structured_response = state.get("structured_response", {})
    return structured_response

def create_react_agent(*args, state_schema=None, name=None, **kwargs):
    react_agent = langgraph.prebuilt.create_react_agent(*args, **kwargs)

    builder = StateGraph(state_schema)

    builder.add_node("react_agent", react_agent)
    builder.add_node("update_structured_response", update_structured_response)

    builder.add_edge(START, "react_agent")
    builder.add_edge("react_agent", "update_structured_response")
    builder.add_edge("update_structured_response", END)

    return builder.compile(name=name)

