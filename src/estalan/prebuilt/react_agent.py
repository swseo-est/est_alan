from typing import Dict, Any

import langgraph.prebuilt
from langgraph.graph import START, END, StateGraph


def update_structured_response(state: Dict[str, Any]) -> Dict[str, Any]:
    structured_response = state.get("structured_response", {})
    return structured_response

def create_react_agent(*args, state_schema=None, pre_agent_node=None, post_agent_node=None, name=None, **kwargs):
    react_agent = langgraph.prebuilt.create_react_agent(*args, **kwargs)

    builder = StateGraph(state_schema)

    # Add pre_agent_node if provided
    if pre_agent_node is not None:
        builder.add_node("pre_agent_node", pre_agent_node)

        builder.add_edge(START, "pre_agent_node")
        builder.add_edge("pre_agent_node", "react_agent")
    else:
        builder.add_edge(START, "react_agent")

    builder.add_node("react_agent", react_agent)
    builder.add_node("update_structured_response", update_structured_response)

    # Add post_agent_node if provided
    if post_agent_node is not None:
        builder.add_node("post_agent_node", post_agent_node)

        builder.add_edge("react_agent", "update_structured_response")
        builder.add_edge("update_structured_response", "post_agent_node")
        builder.add_edge("post_agent_node", END)
    else:
        builder.add_edge("react_agent", "update_structured_response")
        builder.add_edge("update_structured_response", END)

    return builder.compile(name=name)

