from typing import Dict, Any


def alan_agent_finish_node(state: Dict[str, Any]) -> Dict[str, Any]:

    metadata = state["metadata"].copy()
    metadata["status"] = "finish"
    return state