from typing import Dict, Any


def alan_agent_start_node(state: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict()
    metadata["chat_status"] = "unavailable"

    return {"metadata": metadata}


def alan_agent_finish_node(state: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict()
    metadata["chat_status"] = "available"

    return {"metadata": metadata}
