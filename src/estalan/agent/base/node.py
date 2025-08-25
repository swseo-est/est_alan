from typing import Dict, Any
from estalan.messages.utils import create_ai_message


def alan_agent_start_node(state: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict()
    metadata["chat_status"] = "unavailable"
    dummy = create_ai_message(content="")

    return {"metadata": metadata, "messages": [dummy]}


def alan_agent_finish_node(state: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict()
    metadata["chat_status"] = "available"
    dummy = create_ai_message(content="")

    return {"metadata": metadata, "messages": [dummy]}
