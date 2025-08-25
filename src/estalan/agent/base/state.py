from typing import Annotated, Sequence, Literal, TypedDict


from langchain_core.messages import BaseMessage
from langgraph.prebuilt.chat_agent_executor import AgentStateWithStructuredResponse

from estalan.agent.base.reducer_function import add_messages_for_alan, update_metadata


class AlanAgentMetaData(TypedDict, total=False):
    chat_status: Literal["available", "unavailable"]
    status: Literal["start", "finish"]


class BaseAlanAgentState(AgentStateWithStructuredResponse):
    messages: Annotated[Sequence[BaseMessage], add_messages_for_alan]
    metadata: Annotated[AlanAgentMetaData, update_metadata]


class Canvas(TypedDict):
    type: Literal["markdown", "slide", "html", "txt", "image"]
    metadata: dict


class AlanAgentStateWithCanvas(BaseAlanAgentState):
    canvases: list[Canvas]


