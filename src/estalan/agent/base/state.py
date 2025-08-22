from typing import Annotated, Sequence, Literal
from pydantic import BaseModel

from langchain_core.messages import BaseMessage
from langgraph.prebuilt.chat_agent_executor import AgentStateWithStructuredResponse

from estalan.agent.base.reducer_function import add_messages_for_alan


class BaseAlanAgentState(AgentStateWithStructuredResponse):
    messages: Annotated[Sequence[BaseMessage], add_messages_for_alan]
    metadata: dict



class Canvas(BaseModel):
    type: Literal["markdown", "slide", "html", "txt", "image"]
    metadata: dict


class AlanAgentStateWithCanvas(BaseAlanAgentState):
    canvases: list[Canvas]
