import uuid
from typing import Optional
from pydantic import Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage


class BaseAlanMessage(BaseMessage):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), coerce_numbers_to_str=True)


class AlanAIMessage(AIMessage, BaseAlanMessage):
    pass


class AlanHumanMessage(HumanMessage, BaseAlanMessage):
    pass


class AlanSystemMessage(SystemMessage, BaseAlanMessage):
    pass


class AlanToolMessage(ToolMessage, BaseAlanMessage):
    pass