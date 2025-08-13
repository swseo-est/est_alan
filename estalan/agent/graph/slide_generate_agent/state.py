from langchain_core.messages import BaseMessage

from typing import List, Annotated, TypedDict, Optional, Sequence
from langgraph.graph.message import add_messages


class Section(TypedDict):
    slide_type: str # title, contents, etc

    topic: str

    idx: int
    name: str
    description: str
    requirements: List[str]
    research: bool

    content: str
    img_url: str

    design: str
    html: str
    width: int
    height: int

    design_prompt: str


class ExecutorState(Section):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class SlideGenerateAgentMetadata(TypedDict):
    topic: str
    requirements: str
    num_sections: int
    num_slides: int
    status: str
