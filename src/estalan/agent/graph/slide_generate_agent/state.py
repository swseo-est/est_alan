from langchain_core.messages import BaseMessage

from typing import List, Annotated, TypedDict, Optional, Sequence
from langgraph.graph.message import add_messages
from estalan.agent.base.state import BaseAlanAgentState, AlanAgentMetaData
from estalan.agent.base.reducer_function import update_metadata
import operator

from estalan.prebuilt.requirement_analysis_agent import Requirement


class Section(TypedDict):
    description: str
    requirements: List[str]
    research: bool

    slide_type: str # title, contents, etc
    topic: str
    idx: int
    name: str

    content: str
    img_url: str

    design: str
    html_template: str

    html: str
    width: int
    height: int

    design_prompt: str


class ExecutorState(Section):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    template_folder: str


class SlideGenerateAgentMetadata(AlanAgentMetaData):
    topic: str
    requirements: str
    num_sections: int
    num_slides: int


class SlideGenerateAgentState(BaseAlanAgentState):
    sections: List[Section]
    slides: Annotated[List[Section], operator.add]

    metadata: Annotated[SlideGenerateAgentMetadata, update_metadata]

    requirements: list[Requirement]  # 수집된 모든 요구사항
    requirements_docs: str
