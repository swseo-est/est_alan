from langchain_core.messages import BaseMessage

from typing import List, Annotated, TypedDict, Optional, Sequence
from estalan.agent.graph.slide_generate_agent.planning_agent import  Section
from langgraph.graph.message import add_messages



class ExecutorState(Section):
    messages: Annotated[Sequence[BaseMessage], add_messages]
