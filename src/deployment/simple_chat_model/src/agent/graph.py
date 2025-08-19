from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from estalan.llm import create_chat_model

from dotenv import load_dotenv


load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


def get_graph():
    graph_builder = StateGraph(State)

    llm = create_chat_model(provider="azure_openai", model="gpt-4o-mini")

    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph = graph_builder.compile()
    return graph
