from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage


class BaseAlanMessage(BaseMessage):
    pass


class AlanAIMessage(AIMessage, BaseAlanMessage):
    pass


class AlanHumanMessage(HumanMessage, BaseAlanMessage):
    pass


class AlanSystemMessage(SystemMessage, BaseAlanMessage):
    pass


class AlanToolMessage(ToolMessage, BaseAlanMessage):
    pass