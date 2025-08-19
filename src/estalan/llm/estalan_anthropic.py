from langchain_anthropic import ChatAnthropic
from estalan.llm.base import AlanBaseChatModelWrapper


class AlanChatAnthropic(AlanBaseChatModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(ChatAnthropic(**kwargs))


