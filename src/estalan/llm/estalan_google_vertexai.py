import os

from google.oauth2 import service_account
from langchain_google_vertexai import ChatVertexAI
from estalan.llm.base import AlanBaseChatModelWrapper


class AlanChatVertexAI(AlanBaseChatModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            ChatVertexAI(
                **kwargs,
                credentials=service_account.Credentials.from_service_account_file(
                    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                ),
            )
        )