import os
from google.oauth2 import service_account

from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from estalan.llm.base import AlanBaseChatModelWrapper


class AlanChatAnthropic(AlanBaseChatModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(ChatAnthropic(**kwargs))


class AlanChatAnthropicVertex(AlanBaseChatModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            ChatAnthropicVertex(
                **kwargs,
                credentials=service_account.Credentials.from_service_account_file(
                    os.getenv("ANTHROPIC_VERTEXAI_CREDENTIALS")
                ).with_scopes(["https://www.googleapis.com/auth/cloud-platform"]),
                location=os.getenv("ANTHROPIC_VERTEXAI_LOCATION")
            )
        )


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    llm = AlanChatAnthropicVertex(
        model="claude-sonnet-4@20250514",
    )
    result = llm.invoke("hi")
    print(result)