import json
import os
import re
import time
from typing import Any, Type

from dotenv import load_dotenv
from google.oauth2 import service_account
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import RemoveMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.utils import messages_from_dict
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBinding
from langchain_fireworks import ChatFireworks
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from alan.core.llm import DeepSeekR1_Continue
from alan.core.node import AlanState, QueryAnalysis, ToolCalling, route_tools
from alan.core.prompt import (
    CONTINUE_PROMPT,
    AlanPrompt,
    SuggestPrompt,
    VanillaChatPrompt,
)
from alan.logging_config import get_logger
from alan.tools.base import AsyncTool
from alan.tools.mixins import ChromeExtensionMixin, MessageMixin
from alan.tools.utils import add_graph_components

load_dotenv()
logger = get_logger(__name__)

STOP_REASON_MAPPING = {
    "end_turn": "stop",
    "max_tokens": "length",
}

default_config = {
    "configurable": {"thread_id": "1"},
    "recursion_limit": 200,
}  # TODO: recursion_limit에 도달했을 때의 error 처리가 되어 있어야 할 듯.

suggest_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    openai_api_type=os.getenv("AZURE_OPENAI_API_TYPE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model_name=os.getenv("AZURE_OPENAI_MODEL_NAME"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.1,
    max_tokens=1024,
    request_timeout=(10, 200),
)


class AlanAgent(BaseModel, ChromeExtensionMixin, MessageMixin):
    llm: BaseLanguageModel | RunnableBinding
    graph: CompiledStateGraph
    config: dict = default_config

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(
        cls,
        llm: BaseLanguageModel | RunnableBinding,
        llm_type: str,
        filter_llm: BaseLanguageModel | RunnableBinding,
        tools: list[AsyncTool],
        init_data: dict[str, Any] | None = None,
        max_tool_calls: int = 2,
    ):
        logger.info(f"Creating AlanAgent with LLM type: {llm_type}")
        logger.debug(f"Tools: {[tool.name for tool in tools]}")

        if init_data:
            logger.debug("Agent initialized with existing data")

        try:
            # Build Graph
            graph = cls._create_graph(
                llm=llm,
                llm_type=llm_type,
                filter_llm=filter_llm,
                tools=tools,
                init_data=init_data,
                config=default_config,
                max_tool_calls=max_tool_calls,
            )

            agent = AlanAgent(
                llm=llm,
                graph=graph,
            )

            logger.info("AlanAgent created successfully")
            return agent

        except Exception as e:
            logger.error(f"Failed to create AlanAgent: {str(e)}")
            raise

    @staticmethod
    def _create_graph(
        llm: BaseLanguageModel | RunnableBinding,
        llm_type: str,
        filter_llm: BaseLanguageModel | RunnableBinding,
        tools: list[AsyncTool],
        init_data: dict[str, Any] | None,
        config: dict,
        max_tool_calls: int,
    ) -> CompiledStateGraph:
        logger.debug("Creating agent graph")

        graph_builder = StateGraph(AlanState)

        guardrail_llm = ChatVertexAI(
            credentials=service_account.Credentials.from_service_account_file(
                os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            ),
            model_name="gemini-2.0-flash",
            temperature=0.1,
            max_tokens=1024,
            max_retries=2,
        )

        nodes = [
            (
                "query_analysis",
                QueryAnalysis(
                    llm,
                    AlanPrompt(llm_type),
                    tools,
                    guardrail_llm=guardrail_llm,
                    max_tool_calls=max_tool_calls,
                ),
            ),
            ("tool_calling", ToolCalling(tools, filter_llm=filter_llm)),
        ]
        edges = [
            (START, "query_analysis"),
            ("tool_calling", "query_analysis"),
        ]
        edges_with_conditions = [
            (
                "query_analysis",
                route_tools,
                {"tool_calling": "tool_calling", "__end__": "__end__"},
            )
        ]

        graph_builder = add_graph_components(
            StateGraph(AlanState), nodes, edges, edges_with_conditions
        )

        graph = graph_builder.compile(checkpointer=MemorySaver())

        if init_data:
            # 이전 질문에서의 연관 이미지 및 동영상 정보 제거.
            # Backend에서 질문마다 instance를 만드므로, 해당 위치에 로직 추가.
            # TODO: 더 적절한 구조 생각해보기.
            init_data["image_info"] = []
            init_data["video_info"] = []
            logger.debug("Reset image and video info in init_data")

            init_data["messages"] = messages_from_dict(init_data["messages"])
            logger.debug("Updating graph state with init_data")

            graph.update_state(config=config, values=init_data)

        logger.debug("Graph created successfully")
        return graph

    async def astream_events(
        self, user_input: str, stream_mode: str = "values", version="v2"
    ):
        logger.info(f"Starting stream events for user input: {user_input[:100]}...")
        try:
            return self.graph.astream_events(
                input={"messages": [HumanMessage(content=user_input)]},
                config=self.config,
                stream_mode=stream_mode,
                version=version,
            )
        except Exception as e:
            logger.error(f"Error in astream_events: {str(e)}")
            raise

    async def astream_continue_events(self, stream_mode: str = "values", version="v2"):
        logger.info("Starting continue stream events")

        async def _continue_generator():
            try:
                if isinstance(self.llm, ChatFireworks):  # DeepSeek-R1인 경우,
                    logger.debug("Using DeepSeek-R1 continue mode")

                    # Get current state messages
                    current_state_values = self.graph.get_state(
                        config=self.config
                    ).values
                    messages = current_state_values.get("messages", [])

                    if not messages:
                        raise ValueError(
                            "No messages found in current state for continue"
                        )

                    response = None
                    async for chunk in DeepSeekR1_Continue(
                        model=os.environ["DEEPSEEK_MODEL_NAME"],
                        base_url=os.environ["DEEPSEEK_ENDPOINT"],
                        fireworks_api_key=os.environ["DEEPSEEK_API_KEY"],
                    ).astream(messages):
                        # 해당 chunk는 graph에서 stream되는 구조와 다르기 때문에, 임의로 event schema를 맞춰주도록 구현함.
                        event = {
                            "event": "on_chat_model_stream",
                            "data": {"chunk": chunk},
                            "tags": ["answer"],
                        }
                        yield event

                        if response:
                            response += chunk
                        else:
                            response = chunk

                    # Update the last AI message in state with the continued response
                    # Find the last AI message and replace it with the continued response
                    updated_messages = messages.copy()
                    for i in reversed(range(len(updated_messages))):
                        if isinstance(updated_messages[i], (AIMessage, AIMessageChunk)):
                            # Replace the last AI message with the continued response
                            updated_messages[i] = AIMessage(
                                content=updated_messages[i].content + response
                            )
                            break
                    else:
                        # If no AI message found, append the response
                        updated_messages.append(AIMessage(content=response))

                    # Update state with the continued response
                    updated_state = current_state_values.copy()
                    updated_state["messages"] = updated_messages

                    await self.graph.aupdate_state(
                        config=self.config,
                        values=updated_state,
                    )

                    logger.debug("DeepSeek-R1 continue completed")
                else:
                    logger.debug("Using standard continue mode")

                    # Stream with continue prompt
                    async for chunk in self.graph.astream_events(
                        input={"messages": [SystemMessage(content=CONTINUE_PROMPT)]},
                        config=self.config,
                        stream_mode=stream_mode,
                        version=version,
                    ):
                        yield chunk

                    # Remove the CONTINUE_PROMPT system message from state after streaming
                    current_state = self.graph.get_state(config=self.config)
                    current_values = current_state.values
                    messages = current_values.get("messages", [])

                    # Find and remove the CONTINUE_PROMPT system message
                    # It should be the second-to-last message (before the AI response)
                    updated_messages = []
                    for msg in messages:
                        if not (
                            isinstance(msg, SystemMessage)
                            and msg.content == CONTINUE_PROMPT
                        ):
                            updated_messages.append(msg)

                    updated_values = current_values.copy()
                    updated_values["messages"] = [
                        RemoveMessage(id=REMOVE_ALL_MESSAGES)
                    ] + updated_messages

                    await self.graph.aupdate_state(
                        config=self.config,
                        values=updated_values,
                    )
                    logger.debug("Removed CONTINUE_PROMPT from state after streaming")

            except Exception as e:
                logger.error(f"Error in astream_continue_events: {str(e)}")
                raise

        return _continue_generator()

    async def arestream_events(
        self, user_input: str = None, stream_mode: str = "values", version: str = "v2"
    ):
        logger.info("Starting restream events")

        try:
            # Get current state
            current_state = self.graph.get_state(config=self.config).values
            messages = current_state.get("messages", [])

            if not messages:
                raise ValueError("No messages found in current state for restream")

            # Find the last HumanMessage using MessageMixin method
            last_human_message = self.get_last_message(messages, HumanMessage)

            if not last_human_message:
                raise ValueError("No Human message found for restream")

            # Find the index of the last Human message
            last_human_index = -1
            for i in reversed(range(len(messages))):
                if messages[i] is last_human_message:
                    last_human_index = i
                    break

            # Keep all messages before the last Human message
            updated_messages = messages[:last_human_index]

            logger.debug(f"Found last Human message at index {last_human_index}")
            logger.debug(
                f"Keeping {len(updated_messages)} messages before last Human message"
            )

            # Determine the message to use for restreaming
            if user_input:
                message = HumanMessage(content=user_input)
                logger.debug(f"Using provided user input: {user_input[:100]}...")
            else:
                message = last_human_message
                logger.debug("Using last Human message for restream")

            # Update the graph state with messages excluding everything after last Human message
            reset_state = current_state.copy()
            reset_state["messages"] = [
                RemoveMessage(id=REMOVE_ALL_MESSAGES)
            ] + updated_messages

            await self.graph.aupdate_state(
                config=self.config,
                values=reset_state,
            )

            # Stream events from the reverted state
            return self.graph.astream_events(
                input={"messages": [message]},
                config=self.config,
                stream_mode=stream_mode,
                version=version,
            )

        except Exception as e:
            logger.error(f"Error in arestream_events: {str(e)}")
            raise

    def dump(self):
        logger.debug("Dumping agent state")
        try:
            state = self.graph.get_state(config=self.config).values
            state["messages"] = messages_to_dict(state.get("messages", []))
            logger.debug("Agent state dumped successfully")
            return state
        except Exception as e:
            logger.error(f"Error dumping agent state: {str(e)}")
            raise

    def convert(self, dumped_data: dict):
        """
        앨런 v2.0.0 미만 버전의 데이터를 처리하기 위한 함수.

        Schema for `dumped_state`:

        dumped_data: dict
            A dictionary containing the current state of the system. It has the following keys:

            - memory: dict
                Represents the system's memory data. It contains:

                - summaries: list[str]
                    Stores summarized information related to various contexts.

                - messages: list[dict]
                    Contains historical messages or communication logs.

            - references: list[dict]
                A structure that holds references or metadata related to the current state.
        """
        logger.info("Converting legacy data format")

        try:
            state = AlanState()

            # Convert messages if they exist
            messages = dumped_data.get("memory", {}).get("messages", [])
            converted_messages = []

            for i, msg in enumerate(messages):
                # Skip None messages
                if msg is None:
                    continue

                # Check if this is a function message
                if (
                    msg.get("type") == "ai"
                    and msg.get("data", {}).get("additional_kwargs", {}).get("type")
                    == "function"
                ):

                    # Get the next message which should be the response
                    next_msg = messages[i + 1] if i < len(messages) - 1 else None

                    # Only add tool_calls if we have a valid tool response
                    if (
                        next_msg is not None
                        and next_msg.get("type") == "system"
                        and next_msg.get("data", {})
                        .get("additional_kwargs", {})
                        .get("type")
                        == "tool_result"
                    ):

                        try:
                            # Extract command data
                            content = msg["data"]["content"]
                            json_str = (
                                content.strip()
                                .replace("```json\n", "")
                                .replace("```", "")
                            )
                            content_data = json.loads(json_str)
                            command = content_data.get("command", {})

                            # Generate unique ID
                            tool_id = f"call_{int(time.time() * 1000)}"

                            # Create tool call format
                            tool_call = {
                                "name": command.get(
                                    "name",
                                    msg["data"]["additional_kwargs"].get("tool_name"),
                                ),
                                "args": command.get("args", {}),
                                "id": tool_id,
                                "type": "function",
                            }

                            # Create function call format for additional_kwargs
                            function_call = {
                                "index": 0,
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tool_call["name"],
                                    "arguments": json.dumps(command.get("args", {})),
                                },
                            }

                            # Update the AI message format with tool_calls
                            converted_ai_msg = {
                                "type": "ai",
                                "data": {
                                    "content": "",
                                    "additional_kwargs": {
                                        "tool_calls": [function_call]
                                    },
                                    "response_metadata": {},
                                    "type": "ai",
                                    "name": None,
                                    "id": None,
                                    "example": False,
                                    "tool_calls": [tool_call],
                                    "invalid_tool_calls": [],
                                    "usage_metadata": None,
                                },
                            }

                            # Add the AI message with tool calls
                            converted_messages.append(converted_ai_msg)

                            # If we have a valid tool response, add it as a system message
                            if (
                                next_msg is not None
                                and next_msg.get("type") == "system"
                                and next_msg.get("data", {})
                                .get("additional_kwargs", {})
                                .get("type")
                                == "tool_result"
                            ):

                                # Create system message with tool result
                                system_msg = {
                                    "type": "tool",
                                    "data": {
                                        "content": next_msg["data"]["content"],
                                        "additional_kwargs": {},
                                        "response_metadata": {},
                                        "type": "tool",
                                        "name": tool_call["name"],
                                        "id": None,
                                        "tool_call_id": tool_id,
                                        "artifact": None,
                                        "status": "success",
                                    },
                                }

                                converted_messages.append(system_msg)
                                messages[i + 1] = None  # Mark as processed

                                # Get the final response message if it exists
                                final_response = (
                                    messages[i + 2] if i + 2 < len(messages) else None
                                )
                                if (
                                    final_response is not None
                                    and final_response.get("type") == "AIMessageChunk"
                                ):
                                    converted_messages.append(final_response)
                                    messages[i + 2] = None  # Mark as processed

                        except (json.JSONDecodeError, KeyError) as e:
                            logger.error(
                                f"Error processing function message at index {i}: {str(e)}"
                            )
                            raise e
                    else:
                        converted_messages.append(msg)
                else:
                    converted_messages.append(msg)

            state.messages = converted_messages
            state.references = dumped_data.get("references", [])

            logger.info("Legacy data conversion completed successfully")
            return state.model_dump()

        except Exception as e:
            logger.error(f"Error converting legacy data: {str(e)}")
            raise

    def format_reference(
        self, answer: str, reference_format_string: str, max_references: int = 2
    ) -> str:
        """
        answer: completed tokens
        reference_format_string: reference replace format string
            example: "[{number}]({link})"
        """
        logger.debug(f"Formatting references for answer")

        metadata = []
        state = self.graph.get_state(config=self.config).values

        # comma로 연결된 여러 레퍼런스를 [^1][^2] 형태로 변경
        def _expand_multi_refs(match: re.Match) -> str:
            nums = re.findall(r"\d+", match.group())
            return "".join(f"[^{n}]" for n in nums)

        multi_pattern = re.compile(r"\[?\^?\d+\^?(?:\s*,\s*\^?\d+\^?)+\]?")
        answer = multi_pattern.sub(_expand_multi_refs, answer)

        # 2) 연속된 개별 마커: "[^1^][^2][^3^][^4]" → 앞의 2개만 남기기 (두 형태 모두 처리)
        # TODO: 올바른 표기인 [^%d]로 응답하도록 프롬프트를 변경했으나, 이전 대화들은 [^%d^]꼴도 있을 것이므로 두 형태 모두 대응가능하도록 구현함.
        def _trim_sequence(match: re.Match) -> str:
            markers = re.findall(r"\[\^\d+\^?\]", match.group())
            return "".join(markers[:max_references])

        seq_pattern = re.compile(r"(?:\[\^\d+\^?\]\s*){%d,}" % (max_references + 1))
        answer = seq_pattern.sub(_trim_sequence, answer)

        # 두 형태 모두 매칭: [^1^] 또는 [^1]
        pattern = r"\[\^(\d+)\^?\]"
        if references := re.findall(pattern, answer):
            references = {
                i: ref
                for i in sorted(set(map(int, references)))
                for ref in state.get("references", [])
                if ref.get("number") == i and ref.get("source") is not None
            }

            # 치환 시에도 두 형태 모두 처리
            for key, value in references.items():
                answer = re.sub(
                    rf"\[\^{key}\^?\]",
                    reference_format_string.format(number=key, link=value["source"]),
                    answer,
                )

            # 남은 모든 [^n^] 또는 [^n] 제거
            answer = re.sub(pattern, "", answer)
            metadata = list(references.values())

        logger.debug("Formatted references successfully")
        return answer, metadata

    async def asuggest(self, message_type: BaseMessage = AIMessage) -> list[str] | None:
        logger.debug(
            f"Generating suggestions for message type: {message_type.__name__}"
        )

        try:

            class SuggestScheme(BaseModel):
                suggested_questions: list[str] = Field(
                    description="Four new questions based on the answers. Those questions must be in Korean.",
                )

            messages = self.graph.get_state(config=self.config).values["messages"]
            if message_type == AIMessage:
                answer = self.get_last_message(messages, AIMessage)
            elif (
                message_type == SystemMessage
            ):  # For suggesting questions based on a YouTube summary.
                answer = self.get_first_message(messages, SystemMessage)

            previous_questions = [
                question.content
                for question in self.get_last_k_messages(messages, HumanMessage, k=5)
            ]

            if answer is None:
                logger.warning("No answer found for suggestion generation")
                return []

            # TODO: 해당 부분 Structured Output으로 변경
            parser = PydanticOutputParser(pydantic_object=SuggestScheme)
            prompt = (
                SuggestPrompt()
                .get_prompt_template()
                .partial(format_instructions=parser.get_format_instructions())
            )
            chain = prompt | suggest_llm | parser
            result = await chain.ainvoke(
                {"answer": answer.content, "previous_questions": previous_questions}
            )

            logger.debug(f"Generated {len(result.suggested_questions)} suggestions")
            return result.suggested_questions

        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return []

    @property
    def stop_reason(self):
        messages = self.graph.get_state(config=self.config).values["messages"]

        # claude의 경우, 정상 답변 완료 시, finish_reason 없음.
        finish_reason = messages[-1].response_metadata.get("finish_reason") or messages[
            -1
        ].response_metadata.get("stop_reason")

        if finish_reason in STOP_REASON_MAPPING:
            finish_reason = STOP_REASON_MAPPING[finish_reason]

        if isinstance(finish_reason, str):
            return finish_reason.lower()

        return finish_reason


class VanillaChat(AlanAgent):
    @classmethod
    def create(
        cls,
        llm: BaseLanguageModel | RunnableBinding,
        llm_type: str,
        init_data: dict[str, Any] | None = None,
    ):
        logger.info(f"Creating Vanilla Chat with LLM type: {llm_type}")

        if init_data:
            logger.debug("Agent initialized with existing data")

        try:
            # Build Graph
            graph = cls._create_graph(
                llm=llm,
                llm_type=llm_type,
                init_data=init_data,
                config=default_config,
            )

            agent = VanillaChat(
                llm=llm,
                graph=graph,
            )

            logger.info("VanillaChat created successfully")
            return agent

        except Exception as e:
            logger.error(f"Failed to create AlanAgent: {str(e)}")
            raise

    @staticmethod
    def _create_graph(
        llm: BaseLanguageModel | RunnableBinding,
        llm_type: str,
        init_data: dict[str, Any] | None,
        config: dict,
    ) -> CompiledStateGraph:
        logger.debug("Creating agent graph")

        graph_builder = StateGraph(AlanState)

        nodes = [
            (
                "query_analysis",
                QueryAnalysis(
                    llm=llm,
                    prompt=VanillaChatPrompt(),
                    tools=[],
                ),
            ),
        ]
        edges = [
            (START, "query_analysis"),
        ]
        edges_with_conditions = []

        graph_builder = add_graph_components(
            StateGraph(AlanState), nodes, edges, edges_with_conditions
        )

        graph = graph_builder.compile(checkpointer=MemorySaver())

        if init_data:
            init_data["image_info"] = []
            init_data["video_info"] = []
            logger.debug("Reset image and video info in init_data")

            init_data["messages"] = messages_from_dict(init_data["messages"])
            logger.debug("Updating graph state with init_data")

            graph.update_state(config=config, values=init_data)

        logger.debug("Graph created successfully")
        return graph
