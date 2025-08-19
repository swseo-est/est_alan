import asyncio
import json
import os
import random
import re
from collections import defaultdict
from typing import Annotated, Any, Callable, Literal, Optional, Sequence, Union

from dotenv import load_dotenv
from langchain.schema import BaseMessage, SystemMessage, get_buffer_string
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    ToolMessage,
)
from langchain_core.runnables import RunnableBinding, RunnableConfig, RunnableSequence
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Command
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel, Field

from alan.core.prompt import BasePrompt, ContentFilteringPrompt
from alan.deepsearch.prompt import GuardrailPrompt
from alan.logging_config import get_logger
from alan.model_config import get_max_context_size_from_llm, supports_tool_calling
from alan.tools.base import AsyncTool

load_dotenv()
logger = get_logger(__name__)

Messages = Union[list[MessageLikeRepresentation], MessageLikeRepresentation]
TOOL_PRIORITY = [
    "search_web",
    "search_news",
    "search_weather",
    "summarize_url",
    "summarize_references",
]

from vertexai.preview.tokenization import get_tokenizer_for_model

tokenizer = get_tokenizer_for_model("gemini-1.5-flash-002")


def get_num_tokens_from_messages(messages: list[BaseMessage]):
    return tokenizer.count_tokens(
        [message.content for message in messages]
    ).total_tokens


async def echo(text: str):
    return AIMessage(content=text)


class AlanStateV1(BaseModel):
    version: str = "v1"
    messages: Annotated[Messages, add_messages] = Field(default_factory=list)
    references: list[dict] = Field(default_factory=list)
    image_info: list[dict] = Field(default_factory=list)
    video_info: list[dict] = Field(default_factory=list)
    summaries: list[str] = Field(default_factory=list)


class AlanStateV2(BaseModel):
    version: str = "v2"
    messages: Annotated[Messages, add_messages] = Field(default_factory=list)
    references: list[dict] = Field(default_factory=list)
    image_info: list[dict] = Field(default_factory=list)
    video_info: list[dict] = Field(default_factory=list)


# 버전별 스키마 매핑
STATE_SCHEMAS = {"v1": AlanStateV1, "v2": AlanStateV2}

# 현재 버전
CURRENT_VERSION = "v2"
AlanState = STATE_SCHEMAS[CURRENT_VERSION]


class AsyncRunnableCallable(RunnableCallable):
    def _func(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError


tool_call_llm = AzureChatOpenAI(
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


class QueryAnalysis(AsyncRunnableCallable):
    def __init__(
        self,
        llm: BaseLanguageModel | RunnableBinding,
        prompt: BasePrompt,
        tools: list[AsyncTool],
        *,
        guardrail_llm: BaseLanguageModel | RunnableBinding = None,
        name: str = "query_analysis",
        tags: Optional[list[str]] = None,
        answer_llm_tags: list[str] = ["answer"],
        max_tool_calls: int = 2,
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)

        logger.debug(
            f"Initializing QueryAnalysis with tools: {[tool.name for tool in tools]}"
        )

        self.prompt = prompt

        llm = llm.model_copy()
        llm.disable_streaming = False
        self.llm = llm.with_config(tags=answer_llm_tags)
        self.guardrail_llm = (
            guardrail_llm.with_config(tags=["guardrail"]) if guardrail_llm else None
        )
        self.is_content_safe = False
        self.tools = tools

        self.tool_call_enabled = supports_tool_calling(
            llm
        )  # TODO: MODEL_CONFIG에 있는 Tool calling 키를 사용하지 못하는 이유는 llm_type을 QueryAnalysis에서 받지 않기 때문. 생각해보기.
        self.max_tool_calls = max_tool_calls
        self.tool_call_count = 0

        logger.debug(f"Tool calling enabled: {self.tool_call_enabled}")

        if self.tool_call_enabled:
            self.llm_with_tools = llm.bind_tools(tools).with_config(
                tags=answer_llm_tags
            )
        else:
            self.llm_with_tools = tool_call_llm.bind_tools(
                tools
            )  # 해당 llm의 답변은 stream되어서는 안되므로 tag를 제거함.

    async def _afunc(self, state: AlanState):
        logger.debug(f"Starting {self.name} node")

        if await self._is_context_exceeded(state.messages):
            logger.warning("Context exceeded, ending conversation.")

            return Command(
                goto="__end__",
                update={
                    "messages": [
                        AIMessage(
                            content="모델의 Context를 초과하여 답변할 수 없습니다. 새로운 대화를 시작해주세요.",
                            response_metadata={"finish_reason": "out_of_context"},
                        )
                    ],
                },
            )

        if isinstance(state.messages[-1], HumanMessage):
            await adispatch_custom_event(
                "event", {"speak": "질문의 의도를 이해하고 있어요."}
            )

        # TODO: 앨런 V3인 경우, V3모델로 구동되어야 한다고 말해야 하기 때문에, 아래와 같이 조건문을 설정하였음.
        # TODO: Vanilla일때 잘 들어가는지 Check.
        default_kwargs = {
            "ai_codename": "Alan",
            "ai_nickname": "Alan (앨런)",
            "ai_role": "Assistant",
            "ai_modeltype": (
                "Alan v3"
                if hasattr(self.prompt, "llm_type")
                and any(
                    provider in self.prompt.llm_type for provider in ["azure", "gemini"]
                )
                else (self.prompt.llm_type if hasattr(self.prompt, "llm_type") else "")
            ),
            "ai_abilities": "\n".join([f"- {tool.description}" for tool in self.tools]),
        }

        inputs = await self._aprep_inputs(messages=state.messages, **default_kwargs)

        llm = (
            self.llm
            if self.tool_call_count >= self.max_tool_calls
            else self.llm_with_tools
        )
        logger.debug("Invoking LLM for response generation")

        task = asyncio.create_task(llm.ainvoke(inputs))

        async def _check_guardrail():
            guard_prompt = GuardrailPrompt()
            try:
                response = await self.guardrail_llm.ainvoke(
                    guard_prompt.format_messages(query=state.messages[-1])
                )
            except Exception:
                response = AIMessage(content="")
            if sum(tool in response.content for tool in guard_prompt.tool_info) > 1:
                return True
            return False

        self.is_content_safe = True
        response = await task
        self.tool_call_count += bool(response.tool_calls)

        if (
            not self.tool_call_enabled
            and isinstance(llm.bound, AzureChatOpenAI)
            and not response.tool_calls
        ):
            logger.debug("Using fallback LLM for non-tool response")
            response = await self.llm.ainvoke(inputs)

        response = await self._merge_tool_calls(response)

        def convert_tool_calls_to_text(message: AIMessage):
            text_parts = []

            for tool_call in message.tool_calls:
                function = tool_call.get("function", {})
                name = function.get("name")
                args = function.get("arguments")

                if name and args:
                    text_parts.append(
                        f"Function '{name}' was called with arguments: {args}"
                    )

            return AIMessage(content="\n".join(text_parts))

        logger.debug("Query analysis completed")
        return {
            "messages": [response],
            "version": CURRENT_VERSION,
        }

    async def _is_context_exceeded(self, messages: list[BaseMessage]) -> bool:
        llm_to_use = self.llm_with_tools if self.tool_call_enabled else self.llm

        total_tokens = get_num_tokens_from_messages(messages)
        max_context_size = get_max_context_size_from_llm(llm_to_use)

        return total_tokens > max_context_size

    async def _merge_tool_calls(self, response: AIMessage):
        logger.debug("Merging tool calls")
        response = response.model_copy()

        try:
            if not response.tool_calls:
                return response

            # Tool name별로 그룹화
            calls_by_tool = defaultdict(list)
            for call in response.tool_calls:
                calls_by_tool[call["name"]].append(call)

            merged_calls = []

            for tool_name, calls in calls_by_tool.items():
                if len(calls) == 1:
                    # 병합할 필요 없는 경우
                    merged_calls.append(calls[0])
                    continue

                # 첫 번째 call을 기준으로 병합
                base_call = calls[0].copy()
                base_args = base_call["args"].copy()

                # 나머지 calls의 args를 병합
                for call in calls[1:]:
                    for key, value in call["args"].items():
                        if isinstance(value, list):
                            # 리스트의 경우: 기존 값과 합치고 중복 제거 후 정렬
                            if key in base_args and isinstance(base_args[key], list):
                                base_args[key].extend(value)
                                base_args[key] = sorted(set(base_args[key]))
                            else:
                                base_args[key] = sorted(set(value))
                        else:
                            # 리스트가 아닌 경우: 덮어쓰기
                            base_args[key] = value

                base_call["args"] = base_args
                merged_calls.append(base_call)

            response.tool_calls = merged_calls

            logger.debug(
                f"Merged {len(response.tool_calls)} tool calls from {len([call for calls in calls_by_tool.values() for call in calls])} original calls"
            )

            return response

        except Exception as e:
            logger.error(f"Error merging tool calls: {str(e)}")
            return response

    async def _aprep_inputs(
        self, messages: list[BaseMessage], **kwargs: dict[str, Any]
    ):
        logger.debug("Preparing inputs for LLM")

        # Trim messages based on max tokens
        # trimmed_messages = await self._atrim_messages(
        #     messages=messages,
        #     token_limit=get_max_context_size_from_llm(self.llm_with_tools),
        #     token_counter=self.llm_with_tools.get_num_tokens_from_messages,
        # )

        # Prepare messages
        formatted_prompt = self.prompt.format_messages(messages=messages, **kwargs)

        if not self.tool_call_enabled:
            result = []
            for idx, message in enumerate(formatted_prompt):
                if not message.content and message.tool_calls:
                    continue
                if isinstance(message, ToolMessage):
                    formatted_prompt[idx] = HumanMessage(content=message.content)

                result.append(formatted_prompt[idx])
            formatted_prompt = result

        logger.debug(f"Prepared {len(formatted_prompt)} messages for LLM")
        return formatted_prompt

    # TODO: summaries를 빼면서, langchain 기본 제공 trim_messages를 쓰는게 나을듯.
    # TODO: 아니지... 그냥 tirm_message를 쓸 필요가 없겠구나! Context를 넘으면 다음 대화로 넘어가게 하도록 할것이므로.
    async def _atrim_messages(
        self,
        messages: list[BaseMessage],
        token_limit: int,
        token_counter: Callable[[str], int],
    ):
        logger.debug(f"Trimming messages to fit token limit: {token_limit}")

        token_used = 0
        trim_index = 0
        for i in reversed(range(len(messages))):
            message_token_size = token_counter([messages[i]])

            if token_used + message_token_size > token_limit:
                trim_index = i + 1
                break

            token_used += message_token_size

        if trim_index != 0:
            logger.debug(f"Trimmed {trim_index} messages")
            return messages[trim_index:]
        else:
            logger.debug("No trimming needed")
            return messages[trim_index:]


class ContentFilterResult(BaseModel):
    """Content filtering result with relevant search result indices"""

    filtered: list[int] = Field(
        description="List of indices for search results that are relevant to the query"
    )


class ToolCalling(ToolNode):
    """
    2024-11-21
    - default로 제공하는 ToolNode는 후처리 작업 없이 messages에 Tool Calling 결과물을 추가하므로, 후처리를 하여 저장하도록 직접 구현함.
    """

    filter_llm: BaseLanguageModel | RunnableBinding
    blocked_url_pattern: re.Pattern = re.compile(r"youtube\.com|youtu\.be|tiktok\.com")

    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable]],
        filter_llm: BaseLanguageModel,
        *,
        name: str = "tools",
        tags: Optional[list[str]] = None,
        handle_tool_errors: Union[
            bool, str, Callable[..., str], tuple[type[Exception], ...]
        ] = True,
        messages_key: str = "messages",
    ) -> None:
        super().__init__(
            tools,
            name=name,
            tags=tags,
            handle_tool_errors=handle_tool_errors,
            messages_key=messages_key,
        )
        self.filter_llm = filter_llm.with_structured_output(
            ContentFilterResult  # , method="function_calling"
        )
        logger.debug(
            f"ToolCalling initialized with tools: {[tool.name for tool in tools]}"
        )

    async def _afunc(
        self,
        input: AlanState,
        config: RunnableConfig,
        *,
        store: BaseStore,
    ) -> Any:
        logger.debug(f"Starting {self.name} node")

        tool_calls, input_type = self._parse_input(input, store)
        tool_calls = self.normalize_tool_calls(tool_calls)

        logger.debug(f"Parsed {len(tool_calls)} tool calls")

        highest_priority_tool = next(
            (
                tool_name
                for tool_name in TOOL_PRIORITY
                if any(call["name"] == tool_name for call in tool_calls)
            ),
            None,
        )
        speak_tool_idx = next(
            (
                i
                for i, call in enumerate(tool_calls)
                if call["name"] == highest_priority_tool
            ),
            None,
        )

        for i, call in enumerate(tool_calls):
            call["args"]["verbose"] = speak_tool_idx == i

        await adispatch_custom_event(
            "event", {"used_tools": [call["name"] for call in tool_calls]}
        )

        logger.debug("Executing tool calls")
        outputs = await asyncio.gather(
            *(self._arun_one(call, input_type, config) for call in tool_calls)
        )
        outputs = await self.postprocess_tool_results(input, outputs)

        logger.debug("Tool calling completed")
        return outputs

    # tool_calls의 schema를 LLM이 지키지 못한 경우에 대해 후처리 하는 함수.
    # TODO: 일단은 search_web, search_news에 대해서만
    def normalize_tool_calls(self, tool_calls):
        normalized_calls = []

        for tool_call in tool_calls:
            normalized_call = tool_call.copy()

            if (
                tool_call.get("name") in ["search_web", "search_news"]
                and "args" in tool_call
                and "query" in tool_call["args"]
            ):

                normalized_call["args"] = tool_call["args"].copy()
                query = tool_call["args"]["query"]

                if isinstance(query, str):
                    normalized_call["args"]["query"] = [query]

            normalized_calls.append(normalized_call)

        return normalized_calls

    async def postprocess_tool_results(
        self, state: AlanState, tool_call_results: list[ToolMessage]
    ):
        logger.debug(f"Post-processing {len(tool_call_results)} tool results")

        image_info = state.image_info
        video_info = state.video_info

        for tool_message in tool_call_results:
            name = tool_message.name
            observation = tool_message.content

            if not isinstance(observation, list):
                try:
                    observation = json.loads(observation)
                except Exception as e:
                    logger.error(f"Error parsing tool result: {str(e)}")
                    return {
                        "messages": ToolMessage(
                            content="An error occurred while post-processing the tool result.",
                            tool_call_id=tool_message.tool_call_id,
                        )
                    }

            text_observation, image_observation, video_observation = (
                self._split_results_by_type(observation)
            )

            logger.debug("Filtering content relevance")
            image_observation = await self._filter_unrelative_contents(
                state.messages[-2], image_observation
            )

            video_observation = await self._filter_unrelative_contents(
                state.messages[-2], video_observation
            )

            formatted_tool_message, references = self._format_observation(
                text_observation, state.references, tool_name=name
            )
            image_info = self._format_image_observation(image_observation)
            video_info = self._format_video_observation(video_observation)

            tool_message.content = formatted_tool_message
            state.references = references

        logger.debug("Tool result post-processing completed")
        return {
            "messages": tool_call_results,
            "references": state.references,
            "image_info": image_info if image_info else state.image_info,
            "video_info": video_info if video_info else state.video_info,
        }

    def _split_results_by_type(self, observation: list[dict]):
        text_results = []
        image_results = []
        video_results = []

        for item in observation:
            if item["metadata"].get("type") == "image":
                image_results.append(item)
            elif item["metadata"].get("type") == "video":
                video_results.append(item)
            else:
                text_results.append(item)

        return text_results, image_results, video_results

    def _format_observation(
        self, observation: list[dict], references: list[dict], tool_name: str
    ):
        logger.debug(f"Formatting observation for tool: {tool_name}")
        new_observation: list[dict] = []

        for it in observation:
            obj = {}
            reference = {}
            number = len(references)

            if "source_no" in it["metadata"]:
                source_no = it["metadata"]["source_no"]
                obj["number"] = source_no

                # 해당 번호의 참조를 찾아 내용 업데이트
                for ref in references:
                    if ref.get("number") == source_no:
                        ref["content"] = it["page_content"]
                        break
            elif "source" in it["metadata"]:
                if (
                    self.blocked_url_pattern.search(it["metadata"].get("source", ""))
                    and it["metadata"].get("type", "text") != "youtube_summary"
                ):
                    logger.debug(
                        f"Skipping blocked URL: {it['metadata'].get('source', '')}"
                    )
                    continue

                obj["number"] = number + 1
                reference["number"] = number + 1

                # reference key에 대한 처리.
                reference_key = [
                    "source",
                    "title",
                    "thumbnail",
                    "date",
                    "attributes",
                    "kind",
                ]
                reference.update(
                    {
                        key: it["metadata"][key]
                        for key in reference_key
                        if key in it["metadata"]
                    }
                )
                reference["content"] = it["page_content"]
                reference["tool_name"] = tool_name

                if tool_name != "search_weather":
                    references.append(reference)

            # 페이지 콘텐츠의 타입에 따라 JSON 파싱 또는 그대로 저장.
            if it["metadata"].get("type", "text") == "json":
                obj["content"] = json.loads(it["page_content"].replace(r"\r", ""))
            else:
                obj["content"] = it["page_content"]

            other_keys = [
                key
                for key in it["metadata"]
                if key not in ["content", "source", "source_no", "thumbnail"]
            ]

            for key in other_keys:
                obj[key] = it["metadata"][key]

            new_observation.append(obj)

        formatted_tool_message = json.dumps(
            new_observation, indent=2, ensure_ascii=False
        )

        logger.debug(f"Formatted {len(new_observation)} observations")
        return formatted_tool_message, references

    def _format_image_observation(self, observation: list[dict]):
        logger.debug(f"Formatting {len(observation)} image observations")
        new_observation: list[dict] = []
        for idx, it in enumerate(observation):
            if self.blocked_url_pattern.search(it["metadata"]["link"]):
                logger.debug(f"Skipping blocked image URL: {it['metadata']['link']}")
                continue

            obj = {
                key: it["metadata"][key]
                for key in [
                    "title",
                    "link",
                    "image_url",
                    "imageWidth",
                    "imageHeight",
                    "thumbnail_url",
                    "thumbnailWidth",
                    "thumbnailHeight",
                    "source",
                ]
                if key in it["metadata"]
            }

            if obj.get("imageHeight", 0) >= 400 and obj.get("imageWidth", 0) >= 400:
                new_observation.append(obj)

        # TODO: 2025-04-10, 동형님과 논의. number 임시로 넣어놓고 front단에서 사용하지 않으면 다시 제거하는 걸로.
        for idx, item in enumerate(new_observation, start=1):
            item["number"] = idx

        result = new_observation[:10] if len(new_observation) > 10 else new_observation
        logger.debug(f"Formatted {len(result)} valid image observations")
        return result

    def _format_video_observation(self, observation: list[dict]):
        logger.debug(f"Formatting {len(observation)} video observations")
        for idx, item in enumerate(observation, start=1):
            item["number"] = idx

        result = observation[:10] if len(observation) > 10 else observation
        logger.debug(f"Formatted {len(result)} video observations")
        return result

    async def _filter_unrelative_contents(
        self, user_query: HumanMessage, observation: list[dict]
    ):
        if not observation:
            return observation

        logger.debug(f"Filtering {len(observation)} contents for relevance")
        try:
            for idx, content in enumerate(observation):
                content["number"] = idx

            result = await self.filter_llm.ainvoke(
                ContentFilteringPrompt().format_messages(
                    user_query=user_query.content, observation=observation
                )
            )
            new_observation = [observation[i] for i in result.filtered]

            logger.debug(f"Filtered to {len(new_observation)} relevant contents")
            return new_observation
        except Exception as e:
            logger.error(f"Error filtering content relevance: {str(e)}")
            await adispatch_custom_event("error", {"error": e})
            return observation


async def route_tools(state: AlanState) -> Literal["tool_calling", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    ai_message = state.messages[-1] if state.messages else None

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        logger.debug(
            f"Routing to tool_calling with {len(ai_message.tool_calls)} tool calls"
        )
        return "tool_calling"

    logger.debug("Routing to end")
    return "__end__"
