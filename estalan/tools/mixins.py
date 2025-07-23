import os
import uuid
from typing import Any, Optional

import httpx
from bs4 import BeautifulSoup
from httpx._decoders import TextDecoder
from langchain.schema import Document
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableBinding

from estalan.logging_config import get_logger

logger = get_logger(__name__)


def _clean_url(url: str) -> str:
    cleaned = url.strip("\"'")
    logger.debug(f"Cleaned URL: {url} -> {cleaned}")
    return cleaned


def _parse_content_type_header(header: str) -> tuple[str, dict]:
    logger.debug(f"Parsing content type header: {header}")
    tokens = header.split(";")
    content_type, params = tokens[0].strip(), tokens[1:]
    params_dict = {}
    items_to_strip = "\"' "

    for param in params:
        param = param.strip()
        if param:
            key, value = param, True
            index_of_equals = param.find("=")
            if index_of_equals != -1:
                key = param[:index_of_equals].strip(items_to_strip)
                value = param[index_of_equals + 1 :].strip(items_to_strip)
            params_dict[key.lower()] = value

    logger.debug(f"Parsed content type: {content_type}, params: {params_dict}")
    return content_type, params_dict


class ContentTypeError(Exception):
    pass


class MessageMixin:
    def get_first_message(
        self, messages: list, message_type: type[BaseMessage]
    ) -> Optional[BaseMessage]:
        for message in messages:
            if isinstance(message, message_type):
                return message
        return None

    def get_last_message(
        self, messages: list, message_type: type[BaseMessage]
    ) -> Optional[BaseMessage]:
        for message in reversed(messages):
            if isinstance(message, message_type):
                return message
        return None

    def get_last_k_messages(
        self, messages: list, message_type: type[BaseMessage], k: int = 1
    ) -> list[BaseMessage]:
        return [
            message
            for message in reversed(messages)
            if isinstance(message, message_type)
        ][:k]


class NoSyncChainMixin:
    def _call(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError


class HTTPXMixin:
    async def aget(
        self, url: str, headers: dict = dict(), content_type: str = None, **kwargs
    ) -> str:
        logger.debug(f"Making HTTP GET request to: {url}")

        try:
            async with httpx.AsyncClient(
                timeout=10,
                follow_redirects=True,
                headers=headers,
            ) as client:
                async with client.stream("GET", _clean_url(url), **kwargs) as stream:
                    stream.raise_for_status()
                    logger.debug(f"HTTP request successful: {stream.status_code}")

                    if (
                        content_type is not None
                        and content_type not in stream.headers["Content-Type"]
                    ):
                        logger.warning(
                            f"Content type mismatch. Expected: {content_type}, Got: {stream.headers['Content-Type']}"
                        )
                        raise ContentTypeError

                    content = await anext(stream.aiter_bytes(2 * 1024 * 1024))
                    content_type_header, params = _parse_content_type_header(
                        stream.headers["content-type"]
                    )

                    if "charset" in params:
                        encoding = params["charset"].strip("'\"")
                    else:
                        encoding = chardet.detect(content).get("encoding") or "utf-8"

                    logger.debug(f"Using encoding: {encoding}")
                    decoder = TextDecoder(encoding=encoding)
                    result = "".join([decoder.decode(content), decoder.flush()])

                    logger.info(
                        f"Successfully fetched content from {url} ({len(result)} characters)"
                    )
                    return result

        except Exception as e:
            logger.error(f"HTTP request failed for {url}: {str(e)}")
            raise


class HTMLToMarkdownMixin:
    async def aclean_html(
        self,
        html_text: Document,
    ) -> Document:
        logger.debug(
            f"Cleaning HTML document from {html_text.metadata.get('source', 'unknown')}"
        )

        try:
            metadata = html_text.metadata
            md_text = self._transform_html_readability_markdown(html_text.page_content)

            if not md_text:
                logger.warning("No content found in HTMLToMarkdownMixin")
                metadata.update({"error": "No content found in HTMLToMarkdownMixin."})
                return Document(
                    page_content="콘텐츠가 존재하지 않습니다.",
                    metadata=metadata,
                )

            logger.debug(f"HTML cleaning successful, markdown length: {len(md_text)}")
            return Document(page_content=md_text, metadata=metadata)

        except Exception as e:
            logger.error(f"Error cleaning HTML: {str(e)}")
            raise

    def _strip_stylesheets(self, html: str) -> str:
        logger.debug("Stripping stylesheets from HTML")
        try:
            soup = BeautifulSoup(html, "lxml")

            for node in soup.select("style, link[rel=stylesheet]"):
                node.decompose()

            result = str(soup)
            logger.debug("Stylesheets stripped successfully")
            return result

        except Exception as e:
            logger.error(f"Error stripping stylesheets: {str(e)}")
            raise

    def _transform_html_readability_markdown(self, html_text: str) -> str:
        logger.debug("Transforming HTML to markdown using readability")

        try:
            if have_node():
                logger.debug("Using readability with Node.js")
                sanitized_html = self._strip_stylesheets(html_text)
                readability_content = simple_json_from_html_string(
                    sanitized_html,
                    use_readability=True,
                )
                plain_content = readability_content["plain_content"]
            else:
                logger.debug("Using fallback readability method")
                doc = readability.Document(html_text)
                plain_content = doc.summary()

            if plain_content is None:
                raise ValueError("No content extracted")

        except Exception as e:
            logger.warning(
                f"Readability extraction failed, using BeautifulSoup fallback: {str(e)}"
            )
            # BeautifulSoup fallback
            soup = BeautifulSoup(html_text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            plain_content = soup.get_text()

        md_content = markdownify(plain_content, strip=["a", "img"], heading_style="ATX")
        logger.debug(
            f"HTML to markdown transformation completed, length: {len(md_content)}"
        )
        return md_content

    async def aget_title(
        self, url: str, headers=dict(), content_type=None, **kwargs
    ) -> str:
        logger.debug(f"Fetching title from: {url}")

        try:
            async with httpx.AsyncClient(
                timeout=10,
                follow_redirects=True,
                headers=headers,
            ) as client:
                async with client.stream("GET", _clean_url(url), **kwargs) as stream:
                    stream.raise_for_status()
                    if (
                        content_type is not None
                        and content_type not in stream.headers["Content-Type"]
                    ):
                        raise ContentTypeError

                    content = await anext(stream.aiter_bytes(1 * 1024 * 1024))

                    soup = BeautifulSoup(content.decode("utf-8"), "html.parser")
                    title_tag = soup.find("title").get_text(strip=True)

            if title_tag:
                logger.debug(f"Title extracted: {title_tag}")
            else:
                logger.warning(f"No title found for {url}")

            return title_tag or None

        except Exception as e:
            logger.error(f"Error fetching title from {url}: {str(e)}")
            raise


class ChromeExtensionMixin(HTMLToMarkdownMixin):
    llm: BaseLanguageModel | RunnableBinding

    async def get_article_content(self, url: str, user_id: str) -> str:
        logger.debug(f"Getting article content for user {user_id} from {url}")

        try:
            async with httpx.AsyncClient() as client:
                api_url = os.environ.get("ALAN_OPENAPI_ENDPOINT") + "/api/v1/cache/page"
                response = await client.get(
                    api_url,
                    params={"user_id": user_id, "url": url},
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("content", "")

            result = self._transform_html_readability_markdown(text)
            logger.info(
                f"Article content retrieved successfully for {url} ({len(result)} characters)"
            )
            return result

        except Exception as e:
            logger.error(f"Error getting article content from {url}: {str(e)}")
            raise

    async def read_and_simplify(
        self, url: str, user_id: str, content: Optional[str] = None
    ) -> tuple[str, dict[str, Any]]:
        logger.info(f"Simplifying content for user {user_id} from {url}")

        try:
            full_text = await self.get_article_content(url, user_id)

            if content is None:
                logger.debug("Simplifying full article content")
                # content가 없을 때는 전체 본문을 사용하고, 다른 프롬프트 구조 사용
                prompt = f"""
You are an expert at simplifying complex texts so that third-grade elementary school students can understand them easily. Read the given text and rewrite it in Korean so that young students can understand it without difficulty.

Here is the text to simplify:

<original_text>
{full_text}
</original_text>

Please simplify the text according to the following guidelines:

1. Use short and simple sentences.
2. Avoid difficult words or technical terms. If you must use them, explain them in easy words.
3. Use examples from everyday life to explain complicated ideas.
4. You may use analogies or metaphors to help with understanding, but keep them simple.
5. Keep the important information, but it's okay to leave out unnecessary details.

The output must be written in **Korean**.

Keep the original meaning, but make sure a third-grade elementary school student can understand it. If there are difficult concepts, explain them in a simple way, and use familiar examples when necessary.

Now, read the original text and rewrite it in easy Korean for young students to understand.
"""
            else:
                logger.debug("Simplifying specific content section")
                # content가 있을 때는 content를 사용하고, 전체 본문을 문맥으로 사용
                prompt = f"""
You are an expert at simplifying complex texts so that third-grade elementary school students can understand them easily. Read the given text and rewrite it in Korean so that young students can understand it without difficulty.

<context>
{full_text}
</context>

Here is the specific part of the text to simplify:

<original_text>
{content}
</original_text>

Please simplify the text according to the following guidelines:

1. Use short and simple sentences.
2. Avoid difficult words or technical terms. If you must use them, explain them in easy words.
3. Use examples from everyday life to explain complicated ideas.
4. You may use analogies or metaphors to help with understanding, but keep them simple.
5. Keep the important information, but it's okay to leave out unnecessary details.

The output must be written in **Korean**.

Keep the original meaning, but make sure a third-grade elementary school student can understand it. If there are difficult concepts, explain them in a simple way, and use familiar examples when necessary.

Now, read the original text and rewrite it in easy Korean for young students to understand.
"""

            logger.debug("Invoking LLM for content simplification")
            response = await self.llm.ainvoke(prompt)

            conversation_state = self._create_conversation_state(
                "쉽게 읽기", response.content, response.response_metadata
            )

            logger.info("Content simplification completed successfully")
            return response.content, conversation_state

        except Exception as e:
            logger.error(f"Error in read_and_simplify: {str(e)}")
            raise

    async def summarize_by_level(
        self, url: str, user_id: str, level: str, content: Optional[str] = None
    ) -> tuple[str, dict[str, Any]]:
        logger.info(
            f"Summarizing content by level '{level}' for user {user_id} from {url}"
        )

        try:
            full_text = await self.get_article_content(url, user_id)

            if level.lower() == "middle":
                logger.debug("Creating middle-level summary")
                if content is None:
                    # content가 없을 때는 전체 본문을 요약하는 프롬프트 사용
                    prompt = f"""
You are an expert at concisely summarizing text content. Please read the following text and create a short-level summary.

<original_text>
{full_text}
</original_text>

Short-level summary guidelines:
1. Include only the main flow and key points of the text.
2. Omit details and focus on important concepts and ideas.
3. Write the summary in Korean.
4. Use short and simple sentences.
5. Format your summary as a markdown bullet list, with each key point as a separate bullet point.
6. Each bullet point should start with "- " and contain a complete thought or concept.

Please write a short-level summary following the above guidelines.
"""
                else:
                    # content가 있을 때는 content를 요약하고, 전체 본문을 문맥으로 사용
                    prompt = f"""
You are an expert at concisely summarizing text content. Please read the following text and create a short-level summary.
    
<context>
{full_text}
</context>

Here is the specific part of the text to summarize:

<original_text>
{content}
</original_text>

Short-level summary guidelines:
1. Include only the main flow and key points of the text.
2. Omit details and focus on important concepts and ideas.
3. Write the summary in Korean.
4. Use short and simple sentences.
5. Format your summary as a markdown bullet list, with each key point as a separate bullet point.
6. Each bullet point should start with "- " and contain a complete thought or concept.

Please write a short-level summary following the above guidelines.
"""
                user_content = "중간길이 요약"

            elif level.lower() == "long":
                logger.debug("Creating long-level summary")
                if content is None:
                    # content가 없을 때는 전체 본문을 요약하는 프롬프트 사용
                    prompt = f"""
You are an expert at systematically summarizing detailed text content. Please read the following text and create a medium-level summary.

<original_text>
{full_text}
</original_text>

Medium-level summary guidelines:
1. Include only the main flow and key points of the text.
2. Omit details and focus on important concepts and ideas.
3. Write the summary in Korean.
4. Follow the structure of the original text but exclude unnecessary repetition or additional explanations.
5. Format your summary as a markdown bullet list, with each key point as a separate bullet point.
6. Each bullet point should start with "- " and contain a complete thought or concept.

Please write a medium-level summary following the above guidelines.
"""
                else:
                    # content가 있을 때는 content를 요약하고, 전체 본문을 문맥으로 사용
                    prompt = f"""
You are an expert at systematically summarizing detailed text content. Please read the following text and create a medium-level summary.

<context>
{full_text}
</context>

Here is the specific part of the text to summarize:

<original_text>
{content}
</original_text>

Medium-level summary guidelines:
1. Include only the main flow and key points of the text.
2. Omit details and focus on important concepts and ideas.
3. Write the summary in Korean.
4. Follow the structure of the original text but exclude unnecessary repetition or additional explanations.
5. Format your summary as a markdown bullet list, with each key point as a separate bullet point.
6. Each bullet point should start with "- " and contain a complete thought or concept.

Please write a medium-level summary following the above guidelines.
"""
                user_content = "길게 요약"
            else:
                logger.error(f"Invalid summary level: {level}")
                raise ValueError(f"Invalid level: {level}")

            logger.debug("Invoking LLM for summary generation")
            response = await self.llm.ainvoke(prompt)

            # Create conversation state entries
            conversation_state = self._create_conversation_state(
                user_content, response.content, response.response_metadata
            )

            logger.info(f"Summary by level '{level}' completed successfully")
            return response.content, conversation_state

        except Exception as e:
            logger.error(f"Error in summarize_by_level: {str(e)}")
            raise

    async def translate(
        self, url: str, user_id: str, content: Optional[str] = None
    ) -> tuple[str, dict[str, Any]]:
        logger.info(f"Translating content for user {user_id} from {url}")

        try:
            full_text = await self.get_article_content(url, user_id)

            if content is None:
                logger.debug("Translating full article content")
                # content가 없을 때는 전체 본문을 번역
                prompt = f"""
You are a highly skilled translator specializing in converting text into clear, natural-sounding Korean. Please translate the following text into Korean.

<original_text>
{full_text}
</original_text>

Translation guidelines:
1. Translate the text into natural, fluent Korean.
2. Preserve the original meaning, tone, and style.
3. Maintain any formatting, special characters, or technical terms appropriately.
4. Ensure the translation is culturally appropriate and easily understandable by Korean readers.
5. If there are ambiguous terms or phrases, choose the most likely meaning based on context.

Please provide only the Korean translation without additional comments or explanations.
"""
            else:
                logger.debug("Translating specific content section")
                # content가 있을 때는 content를 번역하고, 전체 본문을 문맥으로 사용
                prompt = f"""
You are a highly skilled translator specializing in converting text into clear, natural-sounding Korean. Please translate the following text into Korean.

<context>
{full_text}
</context>

Here is the specific part of the text to translate:

<original_text>
{content}
</original_text>

Translation guidelines:
1. Translate the text into natural, fluent Korean.
2. Preserve the original meaning, tone, and style.
3. Maintain any formatting, special characters, or technical terms appropriately.
4. Ensure the translation is culturally appropriate and easily understandable by Korean readers.
5. If there are ambiguous terms or phrases, choose the most likely meaning based on context.
6. Use the context provided to ensure accurate translation of terms that might be ambiguous without context.

Please provide only the Korean translation without additional comments or explanations.
"""

            logger.debug("Invoking LLM for translation")
            response = await self.llm.ainvoke(prompt)

            # Create conversation state entries
            conversation_state = self._create_conversation_state(
                "번역", response.content, response.response_metadata
            )

            logger.info("Translation completed successfully")
            return response.content, conversation_state

        except Exception as e:
            logger.error(f"Error in translate: {str(e)}")
            raise

    def _create_conversation_state(
        self, user_content: str, ai_content: str, response_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        logger.debug(f"Creating conversation state for action: {user_content}")

        try:
            # current_state = self.dump()
            state = {
                "messages": []
            }  # TODO: 임시로 작성한 것. 따로 함수 등을 만들어서 AlanState에 적합한 형태로 받을 수 있어야...
            query_id = str(uuid.uuid4())

            user_message = {
                "type": "human",
                "data": {
                    "content": user_content,
                    "additional_kwargs": {"query_id": query_id},
                    "response_metadata": {},
                    "type": "human",
                    "name": None,
                    "id": None,
                    "example": False,
                },
            }

            ai_message = {
                "type": "AIMessageChunk",
                "data": {
                    "content": ai_content,
                    "additional_kwargs": {
                        "type": "answer",
                        "answer_tool": "do_nothing",
                        "stop_reason": "stop",
                        "query_id": query_id,
                    },
                    "response_metadata": {
                        "finish_reason": "stop",
                        "model_name": response_metadata["model_name"],
                        "system_fingerprint": response_metadata.get(
                            "system_fingerprint", "fp_ded0d14823"
                        ),
                    },
                    "type": "AIMessageChunk",
                    "name": None,
                    "id": f"run-{str(uuid.uuid4())}",
                    "example": False,
                    "tool_calls": [],
                    "invalid_tool_calls": [],
                    "usage_metadata": None,
                    "tool_call_chunks": [],
                },
            }

            state["messages"].append(user_message)
            state["messages"].append(ai_message)
            state["summaries"] = [
                f'I was created. I received a query from a human asking about "{user_content}".',
                f'I was created. I received a query from a human asking about "{user_content}". I provided information that {ai_content[:100]}...',
            ]
            state["references"] = []

            logger.debug("Conversation state created successfully")
            return state

        except Exception as e:
            logger.error(f"Error creating conversation state: {str(e)}")
            raise
