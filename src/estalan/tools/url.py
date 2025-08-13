import base64
import os
import re
import xml.etree.ElementTree as ET
from typing import Annotated, Optional
from xml.sax.saxutils import unescape as unescape_xml

import httpx
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain.schema import BaseMessage, Document, HumanMessage
from langchain_core.callbacks.manager import adispatch_custom_event
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from estalan.logging_config import get_logger
from estalan.tools.base import AsyncTool
from estalan.tools.mixins import (
    ContentTypeError,
    HTMLToMarkdownMixin,
    HTTPXMixin,
    MessageMixin,
)
from estalan.tools.summarize import MapReduceSummarizationSubgraph
from estalan.tools.utils import noop

RAPID_API_HOST = os.getenv("RAPID_API_ENDPOINT").replace("https://", "")
RAPID_API_KEY = os.getenv("RAPID_API_KEY")
HEADERS = {
    "x-rapidapi-host": RAPID_API_HOST,
    "x-rapidapi-key": RAPID_API_KEY,
}
CLIENT_TIMEOUT = 10  # seconds

logger = get_logger(__name__)

youtube_url_pattern = re.compile(
    r"""^(?:https?:\/\/)?
    (?:www\.|m\.)?
    (?:youtube(?:-nocookie)?\.com|youtu\.be)
    \/
    (?:
        (?:watch(?:\?(?:.*&)?v=|\/))
      | (?:v|embed|e|shorts|live)\/
    )?
    ([A-Za-z0-9_-]{11})
    (?:[?&][^#]*)?
    (?:\#.*)?
    $""",
    re.VERBOSE,
)


class HTTPURLArgs(BaseModel):
    urls: list[str] = Field(
        ...,
        description="URL of the pages to visit (maximum 5)",
        min_items=1,
        max_items=5,
    )
    messages: Annotated[list[BaseMessage], InjectedState("messages")] = []
    verbose: bool = Field(
        default_factory=bool,
    )


class HTTPURLArgsForDeepSearch(BaseModel):
    urls: list[str] = Field(
        ...,
        description="URL of the pages to visit (maximum 5)",
        min_items=1,
        max_items=5,
    )
    messages: Annotated[list[BaseMessage], InjectedState("unit_messages")] = []
    verbose: bool = Field(
        default_factory=bool,
    )


class HTTPRefArgs(BaseModel):
    refs: list[int] = Field(
        ...,
        description="Reference numbers to visit (maximum 5)",
    )
    messages: Annotated[list[BaseMessage], InjectedState("messages")] = []
    references: Annotated[list[dict], InjectedState("references")] = {}
    verbose: bool = Field(
        default_factory=bool,
    )


class URLPattern:
    DOMAIN = re.compile(
        r"(http(s)?:\/\/)?(www\.)?([-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6})\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
    )

    @classmethod
    def extract_domain(cls, url: str) -> Optional[str]:
        match = cls.DOMAIN.match(url)
        domain = match.group(4) if match else None
        logger.debug(f"Extracted domain from {url}: {domain}")
        return domain


class URLProcessor:
    def __init__(self, patterns: dict[str, str]):
        self.patterns = patterns
        logger.debug(f"URLProcessor initialized with {len(patterns)} patterns")

    def process_url(self, url: str) -> str:
        logger.debug(f"Processing URL: {url}")
        original_url = url

        for pattern, replacement in self.patterns.items():
            match = re.search(pattern, url)
            if match:
                url = replacement.format(*match.groups())
                logger.debug(f"URL transformed: {original_url} -> {url}")
                break

        return url


class ContentFetcher(HTTPXMixin):
    def __init__(self):
        self.url_processor = URLProcessor(
            {
                r"^https?:\/\/(?:m\.)?blog\.naver\.com\/([^\/?#]+)\/([0-9]+)\/?": "https://m.blog.naver.com/PostView.naver?blogId={0}&logNo={1}"
            }
        )
        logger.debug("ContentFetcher initialized")

    async def fetch_content(self, url: str, headers: dict) -> Document:
        logger.info(f"Fetching content from: {url}")

        try:
            processed_url = self.url_processor.process_url(url)

            if processed_url.endswith(".pdf"):
                logger.debug("Detected PDF file, using PDF fetcher")
                return await self._fetch_pdf(processed_url)
            elif youtube_url_pattern.match(processed_url):
                logger.debug("Detected YouTube URL, using YouTube fetcher")
                # youtube 영상 url 종류
                # 도메인: youtube.com, youtu.be, youtube-nocookie.com
                # 경로:
                #   - /watch : URL 내에 쿼리 파라미터로 v=가 있거나, /watch/ 뒤에 직접 영상ID가 오는 경우
                #   - /v/, /embed/, /e/, /shorts/, /live/ : 바로 뒤에 직접 영상ID가 오는 경우
                # 영상 ID: YouTube 영상 ID는 [A-Za-z0-9_-]{11} 형태로 11자리.
                return await self._fetch_youtube(processed_url)

            logger.debug("Using HTML fetcher for regular web page")
            return await self._fetch_html(processed_url, headers)

        except ContentTypeError as e:
            logger.warning(f"Content type error for {url}: {str(e)}")
            return Document(
                page_content="올바른 타입의 컨텐츠가 아닙니다.",
                metadata={
                    "error": type(e).__name__,
                    "error_content": str(e),
                    "source": url,
                },
            )
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return Document(
                page_content="콘텐츠에 접근할 수 없습니다.",
                metadata={
                    "error": type(e).__name__,
                    "error_content": str(e),
                    "source": url,
                },
            )

    async def _fetch_pdf(self, url: str) -> Document:
        logger.debug(f"Fetching PDF content from: {url}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                pdf = base64.b64encode(response.content).decode("utf-8")

            logger.info(
                f"PDF fetched successfully from {url} ({len(pdf)} base64 characters)"
            )
            return Document(page_content=pdf, metadata={"source": url, "type": "pdf"})

        except Exception as e:
            logger.error(f"Error fetching PDF from {url}: {str(e)}")
            raise

    async def _fetch_html(self, url: str, headers: dict) -> Document:
        logger.debug(f"Fetching HTML content from: {url}")

        try:
            html_text = await self.aget(url, headers, content_type="text")
            logger.info(
                f"HTML fetched successfully from {url} ({len(html_text)} characters)"
            )
            return Document(
                page_content=html_text, metadata={"source": url, "type": "text"}
            )
        except Exception as e:
            logger.error(f"Error fetching HTML from {url}: {str(e)}")
            raise

    async def _fetch_youtube(self, url: str) -> Document:
        logger.debug(f"Fetching YouTube transcript from: {url}")

        try:
            # get video id from url
            pattern = re.compile(
                r'(?:youtube(?:-nocookie)?\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?/?|.*[?&]v=)|(?:embed/|shorts/|live/))|youtu\.be/)([^"&?/ ]{11})'
            )
            match = pattern.search(url)
            video_id = match.group(1)
            logger.debug(f"Extracted YouTube video ID: {video_id}")
            proxies = {
                "http": f"http://{os.environ['PROXY_ID']}:{os.environ['PROXY_PASSWORD']}@{os.environ['PROXY_HOST']}:{os.environ['PROXY_PORT']}",
                "https": f"https://{os.environ['PROXY_ID']}:{os.environ['PROXY_PASSWORD']}@{os.environ['PROXY_HOST']}:{os.environ['PROXY_PORT']}",
            }
            try:
                transcript_info = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=LANGUAGE_CODES, proxies=proxies
                )
                transcript_info = sorted(transcript_info, key=lambda x: x["start"])
                full_scripts = [script["text"].strip() for script in transcript_info]
                transcript_text = "\n".join(full_scripts)

            except Exception as e:
                logger.warning(
                    f"Error fetching YouTube script via YouTubeTranscriptApi: {e}"
                )
                transcript_text = await self._fetch_youtube_script(video_id)

            logger.info(
                f"YouTube transcript fetched successfully for {video_id} ({len(transcript_text)} characters)"
            )

            return Document(
                page_content=transcript_text,
                metadata={"source": url, "type": "youtube_summary"},
            )
        except Exception as e:
            logger.error(f"Error fetching YouTube transcript from {url}: {str(e)}")
            raise

    async def _fetch_youtube_script(self, video_id: str) -> Optional[str]:
        rapid_api_endpoint = os.getenv("RAPID_API_ENDPOINT")
        rapid_api_key = os.getenv("RAPID_API_KEY")
        if not (rapid_api_endpoint and rapid_api_key):
            logger.error("RAPID API key is not configured")
            return None

        try:
            async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT) as client:
                list_url = f"{rapid_api_endpoint}/subtitles"
                resp = await client.get(
                    list_url,
                    params={"id": video_id},
                    headers={
                        "x-rapidapi-host": rapid_api_endpoint.replace("https://", ""),
                        "x-rapidapi-key": rapid_api_key,
                    },
                )
                resp.raise_for_status()
                subtitles = resp.json().get("subtitles", [])

                urls_map = {st["languageCode"]: st["url"] for st in subtitles}
                url = next(
                    (urls_map[lc] for lc in LANGUAGE_CODES if lc in urls_map), None
                )
                if url:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    text = await self.extract_text_from_transcript(resp.text.strip())
                    if text:
                        return text
                logger.info(f"No matching subtitles for video {video_id}")

        except Exception as e:
            logger.error(f"Failed to fetch YouTube script: {e}")

        raise TranscriptsDisabled(video_id)

    async def extract_text_from_transcript(self, xml_content: str) -> Optional[str]:
        try:
            root = ET.fromstring(xml_content)

            texts = []
            for text_element in root.findall(".//text"):
                content = text_element.text
                if content:
                    content = unescape_xml(content, {"&quot;": '"'})
                    content = re.sub(r"\[.*?\]", "", content).strip()

                    if content:
                        texts.append(content)

            return " ".join(texts)

        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error: {e}")
            return None


class BaseHTTPTool(AsyncTool, HTMLToMarkdownMixin, MessageMixin):
    name: str = ""
    description: str = ""
    summarize_graph: MapReduceSummarizationSubgraph
    content_fetcher: ContentFetcher = ContentFetcher()

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        total_limit_tokens: int = 4096,
    ) -> AsyncTool:
        logger.debug(f"Creating BaseHTTPTool with token limit: {total_limit_tokens}")
        summarize_graph = MapReduceSummarizationSubgraph.from_llm(
            llm=llm, chunk_size=total_limit_tokens
        )

        return cls(
            summarize_graph=summarize_graph,
        )

    async def _fetch_and_preprocess(self, url: str) -> Document:
        logger.debug(f"Fetching and preprocessing: {url}")

        try:
            doc = await self.content_fetcher.fetch_content(url, {"Accept": "text/*"})
            if "error" in doc.metadata:
                logger.warning(f"Error in fetched document: {doc.metadata['error']}")
                return doc

            md_doc = await self.aclean_html(doc)
            md_doc.page_content = re.sub(r"\n+", " ", md_doc.page_content)

            logger.debug(
                f"Document preprocessed successfully: {len(md_doc.page_content)} characters"
            )
            return md_doc

        except Exception as e:
            logger.error(f"Error in fetch and preprocess for {url}: {str(e)}")
            raise


class URLSummarizeTool(BaseHTTPTool):
    name: str = "summarize_url"
    description: str = (
        "Use a website URL to either browse its content or get a summarized version."
    )
    args_schema: type[BaseModel] = HTTPURLArgs

    def verify_args(self, user_message: str, args: dict) -> bool:
        logger.debug(f"Verifying args for URLSummarizeTool: {args}")
        urls = args["urls"]

        for url in urls:
            domain = URLPattern.extract_domain(url)
            if domain is None:
                logger.warning(f"Invalid domain for URL: {url}")
                return False
            if domain.lower() not in user_message.lower():
                logger.warning(f"Domain {domain} not found in user message")
                return False

        logger.debug("Args verification passed")
        return True

    async def _arun(
        self,
        urls: list[str],
        messages: list[dict] = [],
        verbose: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> list:
        logger.info(f"Starting URL summarization for {len(urls)} URLs")
        dispatcher = adispatch_custom_event if verbose else noop

        urls = [url if url.startswith("http") else "https://" + url for url in urls]
        logger.debug(f"Processed URLs: {urls}")

        md_docs = []
        for url in urls:
            logger.debug(f"Processing URL: {url}")
            md_doc = await self._fetch_and_preprocess(url)
            md_docs.append(md_doc)

        filtered = [doc for doc in md_docs if "error" not in doc.metadata]
        logger.debug(
            f"Filtered documents: {len(filtered)} out of {len(md_docs)} successful"
        )

        if not filtered:
            logger.warning("No accessible sources found")
            return [
                {
                    "page_content": "접근할 수 있는 출처가 없습니다.",
                    "metadata": {"error": "Error"},
                }
            ]

        # Extract titles
        for url, doc in zip(urls, filtered):
            doc.metadata["thumbnail"] = None
            try:
                logger.debug(f"Extracting title for: {url}")
                doc.metadata["title"] = await self.aget_title(
                    url=url, headers={"Accept": "text/*"}, content_type="text"
                )
            except Exception as e:
                logger.warning(f"Failed to extract title for {url}: {str(e)}")
                doc.metadata["title"] = None

        # Prepare display information
        titles, urls_display = [], []
        for doc in filtered:
            title = doc.metadata["title"]
            source = doc.metadata["source"]
            if title is not None:
                if len(title) > 15:
                    title = f"{title[:15]}..."
                titles.append(f"'{title}'")
            else:
                if len(source) > 30:
                    source = f"{source[:30]}..."
                urls_display.append(source)

        combined_items = titles + urls_display
        display_items = combined_items[:4]

        remainder_count = len(combined_items) - 4 if len(combined_items) > 4 else 0
        keywords = ", ".join([f"{item}" for item in display_items]) + (
            f" 외 {remainder_count}개를" if remainder_count > 0 else "을"
        )

        await dispatcher("event", {"speak": f"{keywords} 읽고 있어요."})

        if len(filtered) < len(urls):
            await dispatcher(
                "event",
                {
                    "speak": "일부 컨텐츠에 접근하지 못했어요. 접근 가능한 웹 사이트의 요약을 시작합니다."
                },
            )
        else:
            await dispatcher("event", {"speak": "웹 사이트 요약을 시작합니다."})

        logger.debug("Starting batch summarization")
        results = await self.summarize_graph.abatch(
            md_docs, user_query=self.get_last_message(messages, HumanMessage).content
        )

        if len(md_docs) > 1:
            if len(filtered) < len(urls):
                await dispatcher(
                    "event", {"speak": "접근 가능한 사이트의 요약을 모두 완료했습니다."}
                )
            else:
                await dispatcher(
                    "event", {"speak": "요청하신 사이트의 요약을 완료했습니다."}
                )
        else:
            await dispatcher(
                "event", {"speak": "요청하신 사이트의 요약을 모두 완료했습니다."}
            )

        logger.info(f"URL summarization completed with {len(results)} results")
        return results


class URLSummarizeToolForDeepSearch(URLSummarizeTool):
    args_schema: type[BaseModel] = HTTPURLArgsForDeepSearch


class RefSummarizeTool(BaseHTTPTool):
    name: str = "summarize_references"
    description: str = "Summarize the full content of referenced sources."
    args_schema: type[BaseModel] = HTTPRefArgs

    async def _arun(
        self,
        refs: list[int],
        messages: list[dict] = [],
        references: list[dict] = [],
        verbose: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> list:
        logger.info(f"Starting reference summarization for refs: {refs}")
        dispatcher = adispatch_custom_event if verbose else noop

        referenced_idxs = [
            ref["number"]
            for ref in references
            if ref.get("number") in refs and ref.get("source") is not None
        ][:5]

        logger.debug(
            f"Found {len(referenced_idxs)} valid references: {referenced_idxs}"
        )

        await dispatcher(
            "event", {"speak": "검색 결과 중 요청하신 출처를 더 자세히 읽고 있어요."}
        )

        if not referenced_idxs:
            logger.warning("No accessible references found")
            return [
                {
                    "page_content": "접근할 수 있는 출처가 없습니다.",
                    "metadata": {"error": "Error"},
                }
            ]

        referenced_urls = [
            ref["source"] for ref in references if ref.get("number") in referenced_idxs
        ]

        # Remove duplicate URLs
        unique_urls = {}
        for idx, url in zip(referenced_idxs, referenced_urls):
            if url not in unique_urls:
                unique_urls[url] = idx

        referenced_urls, referenced_idxs = map(list, zip(*unique_urls.items()))
        logger.debug(f"Processing {len(referenced_urls)} unique URLs")

        md_docs = []
        for idx, url in zip(referenced_idxs, referenced_urls):
            logger.debug(f"Processing reference {idx}: {url}")
            md_doc = await self._fetch_and_preprocess(url)
            try:
                md_doc.metadata["title"] = await self.aget_title(
                    url=url, headers={"Accept": "text/*"}, content_type="text"
                )
            except Exception as e:
                logger.warning(f"Failed to extract title for reference {idx}: {str(e)}")
                md_doc.metadata["title"] = None

            md_docs.append(md_doc)

        # Prepare display information
        titles, urls_display = [], []
        for doc in md_docs:
            title = doc.metadata["title"]
            source = doc.metadata["source"]
            if title is not None:
                if len(title) > 15:
                    title = f"{title[:15]}..."
                titles.append(f"'{title}'")
            else:
                if len(source) > 30:
                    source = f"{source[:30]}..."
                urls_display.append(source)

        combined_items = titles + urls_display
        display_items = combined_items[:4]

        remainder_count = len(combined_items) - 4 if len(combined_items) > 4 else 0

        keywords = ", ".join([f"{item}" for item in display_items]) + (
            f" 외 {remainder_count}개" if remainder_count > 0 else ""
        )

        await dispatcher("event", {"speak": f"{keywords} 페이지를 자세히 보는 중..."})

        logger.debug("Starting batch summarization for references")
        results = await self.summarize_graph.abatch(
            md_docs, user_query=messages[-2].content
        )

        # md_docs내의 문서는 referenced_idxs와 일치하고, abatch 내의 gather는 return 순서를 보장하므로.
        for idx, url, result in zip(referenced_idxs, referenced_urls, results):
            # dict이므로, call by reference로 작동하므로.
            result["metadata"]["source_no"] = idx
            result["metadata"]["thumbnail"] = None

        if len(refs) > 1:
            await dispatcher("event", {"speak": "필요한 내용을 모두 확인했어요."})

        logger.info(f"Reference summarization completed with {len(results)} results")
        return results
