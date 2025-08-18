import asyncio
import re
from abc import abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from estalan.logging_config import get_logger
from estalan.tools.base import AsyncTool
from estalan.tools.utils import noop, retry_on_api_empty

logger = get_logger(__name__)


class GoogleSerperSearchArgs(BaseModel):
    query: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "Generate 1 to 3 search queries in various languages relevant to the topic. "
            "Always include Korean queries regardless of the topic, and add queries in other relevant languages. "
            "Select the most appropriate keywords in each language to improve search result accuracy. "
            "Ensure diversity in the queries to cover different aspects of the topic. "
            "For words to be excluded from the search, use exclude_words parameter. "
            "For site-specific searches, use search_site parameter."
        ),
    )
    exclude_words: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "Use this parameter to exclude pages containing any of the words in the list."
            "The search will exclude pages containing any of the words in the list."
            "Example: ['tutorial', 'guide'] will exclude pages containing the word 'tutorial' or 'guide'."
        ),
    )
    search_site: Optional[str] = Field(
        default_factory=str,
        description=(
            "Use this parameter to restrict search results to a specific website or domain."
            "The search will only return pages from the specified domain."
            "If you want to search for all sources, leave it blank."
            "Example: Setting this parameter to 'wikipedia.org' will only return results from Wikipedia."
        ),
    )
    verbose: bool = Field(
        default_factory=bool,
    )
    messages: Annotated[list[BaseMessage], InjectedState("messages")] = Field(
        default_factory=list
    )


class BaseGoogleSerperResult(AsyncTool):
    api_wrapper: GoogleSerperAPIWrapper
    args_schema: type[BaseModel] = GoogleSerperSearchArgs
    k: int

    async def _arun(
        self,
        query: list[str] | None = None,
        exclude_words: list[str] | None = None,
        search_site: str | None = None,
        verbose: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs,
    ) -> list:
        logger.debug(f"Starting {self.__class__.__name__} search")
        dispatcher = adispatch_custom_event if verbose else noop

        query = query or ["뉴스"]
        if isinstance(query, str):
            query = [query]

        query = list(set(query))
        logger.debug(f"Search queries: {query}")

        await dispatcher("event", {"keyword": query})

        query_w_options = []
        for q in query:
            if exclude_words or search_site:
                if exclude_words:
                    for w in exclude_words:
                        query_w_options.append(f'{q} -"{w}"')
                if search_site:
                    query_w_options.append(f"{q} site:{search_site}")
            else:
                query_w_options.append(q)

        keywords = ", ".join([f"‘{q}’" for q in query[:4]]) + (
            f" 외 {len(query) - 4}개의 키워드" if len(query) > 4 else ""
        )
        if self.__class__.__name__ != "GoogleSerperNewsResult":
            await dispatcher("event", {"speak": f"{keywords}로 검색하고 있어요."})

        async def fetch_results(q):
            logger.debug(f"Fetching results for query: {q}")
            result = await self.api_wrapper.aresults(q)
            return self._parse_results(result)

        @retry_on_api_empty()
        async def fetch_all_results(query: list[str]):
            results = []
            fetch_tasks = [fetch_results(q) for q in query]
            results_list = await asyncio.gather(*fetch_tasks)
            for result in results_list:
                results.extend(result)  # 결과를 합침

            return results

        try:
            results = await fetch_all_results(query_w_options) or []
            logger.debug(f"Retrieved {len(results)} search results")

            if self.__class__.__name__ != "GoogleSerperNewsResult" and results:
                await dispatcher(
                    "event", {"speak": f"검색 결과 {len(results)}개를 읽고 있어요."}
                )

            merged_results = {}
            for doc in results:
                link = doc["metadata"].get("source")
                if link in merged_results:
                    if doc["page_content"] not in merged_results[link]["page_content"]:
                        merged_results[link]["page_content"].append(doc["page_content"])
                else:
                    merged_results[link] = doc
                    merged_results[link]["page_content"] = [doc["page_content"]]

            for doc in merged_results.values():
                doc["page_content"] = ", ".join(doc["page_content"])

            final_results = list(merged_results.values())
            logger.info(
                f"{self.__class__.__name__} completed with {len(final_results)} merged results"
            )
            return final_results

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {str(e)}")
            raise

    def convert_to_iso8601(self, time_str):
        if time_str is None:
            return None

        now = datetime.now(timezone(timedelta()))

        relative_match = re.match(
            r"(\d+)\s*(seconds?|minutes?|hours?|days?|weeks?|months?)\s*ago", time_str
        )
        if relative_match:
            value, unit = int(relative_match.group(1)), relative_match.group(2).lower()

            if "second" in unit:
                delta = timedelta(seconds=value)
            elif "minute" in unit:
                delta = timedelta(minutes=value)
            elif "hour" in unit:
                delta = timedelta(hours=value)
            elif "day" in unit:
                delta = timedelta(days=value)
            elif "week" in unit:
                delta = timedelta(weeks=value)
            elif "month" in unit:
                delta = timedelta(days=value * 30)
            else:
                return None

            past_time = now - delta
            return past_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            absolute_time = datetime.strptime(time_str, "%b %d, %Y")
            absolute_time = absolute_time.replace(tzinfo=timezone(timedelta()))
            return absolute_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None

    @abstractmethod
    def _parse_results(self, results: dict) -> list[dict]:
        pass


class GoogleSerperSearchResult(BaseGoogleSerperResult):
    name: str = "search_web"
    description: str = (
        "Search the web for any kind of information including technical solutions, error handling, best practices, and general facts. "
        "Search the web for information, even if you think you know the answer. "
        "This helps verify facts and provide the most up-to-date information with proper citations. "
        "Always prioritize using this tool when answering questions about facts, events, or specific information, as it provides reliable real-world sources."
    )

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        k: int = 5,
    ) -> AsyncTool:
        logger.debug(f"Creating GoogleSerperSearchResult with k={k}")
        return cls(
            api_wrapper=GoogleSerperAPIWrapper(
                hl="en",
                gl="kr",
                serper_api_key=api_key,
                type="search",
                k=k,
            ),
            k=k,
        )

    def _parse_results(
        self,
        results: dict,
    ):
        docs = []

        if (kg := results.get("knowledgeGraph", {})) and (
            description := kg.get("description")
        ):
            docs.append(
                {
                    "page_content": description,
                    "metadata": {
                        "title": kg.get("title"),
                        "date": self.convert_to_iso8601(kg.get("date")),
                        "thumbnail": kg.get("thumbnail_url"),
                        "kind": kg.get("type"),
                        "source": kg.get("descriptionLink", ""),
                        "attributes": kg.get("attributes"),
                    },
                }
            )

        for result in results["organic"]:
            if result.keys() & {"title", "snippet", "link"} != {
                "title",
                "snippet",
                "link",
            }:
                continue

            docs.append(
                {
                    "page_content": result["snippet"],
                    "metadata": {
                        "title": result["title"],
                        "date": self.convert_to_iso8601(result.get("date")),
                        "thumbnail": result.get("thumbnail_url"),
                        "source": result["link"],
                    },
                }
            )

        logger.debug(f"Parsed {len(docs)} search results")
        return docs


class GoogleSerperNewsResult(BaseGoogleSerperResult):
    name: str = "search_google_news"
    description: str = "Search for high-cost news, including relatively long summaries."

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        k: int = 5,
    ) -> AsyncTool:
        logger.debug(f"Creating GoogleSerperNewsResult with k={k}")
        return cls(
            api_wrapper=GoogleSerperAPIWrapper(
                hl="en",
                gl="kr",
                serper_api_key=api_key,
                type="news",
                k=k,
            ),
            k=k,
        )

    def _parse_results(
        self,
        results: dict,
    ):
        docs = []
        for result in results["news"]:
            if result.keys() & {"title", "snippet", "link"} != {
                "title",
                "snippet",
                "link",
            }:
                continue

            docs.append(
                {
                    "page_content": result["snippet"],
                    "metadata": {
                        "title": result["title"],
                        "date": self.convert_to_iso8601(result.get("date", None)),
                        "thumbnail": result.get("thumbnail_url"),
                        "source": result["link"],
                    },
                }
            )

        logger.debug(f"Parsed {len(docs)} news results")
        return docs


class GoogleSerperImageSearchResult(BaseGoogleSerperResult):
    name: str = "search_image"
    description: str = "Low-cost web search with image"

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        k: int = 10,
    ) -> AsyncTool:
        logger.debug(f"Creating GoogleSerperImageSearchResult with k={k}")
        return cls(
            api_wrapper=GoogleSerperAPIWrapper(
                hl="en",
                gl="kr",
                serper_api_key=api_key,
                type="images",
                k=k,
            ),
            k=k,
        )

    def _parse_results(
        self,
        results: dict,
    ):
        docs = []
        if "images" not in results:
            logger.warning("No images found in search results")
            return docs
        image_results = results["images"]

        num_pass = 0
        for result in image_results:
            # 필수 필드 확인
            if not all(key in result for key in ["title", "imageUrl", "link"]):
                continue
            metadata = {
                "title": result["title"],
                "link": result["link"],
                "image_url": result["imageUrl"],
                "imageWidth": result.get("imageWidth"),
                "imageHeight": result.get("imageHeight"),
                "thumbnail_url": result.get("thumbnailUrl"),
                "thumbnailWidth": result.get("thumbnailWidth"),
                "thumbnailHeight": result.get("thumbnailHeight"),
                "domain": result.get("domain"),
                "source": result.get("source"),
                "google_url": result.get("googleUrl"),
                "position": result.get("position"),
                "type": "image",
            }

            cors_violation = is_cors_violation(result["link"])
            if cors_violation:
                logger.warning(f"CORS violation detected for {result['link']}")
                continue

            docs.append(
                {
                    "page_content": result["title"],
                    "metadata": {k: v for k, v in metadata.items() if v is not None},
                }
            )
            num_pass += 1
            if num_pass > self.k:
                break

        logger.debug(f"Parsed {len(docs)} image results")
        return docs

    async def _arun(
        self,
        query: list[str] | None = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs,
    ):
        logger.debug("Starting image search")

        query = query or ["이미지"]

        if isinstance(query, str):
            query = [query]
        results = []

        async def fetch_results(q):
            logger.debug(f"Fetching image results for query: {q}")
            result = await self.api_wrapper.aresults(q)
            return self._parse_results(result)

        try:
            # 각 쿼리에 대해 결과를 비동기적으로 가져옴
            fetch_tasks = [fetch_results(q) for q in query]
            results_list = await asyncio.gather(*fetch_tasks)
            for result in results_list:
                results.extend(result)  # 결과를 합침
            merged_results = {}
            for doc in results:
                link = doc["metadata"].get("image_url")
                if link not in merged_results:
                    merged_results[link] = doc

            final_results = list(merged_results.values())
            logger.info(
                f"Image search completed with {len(final_results)} unique results"
            )
            return final_results

        except Exception as e:
            logger.error(f"Error in image search: {str(e)}")
            raise


class RapidYoutubeSearchResult(AsyncTool):
    name: str = "search_video"
    description: str = "Low-cost web search with video"

    api_key: str
    api_endpoint: str
    args_schema: type[BaseModel] = GoogleSerperSearchArgs
    k: int

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        api_endpoint: str,
        k: int = 10,
    ) -> AsyncTool:
        logger.debug(f"Creating RapidYoutubeSearchResult with k={k}")
        return cls(
            api_key=api_key,
            api_endpoint=api_endpoint,
            k=k,
        )

    async def _arun(
        self,
        query: list[str] | None = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs,
    ):
        logger.debug("Starting YouTube video search")

        try:
            videos = []
            video_ids = set()

            for keyword in query:
                logger.debug(f"Searching videos for keyword: {keyword}")
                for video in (
                    get_video_list_by_keyword(
                        keyword, api_key=self.api_key, api_endpoint=self.api_endpoint
                    )
                    or []
                ):
                    if video["video_id"] not in video_ids:
                        videos.append(video)
                        video_ids.add(video["video_id"])

            results = [
                {**video, "number": i, "metadata": {"type": "video"}}
                for i, video in enumerate(videos)
            ]

            logger.info(f"YouTube search completed with {len(results)} unique videos")
            return results

        except Exception as e:
            logger.error(f"Error in YouTube video search: {str(e)}")
            raise


import requests
from urllib.parse import urlparse


def get_origin_from_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

def is_cors_violation(url: str, method="GET", headers=None) -> bool:
    """check cors violation."""
    origin = get_origin_from_url(url)  # URL에서 자동 추출

    request_headers = headers.copy() if headers else {}
    request_headers["Origin"] = origin

    response = requests.request(method, url, headers=request_headers)

    allow_origin = response.headers.get("Access-Control-Allow-Origin")
    allow_methods = response.headers.get("Access-Control-Allow-Methods", "")
    allow_headers = response.headers.get("Access-Control-Allow-Headers", "")

    if allow_origin not in (origin, "*"):
        return True
    if method.upper() not in allow_methods.upper():
        return True
    if headers:
        allowed_headers_list = [h.strip().lower() for h in allow_headers.split(",")]
        for h in headers.keys():
            if h.lower() != "origin" and h.lower() not in allowed_headers_list:
                return True
    return False
