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

from estalan.logging.base import get_logger
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
        logger.debug("검색 시작", search_type=self.__class__.__name__, query_count=len(query or []))
        dispatcher = adispatch_custom_event if verbose else noop

        query = query or ["뉴스"]
        if isinstance(query, str):
            query = [query]

        query = list(set(query))
        logger.debug("검색 쿼리 정리됨", queries=query, unique_count=len(query))

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
            logger.debug("개별 쿼리 결과 가져오기", query=q)
            result = await self.api_wrapper.aresults(q)
            parsed = self._parse_results(result)
            logger.debug("쿼리 결과 파싱 완료", query=q, result_count=len(parsed))
            return parsed

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
            logger.debug("검색 결과 수집 완료", total_results=len(results))

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
            logger.info("검색 완료", search_type=self.__class__.__name__, 
                       original_results=len(results), merged_results=len(final_results))
            return final_results

        except Exception as e:
            logger.error("검색 중 오류 발생", search_type=self.__class__.__name__, error=str(e))
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
        logger.debug("GoogleSerperSearchResult 생성", k=k)
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

        logger.debug("검색 결과 파싱 완료", result_count=len(docs))
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
        logger.debug("GoogleSerperNewsResult 생성", k=k)
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

        logger.debug("뉴스 결과 파싱 완료", result_count=len(docs))
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
        logger.debug("GoogleSerperImageSearchResult 생성", k=k)
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
            logger.warning("검색 결과에 이미지가 없음")
            return docs
        image_results = results["images"]

        num_pass = 0
        for result in image_results:
            # 필수 필드 확인
            if not all(key in result for key in ["title", "imageUrl", "link"]):
                continue
            metadata = {
                "title": result["title"],
                # "link": result["link"],
                "image_url": result["imageUrl"],
                "imageWidth": result.get("imageWidth"),
                "imageHeight": result.get("imageHeight"),
                # "thumbnail_url": result.get("thumbnailUrl"),
                # "thumbnailWidth": result.get("thumbnailWidth"),
                # "thumbnailHeight": result.get("thumbnailHeight"),
                # "domain": result.get("domain"),
                # "source": result.get("source"),
                # "google_url": result.get("googleUrl"),
                "position": result.get("position"),
                "type": "image",
            }


            if result["imageUrl"] is None:
                continue

            # 이미지 URL 확장자 필터링 (jpg, jpeg, png만 허용)
            image_url = result["imageUrl"].lower()
            if not any(image_url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                continue

            # cors_violation = is_cors_violation(result["imageUrl"])
            # if cors_violation:
            #     print(f"CORS violation detected for {result['imageUrl']}")
            # else:
            #     print(f"Pass CORS for {result['imageUrl']}")
            #
            # num_pass += 1
            # print(f"{num_pass} / {self.k}")

            docs.append(
                {
                    "page_content": result["title"],
                    "metadata": {k: v for k, v in metadata.items() if v is not None},
                }
            )
            # print(f"docs : {len(docs)}")
            if num_pass > self.k:
                break

        logger.debug("이미지 결과 파싱 완료", result_count=len(docs))
        return docs

    async def _arun(
        self,
        query: list[str] | None = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs,
    ):
        logger.debug("이미지 검색 시작", query_count=len(query or []))

        query = query or ["이미지"]

        if isinstance(query, str):
            query = [query]
        results = []

        async def fetch_results(q):
            logger.debug("이미지 검색 결과 가져오기", query=q)
            result = await self.api_wrapper.aresults(q)
            parsed = self._parse_results(result)
            logger.debug("이미지 결과 파싱 완료", query=q, result_count=len(parsed))
            return parsed

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
            logger.info("이미지 검색 완료", unique_results=len(final_results), total_results=len(results))
            return final_results

        except Exception as e:
            logger.error("이미지 검색 중 오류 발생", error=str(e))
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
        logger.debug("RapidYoutubeSearchResult 생성", k=k, api_endpoint=api_endpoint)
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
        logger.debug("YouTube 비디오 검색 시작", query_count=len(query or []))

        try:
            videos = []
            video_ids = set()

            for keyword in query:
                logger.debug("키워드별 비디오 검색", keyword=keyword)
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

            logger.info("YouTube 검색 완료", unique_videos=len(results), total_videos=len(videos))
            return results

        except Exception as e:
            logger.error("YouTube 비디오 검색 중 오류 발생", error=str(e))
            raise


import requests
from urllib.parse import urlparse


def get_origin_from_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

def is_cors_violation(url: str, method="GET", headers=None) -> bool:
    """check cors violation."""
    origin = get_origin_from_url(url)  # URL에서 자동 추출

    try:
        response = requests.head(url, timeout=5)
        allow_origin = response.headers.get("Access-Control-Allow-Origin")

        # CORS 헤더가 없거나 요청 origin과 매칭되지 않으면 False
        if not allow_origin:
            return True

        if allow_origin == "*" or allow_origin == origin:
            return False

        return True

    except requests.RequestException:
        return True
