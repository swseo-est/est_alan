import asyncio
import operator
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from google.oauth2 import service_account
from langchain.chains.combine_documents.reduce import split_list_of_docs
from langchain.schema import Document
from langchain_community.callbacks.infino_callback import get_num_tokens
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import ChatVertexAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from estalan.core.prompt import SummaryPrompt
from estalan.logging_config import get_logger
from estalan.tools.utils import add_graph_components

load_dotenv()
logger = get_logger(__name__)

gemini_llm = ChatVertexAI(
    credentials=service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    ),
    model_name=os.getenv("GOOGLE_MODEL_NAME"),
    temperature=0.1,
    max_tokens=8192,
    max_retries=2,
).with_config(tags=["summarize"])


# TODO: 기존 앨런과 같은 세팅을 위해 임시로 삽입한 함수이기 때문에 추후 제거되어야 함.
async def acollapse_docs(
    docs: list[Document],
    combine_document_func,
    user_query: str,
) -> Document:
    """Execute a collapse function on a set of documents and merge their metadatas.

    Args:
        docs: A list of Documents to combine.
        combine_document_func: A function that takes in a list of Documents and
            optionally addition keyword parameters and combines them into a single
            string.
        **kwargs: Arbitrary additional keyword params to pass to the
            combine_document_func.

    Returns:
        A single Document with the output of combine_document_func for the page content
            and the combined metadata's of all the input documents. All metadata values
            are strings, and where there are overlapping keys across documents the
            values are joined by ", ".
    """
    logger.debug(f"Collapsing {len(docs)} documents")
    try:
        result = await combine_document_func(
            {"context": docs, "user_query": user_query}
        )
        combined_metadata = {k: str(v) for k, v in docs[0].metadata.items()}
        for doc in docs[1:]:
            for k, v in doc.metadata.items():
                if k in combined_metadata:
                    combined_metadata[k] += f", {v}"
                else:
                    combined_metadata[k] = str(v)

        logger.debug("Document collapse completed successfully")
        return Document(page_content=result, metadata=combined_metadata)
    except Exception as e:
        logger.error(f"Error collapsing documents: {str(e)}")
        raise


class SummaryState(BaseModel):
    """개별 요약을 위한 state"""

    content: str | list = Field(default_factory=str)
    user_query: str = Field(default_factory=str)


class MapReduceSummarizationState(BaseModel):
    """
    contents (List[str]): 입력 문서 내용의 리스트.
    summaries (Annotated[list, operator.add]): 개별 노드들에서 생성한 모든 요약들을 하나의 리스트로 결합. (map_summaries에서 시작된 여러 generate_summary들)
    collapsed_summaries (List[Document]): 요약된 문서들의 리스트.
    final_summary (str): 최종 요약.
    """

    contents: list[str] = Field(default_factory=list)
    summaries: Annotated[list, operator.add] = Field(default_factory=list)
    collapsed_summaries: list[Document] = Field(default_factory=list)
    final_summary: str = Field(default_factory=str)
    user_query: str = Field(default_factory=str)


class MapReduceSummarizationSubgraph:
    llm: BaseLanguageModel
    graph: CompiledStateGraph
    text_splitter: TextSplitter

    def __init__(self, llm, graph, text_splitter, chunk_size):
        self.llm = llm
        self.graph = graph
        self.text_splitter = text_splitter
        self.chunk_size = chunk_size
        logger.debug(
            f"MapReduceSummarizationSubgraph initialized with chunk_size: {chunk_size}"
        )

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, chunk_size: int = 4096):
        logger.debug(
            f"Creating MapReduceSummarizationSubgraph with chunk_size: {chunk_size}"
        )

        map_prompt = SummaryPrompt().get_prompt_template()
        reduce_prompt = SummaryPrompt().get_prompt_template()

        map_chain = map_prompt | llm | StrOutputParser()
        reduce_chain = reduce_prompt | llm | StrOutputParser()

        graph = cls._create_graph(llm, map_chain, reduce_chain, chunk_size)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            length_function=lambda x: get_num_tokens(
                x, "gpt-4o-mini"
            ),  # gemini 토큰 수 측정함수는 시간이 오래 소요되어 gpt-4o-mini 토큰 수 사용
        )

        logger.info("MapReduceSummarizationSubgraph created successfully")
        return cls(
            llm=llm, graph=graph, text_splitter=text_splitter, chunk_size=chunk_size
        )

    @staticmethod
    def _create_graph(llm, map_chain, reduce_chain, chunk_size) -> CompiledStateGraph:
        logger.debug("Creating summarization graph")

        async def generate_summary(state: SummaryState):
            """주어진 문서를 요약하는 비동기 함수."""
            logger.debug("Generating individual summary")
            try:
                response = await map_chain.ainvoke(
                    {"context": state.content, "user_query": state.user_query}
                )
                logger.debug("Individual summary generated successfully")
                return {"summaries": [response]}
            except Exception as e:
                logger.error(f"Error generating individual summary: {str(e)}")
                raise

        def map_summaries(state: MapReduceSummarizationState):
            """문서들에 대해 map할 로직을 정의하는 함수."""
            logger.debug(f"Mapping {len(state.contents)} contents for summarization")
            return [
                Send(
                    "generate_summary",
                    SummaryState(content=content, user_query=state.user_query),
                )
                for content in state.contents
            ]

        def length_function(documents: list[Document]) -> int:
            """입력된 내용의 토큰 수를 가져오는 함수. 이걸로 충분히 reduce되었는지 봄"""
            return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

        def collect_summaries(state: MapReduceSummarizationState):
            """요약을 모아서 collapsed_summaries에 저장하는 함수."""
            logger.debug(f"Collecting {len(state.summaries)} summaries")
            return {
                "collapsed_summaries": [
                    Document(summary) for summary in state.summaries
                ]
            }

        async def generate_final_summary(state: MapReduceSummarizationState):
            """collapsed_summaries에 저장된 요약들을 읽어 최종 요약을 생성하는 비동기 함수."""
            logger.debug("Generating final summary")
            try:
                response = await reduce_chain.ainvoke(
                    {
                        "context": state.collapsed_summaries,
                        "user_query": state.user_query,
                    }
                )
                logger.debug("Final summary generated successfully")
                return {"final_summary": response}
            except Exception as e:
                logger.error(f"Error generating final summary: {str(e)}")
                raise

        async def collapse_summaries(state: MapReduceSummarizationState):
            """여기는 collapsed_summaries에 저장된 요약들이 제한 길이를 넘을 때 오는 node.

            collapsed_summaries에 저장된 요약들을 읽어 토큰 수에 따라 분할하고, 각 분할된 리스트를 비동기적으로 병합하여 최종 요약 리스트를 생성하는 함수.
            """
            logger.debug("Collapsing summaries due to token limit")
            try:
                doc_lists = split_list_of_docs(
                    state.collapsed_summaries, length_function, chunk_size
                )
                results = []
                for doc_list in doc_lists:
                    results.append(
                        await acollapse_docs(
                            doc_list, reduce_chain.ainvoke, user_query=state.user_query
                        )
                    )

                logger.debug(f"Collapsed to {len(results)} summary groups")
                return {"collapsed_summaries": results}
            except Exception as e:
                logger.error(f"Error collapsing summaries: {str(e)}")
                raise

        def should_collapse(
            state: MapReduceSummarizationState,
        ) -> Literal["collapse_summaries", "generate_final_summary"]:
            num_tokens = length_function(state.collapsed_summaries)
            if num_tokens > chunk_size:
                logger.debug(
                    f"Token count {num_tokens} exceeds limit {chunk_size}, collapsing"
                )
                return "collapse_summaries"
            else:
                logger.debug(
                    f"Token count {num_tokens} within limit, generating final summary"
                )
                return "generate_final_summary"

        graph_builder = StateGraph(MapReduceSummarizationState)
        nodes = [
            ("generate_summary", generate_summary),
            ("collect_summaries", collect_summaries),
            ("generate_final_summary", generate_final_summary),
            ("collapse_summaries", collapse_summaries),
        ]
        edges = [
            ("generate_summary", "collect_summaries"),
            ("generate_final_summary", END),
        ]
        edges_with_conditions = [
            (START, map_summaries, ["generate_summary"]),
            ("collect_summaries", should_collapse),
            ("collapse_summaries", should_collapse),
        ]

        graph_builder = add_graph_components(
            graph_builder, nodes, edges, edges_with_conditions
        )

        logger.debug("Summarization graph created successfully")
        return graph_builder.compile(checkpointer=False)

    async def arun(self, document: Document, user_query: str = "") -> Document:
        logger.debug(
            f"Running summarization for document (length: {len(document.page_content)})"
        )

        try:
            if document.metadata.get("type") == "pdf":
                logger.debug("Processing PDF document with Gemini")
                messages = SummaryPrompt().format_messages(
                    user_query=user_query,
                    file_type="application/pdf",
                    media_file=document.page_content,
                )
                result = await gemini_llm.ainvoke(messages)
                document = Document(
                    page_content=result.content, metadata=document.metadata
                )

            if self.llm.get_num_tokens(document.page_content) < self.chunk_size:
                logger.debug("Document is within token limit, no splitting needed")
                return document

            logger.debug(
                "Document exceeds token limit, performing map-reduce summarization"
            )
            splitted = self.text_splitter.create_documents([document.page_content])
            logger.debug(f"Split document into {len(splitted)} chunks")

            summary = await self.graph.ainvoke(
                {
                    "contents": [doc.page_content for doc in splitted],
                    "user_query": user_query,
                }
            )

            result = Document(
                page_content=summary["final_summary"], metadata=document.metadata
            )
            logger.info("Document summarization completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error during document summarization: {str(e)}")
            url = document.metadata.get("source", "unknown")
            return Document(
                page_content=f"Error occurred during document summarization. Providing snippet only: {document.page_content[:self.chunk_size]}",
                metadata={
                    "error": type(e).__name__,
                    "error_content": str(e),
                    "source": url,
                },
            )

    async def abatch(
        self, documents: list[Document], user_query: str = ""
    ) -> list[dict]:  # list[Document]
        logger.info(f"Batch summarizing {len(documents)} documents")

        async def summarize_doc(doc: Document, user_query: str = ""):
            try:
                if "error" not in doc.metadata:
                    summary = await self.arun(doc, user_query)
                    return summary.model_dump()
                else:
                    logger.warning(
                        f"Skipping document with error: {doc.metadata.get('error', 'Unknown error')}"
                    )
                    return doc.model_dump()
            except Exception as e:
                logger.error(f"Error summarizing individual document: {str(e)}")
                # Return error document instead of raising
                return Document(
                    page_content="Error occurred during summarization",
                    metadata={
                        "error": type(e).__name__,
                        "error_content": str(e),
                        "source": doc.metadata.get("source", "unknown"),
                    },
                ).model_dump()

        try:
            results = await asyncio.gather(
                *[summarize_doc(doc, user_query) for doc in documents],
                return_exceptions=True,
            )

            # Filter out any exceptions that might have slipped through
            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Exception in batch summarization: {str(result)}")
                    # Create error document
                    error_doc = Document(
                        page_content="Error occurred during batch summarization",
                        metadata={
                            "error": type(result).__name__,
                            "error_content": str(result),
                        },
                    ).model_dump()
                    valid_results.append(error_doc)
                else:
                    valid_results.append(result)

            logger.info(
                f"Batch summarization completed with {len(valid_results)} results"
            )
            return valid_results

        except Exception as e:
            logger.error(f"Critical error in batch summarization: {str(e)}")
            raise
