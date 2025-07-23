from typing import Optional

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.tools import BaseTool

from estalan.logging_config import get_logger
from estalan.tools.utils import noop

logger = get_logger(__name__)


class AsyncTool(BaseTool):
    def _run(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError


class ConcatTool(AsyncTool):
    tools: list[AsyncTool]

    @classmethod
    def from_tools(cls, tools: list[AsyncTool]):
        logger.debug(f"Creating ConcatTool from tools: {[tool.name for tool in tools]}")
        main_tool = tools[0]
        tool_attributes = {
            "name": main_tool.name,
            "description": main_tool.description,
            "args_schema": main_tool.args_schema,
        }

        concat_tool = cls(**tool_attributes, tools=tools)
        logger.info(
            f"ConcatTool created: {main_tool.name} with tools: {[tool.name for tool in tools]}"
        )
        return concat_tool

    async def _arun(
        self,
        verbose: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs,
    ) -> list[dict]:
        logger.debug(
            f"Executing ConcatTool with tools: {[tool.name for tool in self.tools]}"
        )
        dispatcher = adispatch_custom_event if verbose else noop

        try:
            results = sum(
                [
                    await tool.ainvoke(
                        tool.args_schema.model_validate(
                            {**kwargs, "verbose": verbose}
                        ).model_dump(),
                        run_manager=run_manager,
                    )
                    for tool in self.tools
                ],
                start=[],
            )

            if self.name == "search_news" and results:
                await dispatcher(
                    "event", {"speak": f"검색 결과 {len(results)}개를 읽고 있어요."}
                )

            logger.debug(
                f"ConcatTool executed successfully, returned {len(results)} results"
            )
            return results

        except Exception as e:
            logger.error(f"Error executing ConcatTool: {str(e)}")
            raise
