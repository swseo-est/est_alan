## Using Dataclasses
from dataclasses import dataclass, field
from typing import Annotated
from estalan.prebuilt.requirement_analysis_agent.prompt import PROMPT_REQUIREMENT_ANALYSIS

@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=PROMPT_REQUIREMENT_ANALYSIS,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent.",
            "json_schema_extra": {"langgraph_nodes": ["react_agent"]},
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="azure_openai/gpt-4o",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name.",
            "json_schema_extra": {"langgraph_nodes": ["react_agent"]},
        },
    )

