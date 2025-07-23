import os
from pydantic import BaseModel, Field
from typing import Any, Optional

class TavilyConfigBase(BaseModel):
    include_images: bool = Field(
        default=False, 
        description="no need images"
    )

    include_favicon: bool = Field(
        default=False, 
        description="no need favicon"
    )

class TavilySearchConfig(TavilyConfigBase):
    
    max_results: int = Field(
        default = 5, 
        description="""
            Maximum results for searching, the number of calling filtering_model is 
            less than or equal this value since for one query search .extract may fail at some url
        """
    )

    auto_parameters: bool = Field(
        default=False, 
        description="dont set auto"
    )

    search_depth: str = Field(
        default="basic", 
        description="'basic' for saving tokens"
    )
    include_answer: bool = Field(
        default=False, 
        description="no need answer of Tavily'llm"
    )
    include_raw_content: bool = Field(
        default=False, 
        description="set to Fasle to increase search speed"
    )

    include_domains: list[str] = Field(
        default=[
            "luatvietnam.vn",
            "https://thuvienphapluat.vn"
        ], 
        description="dedicated domains to lookup"
    )

    exclude_domains: list[str] = Field(
        default=[
            # "luatvietnam.vn",
            "https://dantri.vn"
        ], 
        description="dedicated domains to lookup"
    )

    country: str = Field(
        default="vietnam", 
        description="for vietname search region only"
    )

class TavilyExtractConfig(TavilyConfigBase):

    extract_depth: str = Field(
        default="basic", 
        description="'basic' for saving tokens"
    )

    format:str = Field(
        default = "markdown",
        description="Available options: markdown, text, plain text and may increase latency"
    )

class Configuration(BaseModel):
    """The configuration for the agent."""

    intent_model: str = Field(
        default="gemini-2.5-flash-lite-preview-06-17",
        metadata={
            "description": "The name of the language model to use for intent classification"
        },
    )

    answer_model: str = Field(
        default="gemini-2.5-flash-lite-preview-06-17",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    max_research_loops: int = Field(
        default=1,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    