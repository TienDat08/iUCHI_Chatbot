from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class ClassificationResult(BaseModel):
    """
    Represents the result of classifying a user's question.
    """

    is_legal_question: bool = Field(
        description="Whether the question is related to law, notarization, or authentication."
    )
    reason: str = Field(description="A brief explanation for the classification.")
