from pydantic import BaseModel, Field, HttpUrl, computed_field

class LawDocModel(BaseModel):
    doc_number: str
    url: str
    content: str = Field(
        default=None,
        description="Content of the law doc"
    )

class PAGE_MAIN_CONTENT(BaseModel):
    r"""
    Schema for main content of the web page
    """
    web_page_number: int = Field(
        description="the order of web page input"
    )

    main_content: str = Field(
        description="main content output after filtering"
    )

    url: str = Field(
        description="the url of the web page"
    )

    law_docs: None | list[LawDocModel] = Field(
        default=None,
        frozen=False,
        description="per page can contains some original law documents"
    )

class ClassificationResult(BaseModel):
    """
    Represents the result of classifying a user's question.
    """

    is_legal_question: bool = Field(
        description="Whether the question is related to law, notarization, or authentication."
    )
    reason: str = Field(description="A brief explanation for the classification.")

class ReasoningResult(BaseModel):
    """
    Represents the result of reasoning to legal question with support of web search results
    """

    summary: str = Field(
        description="Trả lời trực tiếp và ngắn gọn câu hỏi (3-5 gạch đầu dòng)."
    )
    detail: str = Field(description="Phân tích sâu hơn, giải thích các quy định liên quan.")

    legal_basis: str = Field(description="Liệt kê các văn bản pháp luật đã sử dụng (ví dụ: 'Điều 52, Luật Nhà ở 2023').")

    notes: str = Field(description="Nêu các điểm rủi ro, các bước tiếp theo hoặc lời khuyên hữu ích")

    @computed_field
    @property
    def tostring(self)->str:
        return f"""
1.  **TÓM TẮT:** {self.summary}
2.  **GIẢI THÍCH CHI TIẾT:** {self.detail}
3.  **CĂN CỨ PHÁP LÝ:** {self.legal_basis}
4.  **LƯU Ý:** {self.notes}
"""

