from google import genai
from google.genai import types
import re
import json
from dotenv import load_dotenv
import os
import asyncio
from typing import Union


from .schemas import PAGE_MAIN_CONTENT, ReasoningResult
from .prompts import FINAL_ANSWER_SYSTEM_PROMPT, FINAL_ANSWER_PROMPT
from .configs import Configuration
from .utils import make_logger, measure_time


logger = make_logger(__name__)                

class FinalAnswerAgent(object):

    _answer_config = types.GenerateContentConfig(
        system_instruction=FINAL_ANSWER_SYSTEM_PROMPT,
        response_schema=ReasoningResult,
        response_mime_type='application/json',
        temperature = 1.0,
        thinking_config=types.ThinkingConfig(
            thinking_budget=2048,
            include_thoughts=False
            )
    )

    _basemodel2string = """
- kết quả tìm kiếm thông tin liên quan: {ith}
- nguồn: {url}
- nội dung bài viết: {raw_content}
"""

    def __init__(self, config: Configuration):
        load_dotenv()
        if os.getenv("GEMINI_API_KEY") is None:
            raise ValueError("GEMINI_API_KEY is not set")

        self.config = config
        self.answer_model = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    @measure_time(logger = logger)
    async def run(self, query:str, search_agent_outputs: list[PAGE_MAIN_CONTENT])->str:
        
        infors = '\n'.join([
            self._basemodel2string.format(
                ith = ith,
                url = output.url,
                raw_content = output.main_content
            )
            for ith, output in
            enumerate(search_agent_outputs)
        ])
        
        response = await self.answer_model.aio.models.generate_content(
            model = self.config.answer_model,
            contents = FINAL_ANSWER_PROMPT.format(
                query = query,
                infors = infors
            ),
            config = self._answer_config,
        )
        if response.parsed is not None:
            return response.parsed.tostring
        else:
            logger.info("response.parsed is not None")
            return response.text
