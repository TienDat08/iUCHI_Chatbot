from google import genai
from google.genai import types
import re
import json
from dotenv import load_dotenv
import os


from .schemas import ClassificationResult
from .prompts import INTENT_PROMPT
from .configs import Configuration
from .utils import make_logger, measure_time

logger = make_logger(__name__)

class IntentAgent(object):

    _intent_config = types.GenerateContentConfig(
        response_schema=ClassificationResult,
        response_mime_type='application/json',
        temperature = 0.4
    )

    _non_legal_msg_response = """
Tôi xin lỗi, tôi chỉ có thể trả lời các câu hỏi liên quan đến luật, công chứng và chứng thực. Vui lòng đặt một câu hỏi khác.
"""

    def __init__(self, config: Configuration):
        load_dotenv()
        if os.getenv("GEMINI_API_KEY") is None:
            raise ValueError("GEMINI_API_KEY is not set")

        self.config = config
        
        self.intent_clf_model = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    @measure_time(logger = logger)
    async def run(self, query:str)->bool:
        response = await self.intent_clf_model.aio.models.generate_content(
            model = self.config.intent_model,
            contents = INTENT_PROMPT.format(query = query),
            config = self._intent_config,
        )

        if response.parsed.is_legal_question:
            return True, None
        else:
            return False, self._non_legal_msg_response
