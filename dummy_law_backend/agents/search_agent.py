from tavily import TavilyClient, AsyncTavilyClient
from google.genai import types
import re
import json
from dotenv import load_dotenv
import os
from typing import Union
import asyncio

from .schemas import PAGE_MAIN_CONTENT, LawDocModel
from .configs import Configuration, TavilySearchConfig, TavilyExtractConfig
from .utils import make_logger, measure_time

logger = make_logger(__name__)

class FindDocLaw(object):
    r"""
    Currently not use now, can benefit for offload data for RAG
    """
    def __init__(self):
        self._general_pattern = "\[(?:Luật|Nghị định).*?\]\(.*?\)"
        self._law_code_pattern = "\[(?:Luật|Nghị định).*?\]"
        self._law_url_pattern = "\]\(.*?\)"

    def find(self, page_content: str)->Union[None, list[LawDocModel]]:
        res: list[str] = re.findall(self._general_pattern , page_content)
        if len(res) == 0:
            logger.info('early return in find')
            return None
        else:
            found_docs = {}
            for single_res in res:
                lawcode_seach = re.search(self._law_code_pattern, single_res)
                lawurl_seach = re.search(self._law_url_pattern, single_res)

                if lawcode_seach is not None and lawurl_seach is not None:
                    found_docs[lawcode_seach.group(0)[1:-1]] = lawurl_seach.group(0)[2:-1]
                else:
                    continue
            
            if len(found_docs) == 0:
                # incase the link is broken
                logger.info('link may broken')
                return None
            else:
                return [
                    LawDocModel(doc_number=k, url= v)
                    for k, v in found_docs.items()
                ]


_PAGE_TOP_PATTERN = r"\[Đăng nhập bằng FaceBook\]\(http.*?png\)|\[Avatar\]\(http.*?png\)|Tham vấn bởi Luật sư|\[Mục lục bài viết\]|Đang tải văn bản|Trang chủ"
_PAGE_BOTTOM_PATTERN = r"\[Hỏi đáp pháp luật\]\(\)\n|Chủ đề đang được đánh giá|vui lòng gửi về Email |=>>Xem thêm|!\[Thư viện nhà đất\]\(\)|!\[Pháp luật\]\(\)|\[\]\(.*?png\)|HỎI ĐÁP PHÁP LUẬT LIÊN QUAN|Nội dung bài viết chỉ mang tính chất tham khảo| Bạn hãy nhập e-mail đã sử dụng để đăng ký thành viên|Trân trọng|Quý khách cần hỏi thêm thông tin về  có thể đặt câu hỏi tại đây.|Đặt câu hỏi"
_CLEAN_PATTERN=r"\[\]\(http.*?png\)"

async def _text_trimming(extract_result_element: dict[str,str])->dict[str,str]:
    url = extract_result_element['url']
    
    text = extract_result_element['raw_content']
    res = re.search(_PAGE_TOP_PATTERN, text)
    try:
        start_idx = res.span()[-1]
    except Exception as e:
        logger.error('error in text_timming start_idx: {}, url: {}'.format(e, url))
        start_idx = 0
        return None

    res = re.search(_PAGE_BOTTOM_PATTERN, text, re.IGNORECASE)
    try:
        end_idx = res.span()[0]
    except Exception as e:
        logger.error("error in text_timming end_idx: {}, url: {}\m------- END LOG".format(e, url))
        end_idx = len(text)-1
        return None

    cleaned_text = re.sub(_CLEAN_PATTERN, '',text[start_idx: end_idx])
    extract_result_element['raw_content'] = cleaned_text
    return extract_result_element

async def _post_tavily_extract_processing(extract_results: list[dict[str,str]]):
    return await asyncio.gather(*[
        _text_trimming(ele)
        for ele in extract_results['results']
    ])

class SearchAgent(object):

    def __init__(self, config: Configuration):
        load_dotenv()
        if os.getenv("TAVILY_API_KEY") is None:
            raise ValueError("TAVILY_API_KEY is not set")

        self.config = config
        self.tavily_client = AsyncTavilyClient(api_key= os.getenv("TAVILY_API_KEY"))
        self._search_kwargs = TavilySearchConfig().model_dump()
        self._extract_kwargs = TavilyExtractConfig().model_dump()

    @measure_time(logger = logger)
    async def run(self, query:str):
        # fast search
        search_results = await self.tavily_client.search(
            query = query,
            **self._search_kwargs,
        )

        # extract list of urls
        extract_results = await self.tavily_client.extract(
            urls=[
                res['url'] 
                for res in search_results['results']
            ],
            **self._extract_kwargs
        )

        # trimming using regex
        trim_results = await _post_tavily_extract_processing(extract_results)

        trim_results = [ele for ele in trim_results if ele is not None]

        return [
            PAGE_MAIN_CONTENT(
                web_page_number= ith,
                main_content= ele['raw_content'],
                url= ele['url']
            )
            for ith, ele in 
            enumerate(trim_results)
        ]
    