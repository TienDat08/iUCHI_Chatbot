from agents.search_agent import SearchAgent
from agents.reasoning_agent import FinalAnswerAgent
from agents.intent_agent import IntentAgent
from agents.configs import Configuration
import asyncio
import json


main_config = Configuration()

intent_clf_agent = IntentAgent(config = main_config)
search_agent = SearchAgent(config = main_config)
final_agent = FinalAnswerAgent(config = main_config)


async def main():
    query = "luật công chứng 2024 có hiệu lực khi nào?"
    # query = "UBND xã được chứng thực loại giấy tờ nào?"
    # query= "Thủ tục hồ sơ công chứng chuyển nhượng QSD đất như thế nào?"
    # query="giấy “Xác nhận tình trạng hôn nhân” do Đại sứ quán cấp có đúng thẩm quyền ko?"
    # query="Ông A và bà B kết hôn với nhau nhưng không có con chung, có tài sản chung là 1 ngôi nhà. Bà B chết trước, khi bà B chết không còn cha mẹ anh chị em ruột, chỉ có 1 người cháu gọi bằng cô, khj ông A chết không còn bất kỳ ai ở cả 3 hàng thừa kế. Hỏi việc phân chia di sản thừa kế trên được thực hiện như thế nào."
    # query ="Quân nhân tại ngũ lập di chúc nhưng không thể yêu cầu công chứng chứng thực thì có giá trị khi nào?"


    result = await intent_clf_agent.run(query = query)
    if result:
        search_result = await search_agent.run(query = query)
        final_result = await final_agent.run(query = query,search_agent_outputs=search_result)
        print(final_result)

if __name__ == "__main__":
    asyncio.run(main())

