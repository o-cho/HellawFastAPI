"""
자유롭게 법률 관련 질의응답을 주고 받을 수 있는 에이전트 (스트리밍 버전)
"""
from openai import AsyncOpenAI
from config import OPENAI_API_KEY
import json, asyncio

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def free_chat_agent(user_query: str, memory_context: str, domain: str):
    prompt = f"""
    당신은 {domain} 분야를 포함한 다양한 법률 지식을 가진 전문 상담사입니다.
    사용자의 질문에 대해 법률 용어나 절차를 이해하기 쉽게 설명해주세요.

    - 너무 형식적인 문체는 피하고, 상담하듯 자연스럽게 답변하세요.
    - 필요시 참고할 만한 법 조항을 인용하되, 쉬운 용어 설명을 덧붙이세요.
    - 지나치게 단정하지 말고 “일반적으로는 ~” 같은 표현을 사용하세요.

    --- 지금까지의 대화 ---
    {memory_context}

    사용자의 질문:
    "{user_query}"

    출력(JSON):
    {{
      "message": "자연스러운 법률 정보 또는 설명"
    }}
    """

    try:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            stream=True
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta.strip():
                yield f"data: {json.dumps({'token': delta}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

        yield "data: [DONE]\n\n"

    except Exception as e:
        err = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"data: {err}\n\n"
