"""
판례 검색 기반 조언 에이전트 (스트리밍 버전)
"""
from openai import AsyncOpenAI
from config import OPENAI_API_KEY
import json, asyncio

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def advising_agent(query: str, memory_context: str, domain: str):
    prompt = f"""
    당신은 {domain} 분야의 법률 전문가입니다.
    다음은 사용자의 사건 요약 및 관련 정보입니다.

    ---
    {memory_context}
    사용자 질문: {query}
    ---

    임무:
    - 사용자의 상황에 맞는 법적 판단과 실질적인 조언을 제시하세요.
    - 관련 판례나 근거를 자연스럽게 언급하세요.
    - 너무 단정하지 말고 "일반적으로는~" 형태로 말하세요.

    출력:
    자연스러운 법률 조언문
    """

    try:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            stream=True,
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
