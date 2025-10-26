"""
조언을 바탕으로 실질적 실행 단계를 제시하는 에이전트 (스트리밍 버전)
"""
from openai import AsyncOpenAI
from config import OPENAI_API_KEY
import json, asyncio

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def guidance_agent(advice_text: str, memory_context: str = "", domain: str = "일반"):
    prompt = f"""
    당신은 {domain} 분야의 전문 법률 조력자입니다.

    지금까지의 상담:
    {memory_context}

    직전 법률 조언:
    {advice_text}

    임무:
    - 사용자가 실제로 취할 수 있는 구체적인 행동 단계를 안내하세요.
    - 필요한 서류, 기관, 주의사항을 단계별로 명확히 정리하세요.
    - 마무리에 "궁금하신 부분이 있나요?"로 끝내세요.
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
