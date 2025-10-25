"""
조언을 바탕으로 사용자가 실제로 실천할 수 있는 실행 방안을 제시하는 에이전트.
"""

from openai import OpenAI
from config import OPENAI_API_KEY
import json
import asyncio

client = OpenAI(api_key=OPENAI_API_KEY)

async def guidance_agent(advice_text: str, memory_context: str = "", domain: str = "일반"):
    prompt = f"""
    당신은 {domain} 분야의 전문 법률 조력자입니다.

    지금까지의 상담 맥락:
    {memory_context}

    ---
    직전 법률 조언:
    {advice_text}
    ---

    임무:
    1. 위 조언을 바탕으로 사용자가 실제로 취할 수 있는 구체적인 '행동 단계'를 안내하세요.
       - 행동: 예) 내용증명 발송, 증거 수집, 소송 제기 등
       - 준비: 필요한 서류, 증거, 기관 등
       - 주의: 법적 시한, 비용, 실수 방지 팁 등
    2. 단계별로 친절하게 제시하고, 법률 용어를 쉽게 풀어서 설명하세요.
    3. 사용자의 후속 질문 가능성을 고려해 “궁금하신 부분이 있나요?”로 마무리하세요.
    4. 출력(JSON):
    {{
      "message": "실행 가능한 단계별 안내문",
      "next_state": "guidance" 또는 "free_chat"
    }}
    """

    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",  # 더 빠른 모델
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
    )


    try:
        content = res.choices[0].message.content.strip()
        return json.loads(content)
    except Exception:
        return {"message": content, "next_state": "guidance"}
