"""
자유롭게 법률 관련 질의응답을 주고 받을 수 있는 에이전트.
기본(default) 모드에서 동작하며, 일반적인 법률 개념·절차·용어 질문에 대응함.
"""

from openai import OpenAI
from config import OPENAI_API_KEY
import asyncio

client = OpenAI(api_key=OPENAI_API_KEY)

async def free_chat_agent(user_query: str, memory_context: str, domain: str):
    """
    자유 질의응답용 에이전트.
    단순한 법률 정보, 용어, 절차, 판례 개념 등에 대한 설명을 제공합니다.
    """
    prompt = f"""
    당신은 {domain} 분야를 포함한 다양한 법률 지식을 가진 전문 상담사입니다.
    사용자의 질문에 대해 법률 용어나 절차를 이해하기 쉽게 설명해주세요.

    - 너무 형식적인 문체는 피하고, 상담하듯 자연스럽게 답변하세요.
    - 필요시 참고할 만한 법 조항을 인용하되, 쉬운 용어 설명을 덧붙이세요.
    - 사용자에게 도움이 될만한 구체적 행동 팁을 덧붙이면 좋습니다.
    - 지나치게 단정하지 말고, “일반적으로는 ~” 같은 표현을 사용하세요.

    ---
    지금까지의 대화:
    {memory_context}

    사용자의 질문:
    "{user_query}"
    ---

    출력(JSON):
    {{
      "message": "자연스러운 법률 정보 또는 설명"
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
        # 모델이 JSON 형식으로 반환할 경우 안전하게 파싱
        return eval(res.choices[0].message.content)
    except Exception:
        # JSON 실패 시 일반 텍스트 그대로 반환
        return {"message": res.choices[0].message.content.strip()}
