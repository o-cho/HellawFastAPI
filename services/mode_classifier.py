from openai import OpenAI
from config import OPENAI_API_KEY
import asyncio

client = OpenAI(api_key=OPENAI_API_KEY)

async def mode_classifier(user_query: str, memory_context: str, domain: str):
    """
    LLM 기반 대화 단계 분류기.
    """
    prompt = f"""
    당신은 법률 상담 대화의 단계 판별을 담당하는 에이전트입니다.

    아래 대화 기록을 보고, 현재 사용자가 어떤 단계에 해당하는지 판단하세요.

    ---
    지금까지의 대화:
    {memory_context}

    사용자 입력: "{user_query}"
    분야: {domain}
    ---

    가능한 단계:
    - "free_chat": 일반 법률 질의나 정보 탐색
    - "info_gathering": 사용자가 실제 사건/사례를 이야기하고 있으며, 후속 질문을 통해 상황 파악이 필요한 단계
    - "advising": 실제 사건/사례에 대한 정보가 충분하여 법적 판단이나 판례 조언이 필요한 단계
    - "guidance": 법적 판단이 끝나고, 실질적 행동/절차 안내가 필요한 단계

    ---
    출력(JSON):
    {{
      "next_mode": "free_chat" | "info_gathering" | "advising" | "guidance",
      "reason": "판단 근거 간략히"
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
        return eval(res.choices[0].message.content)
    except Exception:
        return {"next_mode": "free_chat", "reason": "판단 실패"}
