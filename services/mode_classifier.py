from openai import OpenAI
from config import OPENAI_API_KEY
import asyncio, json

client = OpenAI(api_key=OPENAI_API_KEY)

async def mode_classifier(user_query: str, memory_context: str, domain: str):
    """
    LLM 기반 대화 단계 분류기.
    """
    prompt = f"""
    당신은 법률 상담 대화의 단계 판별을 담당하는 역할입니다.

    아래 대화 기록을 보고, 현재 사용자가 어떤 단계에 해당하는지 판단하세요.

    ---
    지금까지의 대화:
    {memory_context}

    사용자 입력: "{user_query}"
    분야: {domain}
    ---

    가능한 단계:
    최근 상담의 맥락을 참고하여, 사용자의 이번 발화가 다음 중 어떤 의도에 가까운지 판단하세요:
    1. 사용자의 실제 사례를 이야기함 → info_gathering
    2. 더 구체적인 조언(판례 검색 등)을 원함 → advising
    3. 구체적인 실행 방법을 물음 ("어떻게 해야", "무엇을 하면") → guidance
    4. 조언을 거절하거나 대화를 마무리하려는 뉘앙스 ("아니요", "ㄴㄴ", "됐어요", "그만") → free_chat
    5. 그 외의 일반적인 법률 관련 질문 → free_chat

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
        content = res.choices[0].message.content.strip()
        content = content[content.find("{") : content.rfind("}") + 1]
        return json.loads(content)
    except Exception as e:
        print(f"[ModeClassifier ParseError] {e}")
        return {"next_mode": "free_chat", "reason": "판단 실패"}