"""
사용자의 상황을 파악하고 필요한 정보를 수집하는 에이전트 (진짜 스트리밍 버전)
"""
from openai import AsyncOpenAI
from config import OPENAI_API_KEY
import json, asyncio

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def info_gathering_agent(query: str, domain: str, memory_context: str = ""):
    """
    사용자의 상황을 분석하고 필요한 정보를 수집하며,
    한 글자씩 실시간으로 전송합니다.
    """
    checklist = ["상황 설명", "원인", "결과"]

    prompt = f"""
    당신은 {domain} 분야의 전문 법률 상담사입니다.  
    사용자의 발화를 근거로 사건 파악을 위해 필요한 질문을 던지세요.
    공감하는 어투로 자연스럽게 이어가세요.

    ---
    지금까지의 대화:
    {memory_context}

    새 사용자 발화:
    "{query}"
    ---
    """

    try:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            stream=True
        )

        accumulated = ""

        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta.strip():
                accumulated += delta
                # ✅ 프론트로 토큰 단위 전송
                yield f"data: {json.dumps({'token': delta}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

        # ✅ 스트림 완료 후 전체 메시지 추가 전송 (optional)
        yield f"data: {json.dumps({'full': accumulated}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        err = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"data: {err}\n\n"
