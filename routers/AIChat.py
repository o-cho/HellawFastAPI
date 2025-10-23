from fastapi import APIRouter, Request
from models.models import AIChatRequest, AIChatResponse
from services.dialogue import MemoryManager, dialogue_agent
from services.advisor import advisor
import uuid, requests

router = APIRouter(prefix="/AIChat", tags=["AIChat"])
memory = MemoryManager(max_turns=10)
SPRING_API_URL = "http://localhost:8087/hellaw/api/AIChat"

@router.post("/", response_model=AIChatResponse)
async def chat(req: AIChatRequest, request: Request):
    user_query = req.question
    domain = req.domain

    # conv_idx 설정
    if req.conv_idx:
        memory.conv_idx = req.conv_idx
    else:
        memory.conv_idx = str(uuid.uuid4())

    # 사용자 발화 추가
    memory.add("user", user_query)
    context = memory.get_context()

    # Dialogue Agent 실행
    agent_result = await dialogue_agent(user_query, context, domain)
    agent_message = agent_result.get("message", "")
    ready_for_advice = agent_result.get("ready_for_advice", False)

    # 메모리에 모델 응답 추가
    memory.add("assistant", agent_message)

    # Spring DB 저장용 payload 구성
    payload = {
        "conv_idx": memory.conv_idx,
        "question": user_query,
        "answer": agent_message  # message만 저장
    }

    print(f"Agent result: {agent_result}")

    # JWT 전달 설정
    headers = {}
    auth_header = request.headers.get("authorization")
    if auth_header:
        headers["Authorization"] = auth_header

    # Spring DB 저장 시도
    try:
        res = requests.post(f"{SPRING_API_URL}/save", json=payload, headers=headers, timeout=5)
        if res.status_code != 200:
            print(f"Spring DB 저장 실패 : {res.text}")
    except Exception as e:
        print(f"Spring 연결 오류 : {e}")

    # ready_for_advice가 True면 advisor 실행
    if ready_for_advice:
        advice_result = await advisor(user_query, memory.get_context(), domain)
        final_message = advice_result.get("advice", "조언 생성 실패")
        memory.add("assistant", final_message)

        # advisor 결과 반환
        return AIChatResponse(
            role="assistant",
            content=final_message,
            conv_idx=memory.conv_idx,
        )

    # 기본 응답 (단순 상담)
    return AIChatResponse(
        role="assistant",
        content=agent_message,
        conv_idx=memory.conv_idx
    )
