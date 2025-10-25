from fastapi import APIRouter, Request
from models.models import AIChatRequest, AIChatResponse
from services.memory_manager import MemoryManager
from services.free_chat import free_chat_agent
from services.info_gathering import info_gathering_agent
from services.advising import advising_agent
from services.guidance import guidance_agent
from services.mode_classifier import mode_classifier
import uuid, requests

router = APIRouter(prefix="/AIChat", tags=["AIChat"])
memory = MemoryManager(max_turns=15)
SPRING_API_URL = "http://localhost:8087/hellaw/api/AIChat"


@router.post("/", response_model=AIChatResponse)
async def chat_pipeline(req: AIChatRequest, request: Request):
    """
    전체 대화 파이프라인.

    free_chat : 자유롭게 법률 관련 질의응답을 주고 받을 수 있는 에이전트.
    info_gathering : 사용자가 사례를 제시하면, 필요한 정보를 수집.
    advising : RAG 기반 판례 조언 제공.
    guidance : 조언에 따른 구체적 실행 방안 안내.

    순서:
    free_chat → info_gathering → advising → guidance → free_chat
    """
    user_query = req.question
    domain = req.domain

    # conv_idx 설정
    memory.conv_idx = req.conv_idx if req.conv_idx else str(uuid.uuid4())

    # 사용자 발화 추가
    memory.add("user", user_query)
    context = memory.get_context()

    # LLM 기반 초기 모드 판단
    classification = await mode_classifier(user_query, context, domain)
    current_mode = classification.get("next_mode", "free_chat")
    reason = classification.get("reason", "")
    memory.set_mode(current_mode)
    print(f"[Mode → {current_mode}] 이유: {reason}")

    # 모드별 에이전트 실행
    if current_mode == "free_chat":
        agent_result = await free_chat_agent(user_query, context, domain)
        agent_message = agent_result.get("message", "")
        next_mode = agent_result.get("next_state", "free_chat")

    elif current_mode == "info_gathering":
        agent_result = await info_gathering_agent(user_query, domain, context)
        agent_message = agent_result.get("message", "")
        next_mode = agent_result.get("next_state", "info_gathering")

        # ready_for_advice가 True → 자동으로 조언(advising) 및 안내(guidance) 실행
        if agent_result.get("ready_for_advice", False):
            print("정보 수집 완료 → 조언 단계로 전환합니다.")
            
            # (1) 판례 기반 조언
            advising_result = await advising_agent(user_query, context, domain)
            advice_message = advising_result.get("advice", "조언 생성 실패")

            # (2) guidance 단계 실행
            guidance_result = await guidance_agent(advice_message, context, domain)
            guidance_message = guidance_result.get("message", "후속 안내 생성 실패")

            # (3) 메시지 통합
            agent_message = f"{advice_message}\n {guidance_message}"

            # (4) 다음 모드는 free_chat으로 복귀
            next_mode = "free_chat"

    # 메모리 및 모드 업데이트
    memory.add("assistant", agent_message)
    memory.set_mode(next_mode)
    print(f"모드 전환 완료: {current_mode} → {next_mode}")

    # Spring DB 저장용 payload 구성
    payload = {
        "conv_idx": memory.conv_idx,
        "question": user_query,
        "answer": agent_message
    }

    # JWT 전달 설정
    headers = {}
    auth_header = request.headers.get("authorization")
    if auth_header:
        headers["Authorization"] = auth_header

    # Spring DB 저장
    try: # TODO: 비동기 통신으로 고치기
        res = requests.post(
            f"{SPRING_API_URL}/save",
            json=payload,
            headers=headers,
            timeout=5
        )
        if res.status_code != 200:
            print(f"⚠️ Spring DB 저장 실패: {res.text}")
    except Exception as e:
        print(f"Spring 연결 오류: {e}")

    # 사용자에게 최종 응답 반환
    return AIChatResponse(
        role="assistant",
        content={"message": agent_message},
        current_mode=next_mode,  # guidance까지 끝났으면 free_chat 반환
        conv_idx=memory.conv_idx
    )
