from fastapi import APIRouter, Request
from models.models import AIChatRequest, AIChatResponse
from services.memory_manager import MemoryManager
from services.free_chat import free_chat_agent
from services.info_gathering import info_gathering_agent
from services.advising import advising_agent
from services.guidance import guidance_agent
from services.mode_classifier import mode_classifier
import uuid, requests, os, json, asyncio
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

router = APIRouter(prefix="/AIChat", tags=["AIChat"])
memory = MemoryManager(max_turns=15)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SPRING_API_URL = "http://localhost:8087/hellaw/api/AIChat"
memory_sessions = {}

@router.get("/stream")
async def stream_chat_pipeline(question: str, domain: str, conv_idx: str = None):
    """
    ✅ 통합 스트리밍 파이프라인 (안정화 버전)
    free_chat → info_gathering → advising → guidance → free_chat
    """
    # 세션별 메모리 가져오기 (없으면 새로 생성)
    if conv_idx not in memory_sessions:
        memory_sessions[conv_idx] = MemoryManager(max_turns=15)
        print(f"🆕 새 MemoryManager 생성: conv_idx={conv_idx}")
    memory = memory_sessions[conv_idx]

    context = memory.get_context(conv_idx)
    current_mode = memory.get_mode(conv_idx)

    print(f"🚀 [STREAM START] conv_idx={conv_idx}, domain={domain}, question={question}")

    # === 1️⃣ 초기 모드 분류 ===
    if current_mode == "free_chat" and len(context.strip()) == 0:
        classification = await mode_classifier(question, context, domain)
        current_mode = classification.get("next_mode", "free_chat")
        reason = classification.get("reason", "")
        memory.set_mode(conv_idx, current_mode)
        print(f"[Mode → {current_mode}] 이유: {reason}")
    else:
        print(f"[Mode 유지] conv_idx={conv_idx}, 현재 모드: {current_mode}")

    # === 2️⃣ 스트림 생성 ===
    async def event_stream():
        accumulated = ""

        try:
            # ✅ 안전한 JSON 파서 (오류 방지용)
            def safe_parse(chunk: str):
                try:
                    if chunk.startswith("data: "):
                        return json.loads(chunk[len("data: "):].strip())
                    else:
                        return json.loads(chunk.strip())
                except Exception:
                    return {}

            # === free_chat 단계 ===
            if current_mode == "free_chat":
                async for chunk in free_chat_agent(question, context, domain):
                    yield chunk
                    data = safe_parse(chunk)
                    accumulated += data.get("token", "")
                next_mode = "info_gathering"

            # === info_gathering 단계 ===
            elif current_mode == "info_gathering":
                async for chunk in info_gathering_agent(question, domain, context):
                    yield chunk
                    data = safe_parse(chunk)
                    accumulated += data.get("token", "")
                next_mode = "info_gathering"

                # ✅ ready_for_advice 감지
                if "ready_for_advice" in accumulated.lower() and "true" in accumulated.lower():
                    print("정보 수집 완료 → 조언 단계로 전환합니다.")
                    async for chunk in advising_agent(question, context, domain):
                        yield chunk
                        data = safe_parse(chunk)
                        accumulated += data.get("token", "")

                    async for chunk in guidance_agent(accumulated, context, domain):
                        yield chunk
                        data = safe_parse(chunk)
                        accumulated += data.get("token", "")

                    next_mode = "free_chat"

            else:
                # fallback
                async for chunk in free_chat_agent(question, context, domain):
                    yield chunk
                    data = safe_parse(chunk)
                    accumulated += data.get("token", "")
                next_mode = "free_chat"

            # === 3️⃣ 메모리 및 Spring DB 저장 ===
            memory.add(conv_idx, "user", question)
            memory.add(conv_idx, "assistant", accumulated)
            memory.set_mode(conv_idx, next_mode)

            payload = {
                "conv_idx": conv_idx,
                "question": question,
                "answer": accumulated
            }

            try:
                requests.post(f"{SPRING_API_URL}/save", json=payload, timeout=5)
            except Exception as e:
                print(f"⚠️ Spring 저장 실패: {e}")

            print(f"✅ 스트리밍 완료: {conv_idx}")
            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"❌ 스트림 오류: {e}")
            err_json = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {err_json}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )




@router.post("/", response_model=AIChatResponse)
async def chat_pipeline(req: AIChatRequest, request: Request):
    """
    전체 대화 파이프라인 (다중 세션 지원)
    free_chat → info_gathering → advising → guidance → free_chat
    """
    user_query = req.question
    domain = req.domain
    conv_idx = req.conv_idx or str(uuid.uuid4())  # ✅ 새 세션이면 자동 생성

    # 세션별 컨텍스트 및 모드 가져오기
    context = memory.get_context(conv_idx)
    current_mode = memory.get_mode(conv_idx)

    # 새로운 세션일 경우만 LLM 분류 실행
    if current_mode == "free_chat" and len(context.strip()) == 0:
        classification = await mode_classifier(user_query, context, domain)
        current_mode = classification.get("next_mode", "free_chat")
        reason = classification.get("reason", "")
        memory.set_mode(conv_idx, current_mode)
        print(f"[Mode → {current_mode}] 이유: {reason}")
    else:
        print(f"[Mode 유지] conv_idx={conv_idx}, 현재 모드: {current_mode}")

    # === 모드별 에이전트 실행 ===
    if current_mode == "free_chat":
        agent_result = await free_chat_agent(user_query, context, domain)
        agent_message = agent_result.get("message", "")
        next_mode = agent_result.get("next_state", "free_chat")

    elif current_mode == "info_gathering":
        agent_result = await info_gathering_agent(user_query, domain, context)
        agent_message = agent_result.get("message", "")
        next_mode = agent_result.get("next_state", "info_gathering")

        if agent_result.get("ready_for_advice", False):
            print("정보 수집 완료 → 조언 단계로 전환합니다.")

            advising_result = await advising_agent(user_query, context, domain)
            advice_message = advising_result.get("advice", "조언 생성 실패")

            guidance_result = await guidance_agent(advice_message, context, domain)
            guidance_message = guidance_result.get("message", "후속 안내 생성 실패")

            agent_message = f"{agent_message}\n\n📘 조언: {advice_message}\n💡 가이드: {guidance_message}"
            next_mode = "free_chat"

    # === 메모리 업데이트 ===
    memory.add(conv_idx, "user", user_query)
    memory.add(conv_idx, "assistant", agent_message)
    memory.set_mode(conv_idx, next_mode)
    print(f"모드 전환 완료: {current_mode} → {next_mode} (conv_idx={conv_idx})")

    # === Spring DB 저장 ===
    payload = {
        "conv_idx": conv_idx,
        "question": user_query,
        "answer": agent_message
    }
    headers = {}

    auth_header = request.headers.get("authorization")

    if auth_header:
        auth_header = auth_header.replace('"', '').strip()

        for prefix in ["bearer:", "bearer ", "access_token:", "access_token "]:
            if auth_header.lower().startswith(prefix):
                auth_header = auth_header[len(prefix):].strip()
                break

        headers["Authorization"] = f"Bearer {auth_header}"

        print(f"[DEBUG] 정제 후 Authorization 헤더: {headers['Authorization']}")



    try:
        res = requests.post(f"{SPRING_API_URL}/save", json=payload, headers=headers, timeout=5)
        if res.status_code != 200:
            print(f"Spring DB 저장 실패: {res.text}")
    except Exception as e:
        print(f"Spring 연결 오류: {e}")

    # === 최종 응답 ===
    return AIChatResponse(
        role="assistant",
        content={"message": agent_message},
        current_mode=next_mode,
        conv_idx=conv_idx
    )

