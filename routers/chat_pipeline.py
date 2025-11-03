from fastapi import APIRouter, Request
from models.models import ChatRequest
from services.memory_manager import MemoryManager
from services.mode_classifier import mode_classifier
from services.chat_agent import (
    free_chat_agent,
    info_gathering_agent,
    advising_agent,
    guidance_agent
)
import os, uuid, json
import pymysql
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI


from config import (
    HELLAW_DB_HOST,
    HELLAW_DB_USER,
    HELLAW_DB_PASSWORD,
    HELLAW_DB_NAME
)

router = APIRouter(prefix="/AIChat", tags=["AIChat"])
memory = MemoryManager()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SPRING_API_URL = "http://localhost:8087/hellaw/api/AIChat"

def get_chat_history_from_db(conv_idx: str):
    conn = pymysql.connect(
        host=HELLAW_DB_HOST,
        user=HELLAW_DB_USER,
        password=HELLAW_DB_PASSWORD,
        database=HELLAW_DB_NAME,
        port=3307,
        cursorclass=pymysql.cursors.DictCursor,
    )
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT question, answer
                FROM tb_ai_chat
                WHERE conv_idx = %s
                ORDER BY created_at ASC
            """,
            (conv_idx,),
            )
            return cursor.fetchall()

def db_has_conv(conv_idx):
    conn = pymysql.connect(
        host=HELLAW_DB_HOST,
        user=HELLAW_DB_USER,
        password=HELLAW_DB_PASSWORD,
        database=HELLAW_DB_NAME,
        port=3307,
        cursorclass=pymysql.cursors.DictCursor
    )
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM tb_ai_chat WHERE conv_idx = %s LIMIT 1", (conv_idx,))
            return cursor.fetchone() is not None

def restore_memory_from_db(conv_idx: str, db_records):
    memory_context = memory.get_memory(conv_idx)
    for row in db_records:
        if row.get("question"):
            memory_context.chat_memory.add_message(
                {
                    "role": "user",
                    "content": row["question"]
                }
            )
        if row.get("answer"):
            memory_context.chat_memory.add_message(
                {
                    "role": "ai",
                    "content": row["answer"]
                }
            )


@router.post("/stream", response_class=StreamingResponse)
async def chat_pipeline(request: ChatRequest):

    # 변수 설정
    conv_idx = request.conv_idx or f"stream_{uuid.uuid4()}"
    domain = request.domain
    query = request.query

    print(f"SSE 요청 수신...")
    print(f"query : {query}")
    print(f"domain : {domain}")
    print(f"conv_idx : {conv_idx}")

    memory_context = memory.get_memory(conv_idx)

    if db_has_conv(conv_idx) and len(memory_context.chat_memory.messages) == 0:
        db_records = get_chat_history_from_db(conv_idx)
        restore_memory_from_db(conv_idx, db_records)
        print(f"[{conv_idx}] DB 기반 메모리 복원 완료 ({len(db_records)}개 메시지)")
    elif not db_has_conv(conv_idx):
        print(f"[{conv_idx}] 신규 대화 시작")
    else:
        print(f"[{conv_idx}] 기존 세션, DB 복원 생략")

    memory_context = memory.get_memory(conv_idx)
    current_mode = memory.get_mode(conv_idx)

    print(f"[세션 : {conv_idx}] 현재 모드 : {current_mode}")

    # 메모리 등록
    memory.add(conv_idx, "user", query)
    print(f"사용자 발화 등록 완료, 메모리 메시지 수: {len(memory_context.chat_memory.messages)}")

    # free_chat일 때 mode 판단
    if current_mode == "free_chat":
        print("모드 분류 중...")
        mode_classification = await mode_classifier(query, memory_context, domain)
        next_mode = mode_classification.get("next_mode", "free_chat")
        reason = mode_classification.get("reason", "")
        memory.set_mode(conv_idx, next_mode)
        print(f"모드 전환 : {current_mode} -> {next_mode} 이유: {reason}")
        current_mode = next_mode
    else:
        print(f"모드 유지 : {current_mode}")
    
    # SSE 이벤트 스트림
    async def event_stream():
        yield f"data: {{\"conv_idx\": \"{conv_idx}\"}}\n\n"
        print(f"[{conv_idx}] 스트리밍 시작 (모드: {current_mode})")

        try:
            if current_mode == "info_gathering":
                # 한 번의 요청에서는 정보수집 1회만 스트리밍하고 종료
                rounds = memory.get_info_rounds(conv_idx)
                next_round = rounds + 1
                print(f"[INFO_GATHERING] 스트리밍 라운드 {next_round}/2 (요청 단위)")

                # 새 말풍선 시작 신호 (항상 별도 말풍선)
                yield "data: {\"new_message\": true}\n\n"
                async for chunk in info_gathering_agent(query, domain, memory_context):
                    print(f"[INFO_GATHERING] 토큰: {chunk[:100]}")
                    yield chunk

                memory.increment_info_rounds(conv_idx)
                if next_round >= 2:
                    # 다음 요청에서 조언 단계가 시작되도록 모드 전환만 미리 설정
                    memory.set_mode(conv_idx, "advising")
                    print("[STATE] 정보수집 2회 완료 → 다음 요청에서 조언 단계로 전환 예정")
                else:
                    print("[STATE] 정보수집 1회 완료 → 다음 요청에서 2차 정보수집 진행")
                # 현재 요청은 여기서 종료 ([DONE]은 info_gathering_agent가 보냄)
                return


            elif current_mode == "advising":
                async for chunk in advising_agent(query, domain, memory_context):
                    print(f"[ADVISING] 토큰: {chunk[:100]}")
                    yield chunk
                # 조언 완료 후, 다음 라운드를 위해 info_rounds 초기화 및 모드 free_chat 유지/복귀
                memory.reset_info_rounds(conv_idx)

            elif current_mode == "guidance":
                async for chunk in guidance_agent(query, domain, memory_context):
                    print(f"[GUIDANCE] 토큰: {chunk[:100]}")
                    yield chunk
                memory.set_mode(conv_idx, "free_chat")
                print(f"[{conv_idx}] guidance 종료 → free_chat 모드로 복귀")

            else:  # free_chat
                async for chunk in free_chat_agent(query, domain, memory_context):
                    print(f"[FREE_CHAT] 토큰: {chunk[:100]}")
                    yield chunk
    
        except Exception as e:
            print(f"스트림 처리 중 예외 발생 : {type(e).__name__} - {e}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        
        print(f"[{conv_idx}] 스트리밍 완료")
        yield "data: [DONE]\n\n"

    # FastAPI SSE 응답
    return StreamingResponse(event_stream(), media_type="text/event-stream")