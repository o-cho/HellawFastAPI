from fastapi import APIRouter, Request
from models.models import ChatRequest
from services.memory_manager import MemoryManager
from services.free_chat import free_chat_agent
from services.info_gathering import info_gathering_agent
from services.advising import advising_agent
from services.guidance import guidance_agent
from services.mode_classifier import mode_classifier
import os, uuid
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import pymysql
import json

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

def get_chat_history_from_db(conv_idx):
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
            cursor.execute("""
                SELECT question, answer
                FROM tb_ai_chat
                WHERE conv_idx = %s
                ORDER BY created_at ASC
            """, (conv_idx,))
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

def restore_memory_from_db(conv_idx, db_records):
    memory_context = memory.get_memory(conv_idx)
    for row in db_records:
        if row.get("question"):
            memory_context.chat_memory.add_message({
                "role": "user",
                "content": row["question"]
            })
        if row.get("answer"):
            memory_context.chat_memory.add_message({
                "role": "ai",
                "content": row["answer"]
            })


@router.post("/stream", response_class=StreamingResponse)
async def chat_pipeline(request: ChatRequest):

    # 변수 설정
    conv_idx = request.conv_idx or f"stream_{uuid.uuid4()}"
    domain = request.domain
    query = request.query

    if db_has_conv(conv_idx):
        db_records = get_chat_history_from_db(conv_idx)
        restore_memory_from_db(conv_idx, db_records)
        print(f"[{conv_idx}] DB 기반 메모리 복원 완료 ({len(db_records)}개 메시지)")
    else:
        print(f"[{conv_idx}] 신규 대화 시작")


    memory_context = memory.get_memory(conv_idx)
    current_mode = memory.get_mode(conv_idx)

    print(f"[세션 : {conv_idx}] 현재 모드 : {current_mode}")

    # 메모리 등록
    memory.add(conv_idx, "user", query)

    # free_chat일 때 mode 판단
    if current_mode == "free_chat":
        mode_classification = await mode_classifier(query, memory_context, domain)
        next_mode = mode_classification.get("next_mode", "free_chat")
        reason = mode_classification.get("reason", "")
        memory.set_mode(conv_idx, next_mode)
        print(f"모드 전환 : {current_mode} -> {next_mode} 이유: {reason}")
        current_mode = next_mode
    
    async def event_stream():
        yield f"data: {{\"conv_idx\": \"{conv_idx}\"}}\n\n"

        if current_mode == "info_gathering":
            async for chunk in info_gathering_agent(query, domain, memory_context):
                yield chunk

        elif current_mode == "advising":
            async for chunk in advising_agent(query, domain):
                yield chunk

        elif current_mode == "guidance":
            async for chunk in guidance_agent(query, domain, memory_context):
                yield chunk

        else:  # free_chat
            async for chunk in free_chat_agent(memory_context, query, domain):
                yield chunk

    # FastAPI SSE 응답
    return StreamingResponse(event_stream(), media_type="text/event-stream")