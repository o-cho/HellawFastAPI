# 요청(Request)과 응답(Response)을 명확하게 정의하기 위해 사용.

from pydantic import BaseModel
from typing import Optional

# Spring → FastAPI 요청 데이터 구조
class AIChatRequest(BaseModel):
    query: str
    domain: str
    conv_idx: Optional[str] = None

# FastAPI → Spring 응답 데이터 구조
class AIChatResponse(BaseModel):
    role: str
    content: dict
    conv_idx: str

# FastAPI → Spring 응답 데이터 구조 (DB 저장용)
class AIChatData(BaseModel):
    conv_idx: str
    query: str
    answer: str

class ChatRequest(BaseModel):
    query: str
    domain: str
    conv_idx: str = None