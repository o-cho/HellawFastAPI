from typing import Optional
from pydantic import BaseModel

# TODO 각 구조가 어디에서 쓰이는지 확인
class AIChatRequest(BaseModel):
    """Spring → FastAPI 요청 데이터 구조"""
    query: str
    domain: str
    conv_idx: Optional[str] = None

class AIChatResponse(BaseModel):
    """FastAPI → Spring 응답 데이터 구조"""
    role: str
    content: dict
    conv_idx: str

class AIChatData(BaseModel):
    """FastAPI → Spring 응답 데이터 구조 (DB 저장용)"""
    conv_idx: str
    query: str
    answer: str

class ChatRequest(BaseModel):
    """FastAPI chat_pipeline endpoint 요청 데이터 구조"""
    query: str
    domain: str
    conv_idx: str = None