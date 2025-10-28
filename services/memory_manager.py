from collections import defaultdict
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class MemoryManager:
    def __init__(self):
        """세션별 LangChain 메모리 및 모드 관리"""
        self.sessions = defaultdict(lambda: {
            "memory": ConversationBufferMemory(memory_key="history", return_messages=True),
            "mode": "free_chat"
        })

    def ensure_session(self, conv_idx):
        """세션이 없으면 새로 만들어주는 함수"""
        if conv_idx not in self.sessions:
            self.sessions[conv_idx] = {
                "memory": ConversationBufferMemory(memory_key="history", return_messages=True),
                "mode": "free_chat"
            }

    def get_memory(self, conv_idx):
        """특정 세션의 LangChain Memory 반환"""
        self.ensure_session(conv_idx)
        return self.sessions[conv_idx]["memory"]

    def get_mode(self, conv_idx):
        """특정 세션의 conv_idx 반환"""
        self.ensure_session(conv_idx)
        return self.sessions[conv_idx]["mode"]

    def set_mode(self, conv_idx, mode):
        """특정 세션의 모드 변경"""
        self.ensure_session(conv_idx)
        self.sessions[conv_idx]["mode"] = mode

    def add(self, conv_idx, role, content):
        """대화를 LangChain Memory에 추가"""
        self.ensure_session(conv_idx)
        memory = self.sessions[conv_idx]["memory"]

        # 역할에 따라 올바른 메시지 객체로 변환
        if role == "user":
            msg = HumanMessage(content=content)
        elif role == "ai":
            msg = AIMessage(content=content)
        elif role == "system":
            msg = SystemMessage(content=content)
        else:
            raise ValueError(f"Unknown role: {role}")

        memory.chat_memory.add_message(msg)
