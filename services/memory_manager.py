from collections import defaultdict
from langchain.memory import ConversationBufferMemory

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
        """대화를 메모리에 추가"""
        self.ensure_session(conv_idx)
        memory = self.sessions[conv_idx]["memory"]
        memory.chat_memory.add_message({"role": role, "content": content})
