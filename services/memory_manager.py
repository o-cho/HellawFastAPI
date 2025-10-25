from collections import deque
import uuid

# 메모리
class MemoryManager:
    def __init__(self, max_turns=15):
        """초기화 메서드."""
        self.history = deque(maxlen=max_turns) # 턴 초과 시 자동으로 오래된 대화 삭제
        self.conv_idx = str(uuid.uuid4()) # 세션 고유 아이디 생성
        self.mode = "free_chat" # 기본 모드
    
    def add(self, role, content):
        """대화 한 턴을 메모리에 추가함."""
        self.history.append({"role": role, "content": content})
    
    def get_context(self):
        """저장된 대화 내용을 개행으로 연결하여 하나의 문자열로 반환함."""
        return "\n".join(
            [f"{h['role'].upper()}: {h['content']}" for h in self.history]
        )
    
    def reset(self):
        """새 대화 시작. 기존 기록을 초기화하고 conv_id를 재발급함."""
        self.history.clear()
        self.conv_idx = str(uuid.uuid4())
        self.mode = "free_chat"

    def set_mode(self, new_mode:str):
        """현재 대화 모드 변경."""
        self.mode = new_mode
    
    def get_mode(self):
        """현재 대화 모드 확인."""
        return self.mode
