from collections import defaultdict, deque
import uuid

class MemoryManager:
    def __init__(self, max_turns=15):
        """
        여러 세션(conv_idx)별로 대화 히스토리와 모드를 관리하는 매니저.
        """
        self.max_turns = max_turns
        self.sessions = defaultdict(lambda: {
            "history": deque(maxlen=max_turns),
            "mode": "free_chat"
        })

    def ensure_session(self, conv_idx: str):
        """세션이 존재하지 않으면 새로 생성."""
        if conv_idx not in self.sessions:
            self.sessions[conv_idx] = {
                "history": deque(maxlen=self.max_turns),
                "mode": "free_chat"
            }

    def add(self, conv_idx: str, role: str, content: str):
        """특정 세션(conv_idx)의 대화 추가."""
        self.ensure_session(conv_idx)
        self.sessions[conv_idx]["history"].append({"role": role, "content": content})

    def get_context(self, conv_idx: str):
        """특정 세션(conv_idx)의 대화 이력을 문자열로 반환."""
        self.ensure_session(conv_idx)
        hist = self.sessions[conv_idx]["history"]
        return "\n".join(
            [f"{h['role'].upper()}: {h['content']}" for h in hist]
        )

    def reset(self, conv_idx: str):
        """특정 세션(conv_idx) 초기화."""
        if conv_idx in self.sessions:
            del self.sessions[conv_idx]

    def get_mode(self, conv_idx: str):
        """특정 세션(conv_idx)의 현재 모드 반환."""
        self.ensure_session(conv_idx)
        return self.sessions[conv_idx]["mode"]

    def set_mode(self, conv_idx: str, new_mode: str):
        """특정 세션(conv_idx)의 모드 설정."""
        self.ensure_session(conv_idx)
        self.sessions[conv_idx]["mode"] = new_mode

    def get_all_sessions(self):
        """디버깅용: 현재 존재하는 세션 목록 반환."""
        return list(self.sessions.keys())
