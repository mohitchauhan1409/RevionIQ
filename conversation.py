from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    role: str  # "assistant" | "user"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    is_network_insight: bool = False


class ConversationHistory:
    def __init__(self):
        self._messages: list[Message] = []

    def add(self, role: str, content: str, is_network_insight: bool = False):
        self._messages.append(
            Message(role=role, content=content, is_network_insight=is_network_insight)
        )

    def get_all(self) -> list[dict]:
        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "is_network_insight": m.is_network_insight,
            }
            for m in self._messages
        ]

    def get_segment(self, from_index: int = 0) -> list[dict]:
        return self.get_all()[from_index:]

    def __len__(self) -> int:
        return len(self._messages)
