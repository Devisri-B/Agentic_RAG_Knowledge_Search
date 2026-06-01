"""Per-session conversation memory. Recent turns are kept verbatim; older turns are
folded into a running summary so long chats stay within the model's context window."""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ConversationMemory:
    max_turns: int = 6       # summarize once the window exceeds this many turns
    keep_recent: int = 4     # turns kept verbatim after summarizing
    summary: str = ""
    turns: list[tuple[str, str]] = field(default_factory=list)  # (user, assistant)

    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append((user, assistant))

    def needs_summary(self) -> bool:
        return len(self.turns) > self.max_turns

    def build_messages(self, new_query: str) -> list[tuple[str, str]]:
        """Compose the message list to send to the agent for this turn."""
        messages: list[tuple[str, str]] = []
        if self.summary:
            messages.append(("system", f"Summary of the earlier conversation:\n{self.summary}"))
        for user, assistant in self.turns:
            messages.append(("user", user))
            messages.append(("assistant", assistant))
        messages.append(("user", new_query))
        return messages

    def summarize_if_needed(self, summarizer: Callable[[str], str]) -> bool:
        """If over the limit, fold older turns into the summary. Returns True if it ran."""
        if not self.needs_summary():
            return False

        overflow = self.turns[: -self.keep_recent] if self.keep_recent else self.turns
        if not overflow:
            return False

        transcript = self._render(overflow)
        prior = f"Previous summary:\n{self.summary}\n\n" if self.summary else ""
        self.summary = summarizer(f"{prior}New exchanges to fold in:\n{transcript}").strip()
        self.turns = self.turns[-self.keep_recent :] if self.keep_recent else []
        return True

    @staticmethod
    def _render(turns: list[tuple[str, str]]) -> str:
        return "\n".join(f"User: {u}\nAssistant: {a}" for u, a in turns)


class SessionStore:
    """In-memory map of session id -> ConversationMemory."""

    def __init__(self, max_turns: int = 6, keep_recent: int = 4):
        self._sessions: dict[str, ConversationMemory] = {}
        self._max_turns = max_turns
        self._keep_recent = keep_recent

    def get(self, session_id: str) -> ConversationMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(
                max_turns=self._max_turns, keep_recent=self._keep_recent
            )
        return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
