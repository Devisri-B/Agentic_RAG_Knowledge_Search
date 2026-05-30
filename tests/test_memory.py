"""Unit tests for conversation memory + summarization (no API calls)."""

from src.memory import ConversationMemory, SessionStore


def fake_summarizer(text: str) -> str:
    return "SUMMARY"


class TestConversationMemory:
    def test_add_and_build_messages(self):
        m = ConversationMemory()
        m.add_turn("hello", "hi there")
        msgs = m.build_messages("follow up")
        assert msgs == [
            ("user", "hello"),
            ("assistant", "hi there"),
            ("user", "follow up"),
        ]

    def test_summary_prepended_as_system(self):
        m = ConversationMemory(summary="earlier stuff")
        roles = [r for r, _ in m.build_messages("q")]
        assert roles[0] == "system"
        assert roles[-1] == "user"

    def test_needs_summary_threshold(self):
        m = ConversationMemory(max_turns=3, keep_recent=2)
        for i in range(3):
            m.add_turn(f"q{i}", f"a{i}")
        assert m.needs_summary() is False
        m.add_turn("q3", "a3")
        assert m.needs_summary() is True

    def test_summarize_folds_and_trims(self):
        m = ConversationMemory(max_turns=3, keep_recent=2)
        for i in range(4):
            m.add_turn(f"q{i}", f"a{i}")

        ran = m.summarize_if_needed(fake_summarizer)
        assert ran is True
        assert m.summary == "SUMMARY"
        # only the last keep_recent turns remain
        assert m.turns == [("q2", "a2"), ("q3", "a3")]

    def test_summarize_noop_when_under_limit(self):
        m = ConversationMemory(max_turns=5, keep_recent=2)
        m.add_turn("q", "a")
        assert m.summarize_if_needed(fake_summarizer) is False
        assert m.summary == ""

    def test_summary_accumulates_prior(self):
        captured = {}

        def capturing_summarizer(text: str) -> str:
            captured["text"] = text
            return "NEW SUMMARY"

        m = ConversationMemory(max_turns=2, keep_recent=1, summary="OLD SUMMARY")
        for i in range(3):
            m.add_turn(f"q{i}", f"a{i}")
        m.summarize_if_needed(capturing_summarizer)
        assert "OLD SUMMARY" in captured["text"]


class TestSessionStore:
    def test_sessions_are_isolated(self):
        store = SessionStore()
        store.get("a").add_turn("x", "y")
        assert len(store.get("a").turns) == 1
        assert len(store.get("b").turns) == 0

    def test_clear_removes_session(self):
        store = SessionStore()
        store.get("a").add_turn("x", "y")
        store.clear("a")
        assert len(store.get("a").turns) == 0
