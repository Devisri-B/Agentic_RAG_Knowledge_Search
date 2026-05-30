"""Unit tests for pure helpers (no model loading, fast in CI)."""

from types import SimpleNamespace
from src.utils import (
    extract_content,
    parse_tool_results,
    extract_citations,
    is_invalid_key,
    retry_delay,
)


def _msg(content, name=None):
    return SimpleNamespace(content=content, name=name)


class TestExtractContent:
    def test_plain_string(self):
        assert extract_content(_msg("hello")) == "hello"

    def test_list_of_text_blocks(self):
        msg = _msg([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
        assert extract_content(msg) == "a b"

    def test_list_skips_non_text_blocks(self):
        msg = _msg([{"type": "text", "text": "keep"}, {"type": "image", "data": "x"}])
        assert extract_content(msg) == "keep"


class TestParseToolResults:
    def test_rag_only(self):
        msgs = [_msg("[Source: Page 1] law text", name="lookup_documents")]
        source, output = parse_tool_results(msgs)
        assert source == "rag"
        assert "law text" in output

    def test_web_only(self):
        msgs = [_msg("breaking news", name="search_web")]
        assert parse_tool_results(msgs)[0] == "web"

    def test_both_tools(self):
        msgs = [_msg("doc", name="lookup_documents"), _msg("web", name="search_web")]
        assert parse_tool_results(msgs)[0] == "rag+web"

    def test_no_tools(self):
        assert parse_tool_results([_msg("just an answer")]) == ("unknown", "")


class TestExtractCitations:
    def test_file_and_page_labels(self):
        text = "[File: resume.pdf] foo\n\n[Source: Page 3] bar"
        assert extract_citations(text) == ["File: resume.pdf", "Source: Page 3"]

    def test_deduplicates_preserving_order(self):
        text = "[File: a.pdf] x [File: a.pdf] y [File: b.pdf] z"
        assert extract_citations(text) == ["File: a.pdf", "File: b.pdf"]

    def test_empty(self):
        assert extract_citations("") == []


class TestKeyAndRetryHelpers:
    def test_invalid_key_detected(self):
        assert is_invalid_key("400 API_KEY_INVALID ...") is True
        assert is_invalid_key("PERMISSION_DENIED") is True

    def test_valid_key_not_flagged(self):
        assert is_invalid_key("RESOURCE_EXHAUSTED quota") is False

    def test_retry_delay_parsed(self):
        assert retry_delay("please retry. retryDelay: 20s") == 20.0

    def test_retry_delay_absent(self):
        assert retry_delay("some other error") == 0.0
