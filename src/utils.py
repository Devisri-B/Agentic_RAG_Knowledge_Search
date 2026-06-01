"""
Pure helper functions with no heavy (model) imports, so they can be unit-tested
quickly in CI without downloading embedding/NLI models.
"""

import re

# Source markers emitted by the retrievers, e.g. "[File: resume.pdf]" / "[Source: Page 3]"
_CITATION_RE = re.compile(r"\[((?:File|Source)[^\]]*)\]")
_RETRY_RE = re.compile(r"retryDelay.*?(\d+\.?\d*)s")

_INVALID_KEY_MARKERS = (
    "API_KEY_INVALID",
    "API key not valid",
    "PERMISSION_DENIED",
    "API key expired",
)


def extract_content(message) -> str:
    """Gemini may return message content as a list of typed blocks; flatten to text."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        content = " ".join(
            block["text"] if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") == "text"
        )
    return str(content)


def parse_tool_results(messages: list) -> tuple[str, str]:
    """Return (source_type, combined_tool_output) from the agent message chain.

    source_type is 'rag', 'web', 'rag+web', or 'unknown'. The combined output is
    the actual text the tools returned, which faithfulness is measured against.
    """
    rag_parts, web_parts = [], []
    for msg in messages:
        name = getattr(msg, "name", None)
        content = getattr(msg, "content", "") or ""
        if name == "lookup_documents":
            rag_parts.append(content)
        elif name == "search_web":
            web_parts.append(content)

    if rag_parts and web_parts:
        return "rag+web", " ".join(rag_parts + web_parts)
    if rag_parts:
        return "rag", " ".join(rag_parts)
    if web_parts:
        return "web", " ".join(web_parts)
    return "unknown", ""


def extract_citations(tool_output: str) -> list[str]:
    """Pull unique source labels (file names / page numbers) from retrieved text."""
    seen, out = set(), []
    for label in _CITATION_RE.findall(tool_output or ""):
        label = label.strip()
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return out


def is_invalid_key(error_str: str) -> bool:
    return any(m in error_str for m in _INVALID_KEY_MARKERS)


def retry_delay(error_str: str) -> float:
    """Parse 'retryDelay: Xs' from a Gemini 429 error string; 0.0 if absent."""
    match = _RETRY_RE.search(error_str)
    return float(match.group(1)) if match else 0.0
