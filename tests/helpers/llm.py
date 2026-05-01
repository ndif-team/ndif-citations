"""Mock LLM client that duck-types the OpenAI SDK surface used by classify_category."""
from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock


class _MockCompletion:
    """Minimal duck-type of openai.types.chat.ChatCompletion."""
    def __init__(self, content: str):
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        self.choices = [choice]


class MockLLMClient:
    """Deterministic LLM mock for classify_category / generate_summary tests."""

    def __init__(self):
        self._replies: deque[str] = deque()
        self._calls: list[list[dict]] = []
        self.chat = self

    # Fluent API: openai_client.chat.completions.create(...)
    @property
    def completions(self):
        return self

    def create(self, *, model=None, messages=None, temperature=None, max_tokens=None, **kwargs):
        self._calls.append(messages or [])
        if self._replies:
            reply = self._replies.popleft()
        else:
            reply = "unclassified"  # safe default when no expectation set
        return _MockCompletion(reply)

    def expect(self, reply: str) -> "MockLLMClient":
        """Queue a canned reply for the next create() call. Returns self for chaining."""
        self._replies.append(reply)
        return self

    def record_calls(self) -> list[list[dict]]:
        """Return all messages lists passed to create()."""
        return list(self._calls)

    def assert_no_calls(self) -> None:
        assert not self._calls, f"Expected no LLM calls, but got {len(self._calls)}"
