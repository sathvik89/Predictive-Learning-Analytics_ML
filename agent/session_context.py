"""
Limits LangChain message history length for each graph invoke.
keeps only last 20 msgs
needs a recent window so context stays relevant without huge payloads.
"""

from __future__ import annotations

from typing import List

from langchain_core.messages import BaseMessage

MAX_LANGCHAIN_MESSAGES: int = 20


def trim_messages(messages: List[BaseMessage], max_n: int | None = None) -> List[BaseMessage]:
    if not messages:
        return []
    cap = max_n if max_n is not None else MAX_LANGCHAIN_MESSAGES
    if cap <= 0 or len(messages) <= cap:
        return list(messages)
    return list(messages[-cap:])
