"""
Shared guardrail response text for the AI Study Coach.

Scope and integrity classification is handled by the master node's structured
LLM plan, so this module does not route requests with keyword lists.
"""

from __future__ import annotations


def academic_scope_message() -> str:
    return (
        "I can only help with academic learning: explaining concepts, building study plans, "
        "reviewing student performance, and creating practice quizzes. Please ask me something "
        "related to your studies, and I will help with that."
    )


def cheating_redirect_message() -> str:
    return (
        "I cannot help with cheating or submitting work as your own. I can still help you "
        "learn the topic, outline an approach, check your reasoning, or create practice "
        "questions so you can complete the work honestly."
    )
