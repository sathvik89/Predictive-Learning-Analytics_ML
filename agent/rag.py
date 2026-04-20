"""
rag.py – FAISS-backed retrieval over an internal academic knowledge base.
Build the index once at import; expose `retrieve(query, k)`.
"""

import numpy as np
import faiss
import json
import os
import re
import urllib.error
import urllib.request
from typing import TypedDict
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = [
    # ── ALGEBRA ──
    "Algebra - Linear Equations: A linear equation is an equation of the form ax + b = 0. "
    "To solve it, isolate the variable by performing inverse operations (addition, subtraction, "
    "multiplication, division) on both sides equally. Example: 2x + 4 = 0 → 2x = -4 → x = -2. "
    "Key idea: maintain balance on both sides.",

    "Algebra - Quadratic Equations: A quadratic equation has the form ax² + bx + c = 0. "
    "It can be solved using factorization, completing the square, or the quadratic formula: "
    "x = (-b ± √(b² - 4ac)) / (2a). The discriminant (b² - 4ac) determines the nature of roots "
    "(real, equal, or complex).",

    "Algebra - Inequalities: Inequalities are similar to equations but involve signs like >, <, ≥, ≤. "
    "When multiplying or dividing both sides by a negative number, the inequality sign must be reversed.",

    # ── ARITHMETIC ──
    "Arithmetic - Percentages: Percentage represents parts per hundred. "
    "Formula: percentage = (part/whole) × 100. Used in profit-loss, discounts, and data interpretation.",

    "Arithmetic - Ratios & Proportions: A ratio compares two quantities, while a proportion states "
    "equality of two ratios. Example: a/b = c/d. Used in scaling, mixtures, and real-life comparisons.",

    # ── GEOMETRY ──
    "Geometry - Triangles: The sum of interior angles of a triangle is 180°. "
    "In a right triangle, Pythagoras' theorem applies: a² + b² = c². Used to find missing sides.",

    "Geometry - Circles: Key elements include radius, diameter, circumference (2πr), and area (πr²). "
    "Understanding relationships between these is essential for solving problems.",

    # ── TRIGONOMETRY ──
    "Trigonometry - Basics: Trigonometric ratios (sin, cos, tan) relate angles to sides of a right triangle. "
    "sin = opposite/hypotenuse, cos = adjacent/hypotenuse, tan = opposite/adjacent.",

    # ── STATISTICS & PROBABILITY ──
    "Statistics - Central Tendency: Mean is average, median is middle value, mode is most frequent value. "
    "Used to summarise datasets.",

    "Probability: Probability measures the likelihood of an event. "
    "Formula: P = favorable outcomes / total outcomes. Value lies between 0 and 1.",

    # ── FUNCTIONS & CALCULUS ──
    "Functions: A function maps an input to exactly one output. "
    "Graphs help visualise relationships between variables.",

    "Calculus - Derivatives: A derivative represents rate of change (slope of a curve). "
    "Example: speed is the derivative of position with respect to time.",

    "Calculus - Integrals: Integrals represent accumulation or area under a curve. "
    "They are the reverse (antiderivative) of derivatives.",

    # ── READING & COMPREHENSION ──
    "Reading Comprehension: Focus on identifying main ideas, supporting details, tone, and inference. "
    "Always connect ideas across paragraphs.",

    "Inference Skills: Inference means understanding what is implied, not directly stated. "
    "Requires combining clues from the text with prior knowledge.",

    # ── WRITING ──
    "Writing - Structure: Use the PEEL method — Point, Evidence, Explanation, Link — "
    "to construct clear and logical paragraphs.",

    # ── PROBLEM SOLVING ──
    "Problem Solving Strategy: First understand the problem, identify knowns and unknowns, "
    "choose a method, solve step-by-step, and verify the answer.",

    "Error Analysis: Reviewing mistakes helps identify conceptual gaps and prevents repeating errors.",

    # ── STUDY SCIENCE ──
    "Active Recall: Testing yourself without notes strengthens memory more effectively than passive review.",

    "Spaced Repetition: Revisiting topics at increasing intervals improves long-term retention.",

    # ── RESOURCES ──
    "For Mathematics learning, Khan Academy (khanacademy.org) offers structured courses from basics to advanced.",

    "For conceptual understanding, 3Blue1Brown provides visual and intuitive explanations of math topics.",

    "For advanced topics, MIT OpenCourseWare offers free university-level lectures.",

    "For structured courses, Coursera allows auditing many courses for free.",

    # ── META LEARNING ──
    "If you cannot explain a concept simply, you have not fully understood it. "
    "Use the Feynman Technique: explain it as if teaching a child.",

    "Learning builds layer by layer; weak fundamentals lead to difficulty in advanced topics. "
    "Always solidify basics before moving forward.",
]

_BASE_DIR = os.path.dirname(__file__)
_KNOWLEDGE_DIR = os.path.abspath(os.path.join(_BASE_DIR, "..", "knowledge"))


def _chunk_text(text: str, source: str, max_chars: int = 900) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(f"{source}: {current}")
            current = para
    if current:
        chunks.append(f"{source}: {current}")
    return chunks


def _load_curated_docs() -> list[str]:
    """Load optional project-owned academic docs from /knowledge."""
    if not os.path.isdir(_KNOWLEDGE_DIR):
        return []

    docs: list[str] = []
    for filename in sorted(os.listdir(_KNOWLEDGE_DIR)):
        if filename.lower() == "readme.md":
            continue
        if not filename.lower().endswith((".md", ".txt")):
            continue
        path = os.path.join(_KNOWLEDGE_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except OSError as e:
            print(f"[rag] WARNING – Could not read {filename}: {e}")
            continue
        if text:
            source = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()
            docs.extend(_chunk_text(text, source))
    return docs


DOCUMENTS = KNOWLEDGE_BASE + _load_curated_docs()

# ─────────────────────────────────────────────
# BUILD INDEX AT IMPORT TIME
# ─────────────────────────────────────────────
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
_embeddings      = _embedding_model.encode(DOCUMENTS, show_progress_bar=False)
_dimension       = _embeddings.shape[1]
#using faiss as vector DB. 
_index = faiss.IndexFlatL2(_dimension)
_index.add(np.array(_embeddings, dtype="float32"))

_doc_store = {i: text for i, text in enumerate(DOCUMENTS)}


class RetrievalResult(TypedDict):
    docs: list[str]
    source_type: str
    sources: list[str]
    web_attempted: bool


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
    "does", "for", "from", "give", "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "should", "tell", "the", "this", "to", "what",
    "when", "where", "which", "who", "why", "with", "you", "explain",
    "about", "please", "student", "students",
}


def _terms(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]+", (text or "").lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def _source_name(doc: str) -> str:
    if doc.startswith("Web Search - "):
        prefix = doc.split("):", 1)[0] if "):" in doc else doc.split(":", 1)[0]
        title = prefix.replace("Web Search - ", "").strip()
        if " (" in title:
            title = title.split(" (", 1)[0].strip()
        return title
    if ":" not in doc:
        return "Internal Knowledge Base"
    return doc.split(":", 1)[0].strip() or "Internal Knowledge Base"


def _local_relevance(query: str, doc: str) -> float:
    q_terms = _terms(query)
    if not q_terms:
        return 0.0
    d_terms = _terms(doc)
    overlap = len(q_terms & d_terms)
    return overlap / max(len(q_terms), 1)


def _search_local(query: str, k: int = 3) -> list[str]:
    query_vec = _embedding_model.encode([query], show_progress_bar=False)
    _, indices = _index.search(np.array(query_vec, dtype="float32"), min(k * 3, len(_doc_store)))
    candidates = [_doc_store[i] for i in indices[0] if i in _doc_store]

    relevant = [doc for doc in candidates if _local_relevance(query, doc) > 0]
    if relevant:
        return relevant[:k]

    # If lexical matching finds nothing, do not return unrelated semantic
    # neighbours. This lets the caller fall back to web search or a polite gap.
    return []

#web search tool
def _web_search(query: str, k: int = 3) -> list[str]:
    """
    Optional Tavily-backed academic web search.

    Disabled unless:
      ACADEMIC_WEB_SEARCH=true
      TAVILY_API_KEY=<key>
    """
    if not web_search_enabled():
        return []

    api_key = os.getenv("TAVILY_API_KEY", "")
    payload = json.dumps({
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": k,
        "include_answer": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[rag] Web search unavailable: {e}")
        return []

    docs: list[str] = []
    for item in data.get("results", [])[:k]:
        title = item.get("title", "Web result").strip()
        url = item.get("url", "").strip()
        content = item.get("content", "").strip()
        if content:
            docs.append(f"Web Search - {title} ({url}): {content}")
    return docs


def web_search_enabled() -> bool:
    return os.getenv("ACADEMIC_WEB_SEARCH", "").lower() == "true" and bool(os.getenv("TAVILY_API_KEY", ""))


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def retrieve(query: str, k: int = 3, include_web: bool = False) -> list[str]:
    """Return relevant curated chunks, optionally supplemented by web search."""
    result = retrieve_academic_context(query, k=k, allow_web=include_web)
    return result["docs"]


def retrieve_academic_context(query: str, k: int = 3, allow_web: bool = True) -> RetrievalResult:
    """
    Local-first retrieval with web fallback.

    source_type:
      - local: local curated knowledge covered the query
      - web: local knowledge was weak; web results supplied context
      - none: no usable local or web context
    """
    local_docs = _search_local(query, k=k)
    if local_docs:
        return {
            "docs": local_docs,
            "source_type": "local",
            "sources": sorted({_source_name(doc) for doc in local_docs}),
            "web_attempted": False,
        }

    should_try_web = allow_web and web_search_enabled()
    web_docs = _web_search(query, k=3) if should_try_web else []
    if web_docs:
        return {
            "docs": web_docs,
            "source_type": "web",
            "sources": sorted({_source_name(doc) for doc in web_docs}),
            "web_attempted": True,
        }

    return {
        "docs": [],
        "source_type": "none",
        "sources": [],
        "web_attempted": should_try_web,
    }
