"""
nodes.py - All LangGraph node functions for the AI Study Coach.
Strictly follows the Colab notebook logic with prompt improvements.
"""

import os
import re
from typing import Optional, List, Literal

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage

from .state import AgentState
from .ml_pipeline import run_ml_pipeline
from .rag import retrieve_academic_context
from .guardrails import (
    academic_scope_message,
    cheating_redirect_message,
    is_academic_query,
    is_disallowed_academic_request,
)

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY", ""),
)


def _last_human(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


def _normalise_query(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _has_student_data_signal(text: str) -> bool:
    value = _normalise_query(text)
    has_number = any(ch.isdigit() for ch in value)
    score_signal = (
        bool(re.search(r"\b(score|scored|marks?|grade)\b", value))
        or bool(re.search(r"\b\d{1,3}\s+(in|for)\s+(math|reading)\b", value))
        or bool(re.search(r"\b(math|reading)\s*(score|marks?|grade|is|=|:)?\s*\d{1,3}\b", value))
    )
    profile_words = (
        "study hour", "weekly hour", "hours per week", "parent", "test prep", "lunch", "sport", "gender",
        "siblings", "first child", "transport",
    )
    return score_signal or (has_number and any(word in value for word in profile_words))


def _same_retrieval_query(state: AgentState, user_msg: str) -> bool:
    return _normalise_query(state.get("last_retrieval_query", "")) == _normalise_query(user_msg)


def _parse_quiz_answers(text: str, expected_count: int) -> list[str]:
    """
    Accept answers like:
      A, C, B
      1. A
      2) C
      Q3: B
    """
    value = (text or "").strip().upper()
    if not value:
        return []

    keyed = re.findall(r"(?:^|\n|\s)(?:Q?\s*)?(\d+)\s*[\).:-]\s*([ABCD])\b", value)
    if keyed:
        ordered: dict[int, str] = {}
        for idx, answer in keyed:
            number = int(idx)
            if 1 <= number <= expected_count:
                ordered[number] = answer
        return [ordered[i] for i in range(1, expected_count + 1) if i in ordered]

    return re.findall(r"\b[ABCD]\b", value)[:expected_count]


def _wants_quiz(text: str) -> bool:
    value = _normalise_query(text)
    quiz_patterns = (
        "quiz", "test me", "mcq", "multiple choice", "practice questions",
        "question paper", "assess me",
    )
    return any(pattern in value for pattern in quiz_patterns)


def _wants_study_plan(text: str) -> bool:
    value = _normalise_query(text)
    plan_patterns = (
        "study plan", "7-day", "7 day", "schedule", "timetable", "revision plan",
        "daily plan", "weekly plan", "roadmap", "plan to improve",
    )
    return any(pattern in value for pattern in plan_patterns)


def _wants_performance_analysis(text: str) -> bool:
    value = _normalise_query(text)
    personal_terms = (" my ", " i ", " me ", " myself ")
    performance_terms = ("perform", "score", "scored", "marks", "grade", "predicted", "category")
    padded = f" {value} "
    return _has_student_data_signal(value) or (
        any(term in value for term in performance_terms)
        and any(term in padded for term in personal_terms)
    )


def _wants_concept_help(text: str) -> bool:
    value = _normalise_query(text)
    concept_patterns = (
        "explain", "what is", "how does", "how should", "why", "teach me",
        "concept", "example", "improve", "revise", "understand", "help with",
    )
    academic_topics = (
        "math", "reading", "writing", "algebra", "geometry", "trigonometry",
        "quadratic", "statistics", "probability", "active recall",
        "spaced repetition", "exam", "at-risk", "performance",
    )
    return any(pattern in value for pattern in concept_patterns) or any(topic in value for topic in academic_topics)


def _deterministic_plan(user_msg: str) -> list[str] | None:
    """Route common single-intent academic requests without spending an LLM call."""
    if not user_msg.strip():
        return ["end"]

    wants_quiz = _wants_quiz(user_msg)
    wants_plan = _wants_study_plan(user_msg)
    wants_analysis = _wants_performance_analysis(user_msg)
    wants_concept = _wants_concept_help(user_msg)

    steps: list[str] = []
    if wants_analysis:
        steps.append("analyser")
    if wants_quiz:
        steps.append("quizzer")
    elif wants_plan:
        if wants_concept or not wants_analysis:
            steps.append("retriever")
        steps.append("planner")
    elif wants_concept:
        steps.append("retriever")

    if steps:
        return list(dict.fromkeys(steps + ["end"]))
    return None


def _source_note(source_type: str, sources: list[str]) -> str:
    if source_type == "web" and sources:
        return (
            "I could not find enough coverage in the local academic knowledge base, "
            f"so I used web search results from: {', '.join(sources)}."
        )
    if source_type == "local" and sources:
        return f"Source basis: {', '.join(sources)}."
    return ""


def _sources_from_docs(docs: list[str]) -> list[str]:
    sources = []
    for doc in docs:
        source = doc.split(":", 1)[0].strip() if ":" in doc else "Internal Knowledge Base"
        if source:
            sources.append(source)
    return sorted(set(sources))


def _no_context_message(web_attempted: bool) -> str:
    if web_attempted:
        return (
            "I do not currently have enough reliable information about that topic in the local knowledge base, "
            "and web search did not return usable academic resources. Please try a more specific academic topic "
            "or add trusted notes for this subject to the knowledge base."
        )
    return (
        "I do not currently have enough information about that topic in the local academic knowledge base. "
        "Please add trusted notes for this subject or enable academic web search."
    )


def _is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return error.__class__.__name__ == "RateLimitError" or "rate limit" in text or "429" in text


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MASTER / SUPERVISOR NODE
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionPlan(BaseModel):
    steps: List[Literal["analyser", "retriever", "planner", "quizzer", "end"]] = Field(
        description=(
            "Ordered list of nodes to visit. "
            "Use ['end'] for greetings, off-topic queries, or anything that doesn't need a specialist."
        )
    )
    reasoning: str = Field(description="Brief explanation of why these steps were chosen.")


def master_node(state: AgentState) -> dict:
    print("--- NODE: MASTER (ARCHITECT) ---")
    user_msg       = _last_human(state)
    user_msg_lower = user_msg.lower()

    # Quiz bypass: if a quiz is active, always go to quizzer
    if state.get("quiz_active", False):
        return {"next_node": "quizzer"}

    if is_disallowed_academic_request(user_msg):
        return {
            "plan": ["end"],
            "current_step_index": 1,
            "next_node": "end",
        }

    if not is_academic_query(user_msg):
        return {
            "plan": ["end"],
            "current_step_index": 1,
            "next_node": "end",
        }

    plan     = state.get("plan", [])
    step_idx = state.get("current_step_index", 0)

    # Execute existing plan step
    if plan and step_idx < len(plan):
        next_node = plan[step_idx]
        print(f"Executing Step {step_idx + 1}/{len(plan)} → {next_node}")
        return {"current_step_index": step_idx + 1, "next_node": next_node}

    deterministic_plan = _deterministic_plan(user_msg)
    if deterministic_plan:
        print(f"Plan      → {deterministic_plan}")
        print("Reasoning → deterministic academic intent route")
        return {
            "plan":               deterministic_plan,
            "current_step_index": 1,
            "next_node":          deterministic_plan[0],
        }

    # Build a new plan
    architect_llm  = llm.with_structured_output(ExecutionPlan)

    system_prompt = f"""
You are an Agentic System Architect. Build a minimal, correct execution plan.

--- NODE RULES ---
- analyser  → ONLY when the user provides personal scores/profile data or asks about their own performance
- retriever → concept explanations, academic advice, generic improvement guidance, study-science questions
- planner   → ONLY when the user explicitly asks for a plan, schedule, timetable, roadmap, or 7-day/weekly plan
- quizzer   → ONLY when the user asks for a quiz or wants to be tested
- end       → for greetings, closings, off-topic, or when no specialist is needed

--- STATE-AWARE RULES (hard) ---
- If ml_results is already populated  → DO NOT include analyser
- If retrieved_docs is already set    → DO NOT include retriever

--- DEPENDENCY RULES ---
- planner + topic keyword → retriever must come before planner
- analyser must come before planner when scores are present

--- CONSTRAINTS ---
- No duplicate nodes
- Always end list with "end"
- Off-topic or greeting → ["end"]
- Explanation-only query → ["retriever", "end"]
- Generic "how should a student improve" query → ["retriever", "end"], NOT planner
- Do not include planner unless the user explicitly asks for a plan/schedule/timetable/roadmap
- Do not include analyser for generic student categories like "an at-risk student"; analyser is for this user's data only

Current state snapshot:
- ml_results populated: {bool(state.get("ml_results"))}
- retrieved_docs set:   {bool(state.get("retrieved_docs"))}

USER QUERY: {user_msg}
"""

    try:
        plan_obj = architect_llm.invoke(system_prompt)
        new_plan = plan_obj.steps
        reasoning = plan_obj.reasoning
    except Exception as e:
        print(f"[master] planner fallback due to llm error: {e}")
        fallback_plan = _deterministic_plan(user_msg) or (
            ["retriever", "end"] if _wants_concept_help(user_msg) else ["end"]
        )
        return {
            "plan": fallback_plan,
            "current_step_index": 1,
            "next_node": fallback_plan[0],
            "messages": (
                [AIMessage(content="The coach is currently under heavy load, so I switched to a lightweight routing path.")]
                if _is_rate_limit_error(e)
                else []
            ),
        }

    # ── Guardrails ────────────────────────────────────────────────────────────
    valid    = {"analyser", "retriever", "planner", "quizzer", "end"}
    new_plan = list(dict.fromkeys([s for s in new_plan if s in valid]))

    if "end" not in new_plan:
        new_plan.append("end")

    # Dependency: planner + topic → needs retriever
    if "planner" in new_plan and "retriever" not in new_plan:
        topic_words = ["explain", "concept", "topic", "learn", "algebra",
                       "geometry", "calculus", "trigonometry", "quadratic",
                       "math", "reading", "writing", "science"]
        if any(w in user_msg_lower for w in topic_words):
            new_plan.insert(new_plan.index("planner"), "retriever")

    has_new_student_data = _has_student_data_signal(user_msg)

    # Deterministic additions for obvious user updates.
    if has_new_student_data and "analyser" not in new_plan:
        insert_at = new_plan.index("planner") if "planner" in new_plan else 0
        new_plan.insert(insert_at, "analyser")

    # State-aware corrections: reuse old analysis only when the current turn does
    # not provide fresh profile/score information.
    if "analyser" in new_plan and state.get("ml_results") and not has_new_student_data:
        new_plan.remove("analyser")

    # Reuse retrieval only for the exact same query. A different topic must get
    # fresh documents instead of leaking stale context from a previous turn.
    if "retriever" in new_plan and state.get("retrieved_docs") and _same_retrieval_query(state, user_msg):
        new_plan.remove("retriever")

    if not new_plan:
        new_plan = ["end"]

    print(f"Plan      → {new_plan}")
    print(f"Reasoning → {reasoning}")

    return {
        "plan":               new_plan,
        "current_step_index": 1,
        "next_node":          new_plan[0],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ANALYSER NODE
# ═══════════════════════════════════════════════════════════════════════════════

class StudentDataSchema(BaseModel):
    math:           Optional[int]   = Field(None, ge=0, le=100, description="Math score 0-100")
    reading:        Optional[int]   = Field(None, ge=0, le=100, description="Reading score 0-100")
    study_hours:    Optional[float] = Field(None, ge=0, le=80, description="Average weekly study hours")
    parent_educ:    Optional[int]   = Field(
        None,
        description="Parent education: high_school=1, some_college=2, associates=3, bachelors=4, masters=5"
    )
    test_prep:      Optional[str]   = Field(None, description="'none' or 'completed'")
    lunch:          Optional[str]   = Field(None, description="'standard' or 'free/reduced'")
    sport:          Optional[int]   = Field(None, ge=0, le=2, description="Sports: never=0, sometimes=1, regularly=2")
    gender:         Optional[str]   = Field(None, description="'male' or 'female'")
    siblings:       Optional[int]   = Field(None, ge=0, le=20, description="Number of siblings")
    is_first_child: Optional[str]   = Field(None, description="'yes' or 'no'")
    transport:      Optional[str]   = Field(None, description="'school_bus' or 'private'")


def _extraction_logic(user_message: str) -> dict:
    structured_llm = llm.with_structured_output(StudentDataSchema)
    prompt = (
        "You are a precise data extraction specialist.\n"
        "Extract student profile values from the message below.\n"
        "Only extract values that are explicitly stated or clearly implied. "
        "Do not guess or invent values.\n\n"
        f'USER MESSAGE: "{user_message}"'
    )
    try:
        obj = structured_llm.invoke(prompt)
        raw = obj.model_dump(exclude_none=True)
        return {k: v for k, v in raw.items() if v is not None}
    except Exception as e:
        print(f"[analyser] Extraction error: {e}")
        return {}


def analyser_node(state: AgentState) -> dict:
    print("--- NODE: ANALYSER ---")

    last_msg        = _last_human(state)
    newly_extracted = _extraction_logic(last_msg)

    current_data = state.get("student_data", {}).copy()
    current_data.update(newly_extracted)

    ml_results = run_ml_pipeline(current_data) if current_data else state.get("ml_results", {})

    captured = ", ".join(newly_extracted.keys())
    if not captured and not current_data:
        analysis_msg = (
            "I've reviewed your message. To give you a more accurate prediction, "
            "could you share your math or reading scores?"
        )
    else:
        score    = ml_results["predicted_score"]
        status   = ml_results["status"]
        category = ml_results["category"]
        assumed_defaults = ml_results.get("assumed_defaults", {})
        prefix = f"I've updated your profile with: **{captured}**." if captured else "Using your saved student profile:"
        analysis_msg = (
            f"{prefix}\n\n"
            f"### Performance Snapshot\n\n"
            f"- **Predicted Exam Score:** {score}\n"
            f"- **Status:** {status}\n"
            f"- **Learner Category:** {category}\n\n"
        )
        if assumed_defaults:
            shown_defaults = list(assumed_defaults.items())[:5]
            default_text = ", ".join([f"{k}: {v}" for k, v in shown_defaults])
            remaining = len(assumed_defaults) - len(shown_defaults)
            if remaining > 0:
                default_text += f", and {remaining} more"
            analysis_msg += (
                "### Assumptions Used\n\n"
                "You did not provide every model input, so I filled missing fields with baseline defaults "
                f"from the training setup: {default_text}.\n\n"
                "For a more personalised analysis, share details such as reading score, weekly study hours, "
                "test-prep status, and other available profile fields.\n\n"
            )
        analysis_msg += (
            "### Next Best Actions\n\n"
            "- Use structured test preparation instead of only rereading notes.\n"
            "- Practise the weakest topic in short daily blocks.\n"
            "- Keep a mistake log so repeated errors become visible."
        )

    return {
        "student_data": current_data,
        "ml_results":   ml_results,
        "last_analysis_input": _normalise_query(str(sorted(current_data.items()))),
        "messages":     [AIMessage(content=analysis_msg)],
        "next_node":    "supervisor",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RETRIEVER NODE (RAG)
# ═══════════════════════════════════════════════════════════════════════════════

def retriever_node(state: AgentState) -> dict:
    print("--- NODE: RETRIEVER (RAG) ---")

    user_query = _last_human(state)
    category   = state.get("ml_results", {}).get("category", "General")

    retrieval = retrieve_academic_context(user_query, k=4, allow_web=True)
    docs = retrieval["docs"]
    if not docs:
        return {
            "retrieved_docs": [],
            "last_retrieval_query": user_query,
            "messages": [AIMessage(content=_no_context_message(retrieval["web_attempted"]))],
            "next_node": "supervisor",
        }

    context = "\n\n".join(docs)
    source_note = _source_note(retrieval["source_type"], retrieval["sources"])

    prompt = f"""
You are an enterprise-grade Academic Tutor for a student study-coach product.

User Category: {category}
User Question: {user_query}
Source Status: {retrieval["source_type"]}
Sources Used: {", ".join(retrieval["sources"]) if retrieval["sources"] else "None"}

Retrieved Facts:
{context}

Task:
- Answer the user's current question only.
- Use ONLY the retrieved facts above.
- If Source Status is "web", explicitly say that web search was used and name the sources.
- Do not create a study plan unless the user explicitly asked for a plan, schedule, timetable, roadmap, or 7-day plan.
- Do not include personal performance analysis unless the user supplied their own scores/profile.
- If the retrieved facts are not enough, say that clearly and ask for a more specific academic topic.

Personalisation:
- Tailor depth to the category ({category}):
  * At-Risk → very simple, step-by-step, lots of examples
  * Average → clear explanation with worked examples
  * High-Performer → detailed, include edge cases and advanced notes
  * General → balanced, accessible explanation

Formatting (required):
- Start with one clear `###` heading that names the topic.
- Then write 1-2 short overview paragraphs.
- Add a `**Key ideas**` bullet list when there are 2+ important points.
- Add a `**Example**` section if the user asked for an example or if an example would make the concept clearer.
- End with a short practical takeaway only if useful.
- Do not use unrelated sections like "Study Plan" or "Performance Insight" unless explicitly requested.
- Keep Markdown clean: blank line between paragraphs, bullets on separate lines, no dense walls of text.
"""

    response = llm.invoke(prompt)
    content = response.content
    if source_note and retrieval["source_type"] == "web" and source_note not in content:
        content = f"{source_note}\n\n{content}"

    return {
        "retrieved_docs": docs,
        "last_retrieval_query": user_query,
        "messages":       [AIMessage(content=content)],
        "next_node":      "supervisor",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLANNER NODE
# ═══════════════════════════════════════════════════════════════════════════════

class StudyDay(BaseModel):
    day:            str       = Field(description="e.g., Day 1")
    topic:          str       = Field(description="Focus area for the day")
    activities:     List[str] = Field(description="List of specific, actionable tasks")
    estimated_time: str       = Field(description="e.g., 2 hours")


class WeeklyPlan(BaseModel):
    title:   str            = Field(description="Title of the study plan")
    days:    List[StudyDay] = Field(description="7 individual daily plans")
    summary: str            = Field(
        description="Practical guidance to follow during the plan — NOT a completion message"
    )


def planner_node(state: AgentState) -> dict:
    print("--- NODE: PLANNER ---")

    results  = state.get("ml_results", {})
    category = results.get("category", "Unknown")
    score    = results.get("predicted_score", "N/A")

    last_user_msg      = _last_human(state)
    external_knowledge = "\n".join(state.get("retrieved_docs", []))
    source_names = _sources_from_docs(state.get("retrieved_docs", []))

    if not external_knowledge.strip():
        msg = (
            "I do not have enough reliable academic material to create a grounded study plan for that topic yet. "
            "Please add trusted notes for this subject to the knowledge base or enable academic web search, then ask again."
        )
        return {
            "study_plan": "",
            "messages": [AIMessage(content=msg)],
            "next_node": "supervisor",
        }

    planner_llm = llm.with_structured_output(WeeklyPlan)

    prompt = f"""
You are an enterprise-grade Study Coach.

STUDENT'S GOAL: "{last_user_msg}"
SOURCE NAMES: {", ".join(source_names) if source_names else "Internal Knowledge Base"}
LEARNING RESOURCES:
{external_knowledge}

ACADEMIC DATA (Secondary Context):
- Category: {category}
- Current Score: {score}

YOUR JOB:
Create a 7-day study plan that turns the LEARNING RESOURCES into a concrete, realistic daily schedule.

CRITICAL RULES:
1. Ground the plan in the LEARNING RESOURCES. Do not invent unrelated syllabus content.
2. If Academic Data is 'Unknown' or missing, build the plan based only on Goal + Resources.
3. Each day must include:
   - a clear objective,
   - at least one active learning task,
   - at least one practice or recall task,
   - a short checkpoint or reflection task.
4. Use Category ONLY to tune pace:
   - 'At-Risk': One concept per day, more breaks, full revision on Day 7.
   - 'Average': Moderate pace, mix of theory and practice.
   - 'High-Performer': Combine multiple concepts, add a Challenge Project on Day 7.
   - 'Unknown': Standard, well-balanced plan. Do NOT mention missing data.
5. Every activity must be specific and actionable, not vague.
6. Do NOT include completion praise. This is a plan to follow, not a task already completed.
"""

    plan = planner_llm.invoke(prompt)

    tag = (
        f"*Optimised for **{category}** performance level*"
        if category != "Unknown"
        else "*Standard Comprehensive Plan*"
    )

    md = f"### 🗓️ {plan.title}\n{tag}\n\n"
    if source_names:
        md += f"*Built from: {', '.join(source_names)}*\n\n"
    for d in plan.days:
        md += f"**{d.day}: {d.topic}**\n"
        md += "- " + "\n- ".join(d.activities)
        md += f"\n\n*Estimated time: {d.estimated_time}*\n\n"

    if plan.summary.strip():
        md += f"---\n**💡 How to Approach This Plan:** {plan.summary}"

    return {
        "study_plan": md,
        "messages":   [AIMessage(content=md)],
        "next_node":  "supervisor",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. QUIZZER NODE
# ═══════════════════════════════════════════════════════════════════════════════

class Question(BaseModel):
    question:       str       = Field(description="The quiz question text")
    options:        List[str] = Field(description="Exactly 4 multiple-choice options")
    correct_answer: str       = Field(description="Exact text of the correct option")
    explanation:    str       = Field(description="Brief explanation of why this is correct")


class QuizSet(BaseModel):
    topic:     str            = Field(description="Topic of the quiz")
    questions: List[Question] = Field(
        min_length=5,
        max_length=5,
        description="Exactly 5 quiz questions"
    )


def quizzer_node(state: AgentState) -> dict:
    print("--- NODE: QUIZZER ---")

    last_msg = _last_human(state)
    if not last_msg:
        return {"next_node": "supervisor"}

    quiz_active = state.get("quiz_active", False)
    current_idx = state.get("current_q_idx", 0)
    questions   = state.get("quiz_questions", [])
    score       = state.get("quiz_score", 0)

    # ── PHASE A: Initialise new quiz ──────────────────────────────────────────
    if not quiz_active or not questions:
        print("[quizzer] Initialising new quiz...")
        retrieval = retrieve_academic_context(last_msg, k=4, allow_web=True)
        docs = retrieval["docs"]
        if not docs:
            return {
                "quiz_active": False,
                "awaiting_answer": False,
                "messages": [AIMessage(content=_no_context_message(retrieval["web_attempted"]))],
                "next_node": "supervisor",
            }

        context = "\n\n".join(docs)
        source_note = _source_note(retrieval["source_type"], retrieval["sources"])
        quiz_llm  = llm.with_structured_output(QuizSet)
        prompt    = f"""
You are an enterprise-grade academic quiz generator.

Student request: {last_msg}
Source Status: {retrieval["source_type"]}
Sources Used: {", ".join(retrieval["sources"]) if retrieval["sources"] else "None"}

Grounding Material:
{context}

Create exactly 5 multiple-choice questions from the grounding material.

Rules:
- Each question must test understanding, not trivia.
- Each question must have exactly 4 distinct answer options.
- Exactly one option must be correct.
- correct_answer must exactly match the text of the correct option.
- Vary difficulty from easy to moderate.
- Do not ask about facts that are not present in the grounding material.
"""
        generated = quiz_llm.invoke(prompt)

        questions_md = []
        for q_idx, q in enumerate(generated.questions, start=1):
            opts = "\n".join([f"- **{chr(65+i)}.** {o}" for i, o in enumerate(q.options)])
            questions_md.append(f"**Q{q_idx}. {q.question}**\n\n{opts}")

        msg  = (
            f"🎯 **Quiz Started — {generated.topic}**\n\n"
            + (f"{source_note}\n\n" if source_note else "")
            + "\n\n---\n\n".join(questions_md)
            + "\n\n---\n\n"
            "Reply with all answers together, for example:\n\n"
            "`1. A, 2. C, 3. B, 4. D, 5. A`"
        )
        return {
            "quiz_questions":  [q.model_dump() for q in generated.questions],
            "current_q_idx":   0,
            "quiz_score":      0,
            "quiz_active":     True,
            "awaiting_answer": True,
            "retrieved_docs":   docs,
            "last_retrieval_query": last_msg,
            "messages":        [AIMessage(content=msg)],
            "next_node":       "supervisor",
        }

    # ── PHASE B: Evaluate all submitted answers ───────────────────────────────
    answers = _parse_quiz_answers(last_msg, len(questions))
    if len(answers) < len(questions):
        missing = len(questions) - len(answers)
        msg = (
            f"I found **{len(answers)}** answer(s), but this quiz has **{len(questions)}** questions. "
            f"Please send the remaining {missing} answer(s) using A, B, C, or D.\n\n"
            "Example: `1. A, 2. C, 3. B, 4. D, 5. A`"
        )
        return {
            "messages":        [AIMessage(content=msg)],
            "awaiting_answer": True,
            "next_node":       "supervisor",
        }

    results_md: list[str] = []
    weak_topics: list[str] = []
    new_score = 0
    for idx, (question, letter) in enumerate(zip(questions, answers), start=1):
        option_map = {chr(65 + i): opt for i, opt in enumerate(question["options"])}
        selected = option_map.get(letter, "")
        correct = question["correct_answer"]
        is_correct = selected.strip().lower() == correct.strip().lower()
        if is_correct:
            new_score += 1
        else:
            weak_topics.append(question["question"])

        status = "✅ Correct" if is_correct else "❌ Incorrect"
        results_md.append(
            f"**Q{idx}. {status}**\n"
            f"- Your answer: **{letter}. {selected or 'Invalid'}**\n"
            f"- Correct answer: **{correct}**\n"
            f"- Why: {question['explanation']}"
        )

    total = len(questions)
    if new_score == total:
        closing = "🏆 **Perfect score! Outstanding work!**"
    elif new_score >= total // 2:
        closing = "👍 **Good effort! Review the questions you missed and try again soon.**"
    else:
        closing = "📖 **Keep practising — consistency is key. You'll get there!**"

    final = (
        f"{closing}\n\n"
        f"**Final Score: {new_score} / {total}**\n\n"
        + "\n\n---\n\n".join(results_md)
    )
    if weak_topics:
        final += (
            "\n\n---\n\n"
            "### What To Revise Next\n\n"
            "- Review the concepts behind the questions you missed.\n"
            "- Re-attempt similar problems without looking at the explanation.\n"
            "- Ask me for a focused mini-lesson on any missed question."
        )

    return {
        "quiz_active":     False,
        "awaiting_answer": False,
        "quiz_score":      new_score,
        "current_q_idx":   total,
        "messages":        [AIMessage(content=final)],
        "next_node":       "supervisor",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. END / RESPONSE NODE
# ═══════════════════════════════════════════════════════════════════════════════

def end_node(state: AgentState) -> dict:
    print("--- NODE: END/RESPONSE ---")

    last_entry = state["messages"][-1]
    last_message = last_entry.content
    last_user_msg = _last_human(state)

    if is_disallowed_academic_request(last_user_msg):
        return {"messages": [AIMessage(content=cheating_redirect_message())]}

    if not state.get("quiz_active", False) and not is_academic_query(last_user_msg):
        return {"messages": [AIMessage(content=academic_scope_message())]}

    # Specialist nodes already produce polished, intent-specific answers. Do not
    # ask a final LLM to reinterpret them, because that can add irrelevant
    # sections such as study plans to explanation-only answers.
    if isinstance(last_entry, AIMessage):
        return {}

    prompt = f"""
You are an AI Study Coach generating the FINAL response for a student.

LAST MESSAGE IN CONVERSATION:
"{last_message}"

AVAILABLE CONTEXT:
- Student Data:        {state.get("student_data")}
- ML Analysis:         {state.get("ml_results")}
- Retrieved Knowledge: {state.get("retrieved_docs")}
- Study Plan:          {state.get("study_plan")}
- Quiz Active:         {state.get("quiz_active")}

--- CORE PRINCIPLE ---
Use ONLY context that is directly relevant to the last message.
Never blindly dump all available context.

--- INTENT DETECTION ---
Identify ALL intents present:
GREETING | CLOSING | STUDY_PLAN | CONCEPT_EXPLANATION | PERFORMANCE_INSIGHT | QUIZ | OFF_TOPIC

--- PRIORITY (highest to lowest) ---
1. QUIZ
2. STUDY_PLAN
3. PERFORMANCE_INSIGHT
4. CONCEPT_EXPLANATION
5. GREETING / CLOSING
6. OFF_TOPIC

--- CONTEXT SELECTION ---
- STUDY_PLAN          → use study_plan field
- CONCEPT_EXPLANATION → use retrieved_docs
- PERFORMANCE_INSIGHT → use ml_results (explain in plain English, do not dump raw numbers)
- QUIZ                → refer to quiz context only
- GREETING / CLOSING  → no extra context needed
- OFF_TOPIC           → politely redirect to academic topics

--- FORMAT ---
Single intent  → focused prose, no section headers
Multiple intents → use ONLY these headers in order (omit unused ones):
  📊 Performance Insight
  📚 Concept Explanation
  🗓️ Study Plan

--- STYLE ---
- Warm, encouraging, student-friendly tone
- Never expose raw data dumps or internal system terms
- Keep simple replies concise; structured replies may be detailed
- No filler phrases like "Certainly!", "Of course!", "Great question!"

--- MARKDOWN FORMATTING (required) ---
- Short paragraphs separated by a blank line (never one huge wall of text)
- Use bullet lists (- item) when listing 3+ related points
- Use ### only when multiple clear sections help readability
- Bold key terms sparingly

Generate the best possible response now.
"""

    try:
        response = llm.invoke(prompt)
        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        print(f"[end] fallback due to llm error: {e}")
        if _is_rate_limit_error(e):
            msg = (
                "I can still help, but the language model quota is temporarily exhausted right now. "
                "Please retry in a few minutes, or reduce response length while quota resets."
            )
        else:
            msg = (
                "I hit a temporary response-generation issue. Please retry your last message, "
                "and I will continue from the same context."
            )
        return {"messages": [AIMessage(content=msg)]}
