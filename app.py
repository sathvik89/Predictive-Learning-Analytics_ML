"""
app.py – Predictive Learning Analytics + AI Study Coach
========================================================
Changes from v1:
  1. Sidebar uses styled buttons (not dropdown) to switch pages
  2. AI Study Coach page: fully redesigned chat UI matching M1 theme
  3. Chat history persisted in SQLite (public / shared sessions)
  4. Removed "Current Snapshot" sidebar panel from coach page
  5. nodes.py strictly follows Colab logic
"""

import os
import uuid
import traceback
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG (must be FIRST st call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Learning Analytics",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES (from styles.py)
# ─────────────────────────────────────────────
from styles import apply_global_styles
apply_global_styles()

# ─────────────────────────────────────────────
# DB INIT
# ─────────────────────────────────────────────
from agent.chat_history import (
    init_db,
    create_session,
    session_exists,
    save_message,
    load_messages,
    list_sessions,
    update_session_title,
    delete_session,
    get_session,
    save_agent_state,
    load_agent_state,
)
from agent.session_context import trim_messages
from agent.formatting import polish_assistant_markdown
init_db()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

def _sidebar_nav():
    with st.sidebar:
        current = st.session_state.get("current_page", "dashboard")

        # ── TOP: M1 branding + nav (always visible) ───────────────────────────
        # This block is intentionally rendered first so it sits at the very top
        # of the sidebar regardless of which page is active.
        from modules.icons import icon as _icon

        st.markdown(f"""
        <div style='padding: 1.4rem 1rem 1.1rem 1rem; border-bottom: 1px solid var(--border); margin-bottom: 1.1rem;'>
            <div style='display:flex; align-items:start; gap:0.9rem; margin-bottom:1rem;'>
                <div style='margin-top: 0.3rem;'>{_icon("layers", "#b07d4e")}</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.85rem; font-weight: 900;
                            color: var(--text); line-height: 1.05; letter-spacing: -0.02em;'>
                    Student Performance<br>Analytics
                </div>
            </div>
            <div style='font-size: 0.78rem; color: var(--text-dim); margin-top: 0.7rem;
                        text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.75;'>
                ML Analytics &middot; Milestone 1 & 2
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Dashboard sub-nav (Home / Performance / Predict) ─────────────────
        # Only show when on the dashboard page so the coach sidebar stays clean
        if current == "dashboard":
            if st.button("HOME", icon=":material/home:", use_container_width=True, key="dash_home"):
                st.session_state.page = "home"
                st.session_state.prediction_run = False
                st.rerun()
            if st.button("PERFORMANCE", icon=":material/insights:", use_container_width=True, key="dash_perf"):
                st.session_state.page = "performance"
                st.rerun()
            if st.button("PREDICT", icon=":material/sensors:", use_container_width=True, key="dash_pred"):
                st.session_state.page = "predict"
                st.rerun()

        st.markdown('<hr style="border-color:var(--border); margin:0.7rem 0;">', unsafe_allow_html=True)

        # ── Page switcher buttons (Dashboard ↔️ AI Study Coach) ────────────────
        dash_style  = "nav-btn-active" if current == "dashboard" else ""
        coach_style = "nav-btn-active" if current == "coach" else ""

        st.markdown(f'<div class="{dash_style}">', unsafe_allow_html=True)
        if st.button("Dashboard", icon=":material/bar_chart:", use_container_width=True, key="nav_dash"):
            st.session_state.current_page = "dashboard"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="{coach_style}">', unsafe_allow_html=True)
        if st.button("AI Study Coach", icon=":material/smart_toy:", use_container_width=True, key="nav_coach"):
            st.session_state.current_page = "coach"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Coach-specific sidebar content ─────────────────────────────────────
        if current == "coach":
            st.markdown('<hr style="border-color:var(--border); margin:1rem 0 0.8rem 0;">', unsafe_allow_html=True)

            # New conversation button
            if st.button("New Conversation", icon=":material/add:", use_container_width=True, key="new_conv"):
                _start_new_session()
                st.rerun()

            # Quiz progress (only when quiz is active)
            agent_state = st.session_state.get("agent_state", {})
            if agent_state.get("quiz_active", False):
                q_idx   = agent_state.get("current_q_idx", 0)
                q_tot   = len(agent_state.get("quiz_questions", []))
                q_score = agent_state.get("quiz_score", 0)
                pct     = int((q_idx / max(q_tot, 1)) * 100)

                st.markdown(
                    f"""
                    <div style="margin-top:0.8rem;">
                      <div class="sidebar-section-label">{_icon("target", "var(--accent)")} &nbsp;Quiz in Progress</div>
                      <div class="quiz-progress-bar">
                        <div class="quiz-progress-fill" style="width:{pct}%"></div>
                      </div>
                      <div style="font-size:0.78rem; color:var(--text-dim); margin-top:0.4rem;">
                        Question {q_idx + 1} of {q_tot} &nbsp;·&nbsp; Score: {q_score}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Past conversations
            st.markdown(
                f'<div class="sidebar-section-label">{_icon("clock")} Past conversations</div>',
                unsafe_allow_html=True,
            )
            rename_sid = st.session_state.get("coach_rename_sid")
            if rename_sid:
                row = get_session(rename_sid)
                if row:
                    st.markdown(
                        '<div style="font-size:0.78rem; color:var(--text-dim); margin-bottom:0.4rem;">Rename</div>',
                        unsafe_allow_html=True,
                    )
                    new_title = st.text_input(
                        "Title",
                        value=row.get("title") or "Conversation",
                        key="coach_rename_field",
                        label_visibility="collapsed",
                    )
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        if st.button("Save", key="coach_rename_save", use_container_width=True):
                            update_session_title(rename_sid, new_title.strip() or "Conversation")
                            st.session_state.pop("coach_rename_sid", None)
                            st.rerun()
                    with rc2:
                        if st.button("Cancel", key="coach_rename_cancel", use_container_width=True):
                            st.session_state.pop("coach_rename_sid", None)
                            st.rerun()
                else:
                    st.session_state.pop("coach_rename_sid", None)

            sessions    = list_sessions(limit=25)
            current_sid = st.session_state.get("session_id", "")
            if sessions:
                for s in sessions:
                    sid      = s["session_id"]
                    title    = (s.get("title") or "Untitled").strip() or "Untitled"
                    count    = int(s.get("message_count") or 0)
                    last_at  = (s.get("last_at") or s.get("started_at") or "")[:10]
                    is_active = sid == current_sid
                    short    = title if len(title) <= 28 else title[:25] + "…"
                    label    = f"{short}  ·  {last_at}  ·  {count}"

                    row_cols = st.columns([5.2, 1])
                    with row_cols[0]:
                        if st.button(
                            label,
                            key=f"open_sess_{sid}",
                            use_container_width=True,
                            type="primary" if is_active else "secondary",
                        ):
                            _load_session(sid)
                            st.session_state.pop("coach_rename_sid", None)
                            st.rerun()
                    with row_cols[1]:
                        with st.popover("⋮", help="Rename or delete"):
                            if st.button("Rename", key=f"pop_ren_{sid}", use_container_width=True):
                                st.session_state["coach_rename_sid"] = sid
                                st.rerun()
                            if st.button("Delete", key=f"pop_del_{sid}", use_container_width=True):
                                delete_session(sid)
                                if sid == current_sid:
                                    _start_new_session()
                                if st.session_state.get("coach_rename_sid") == sid:
                                    st.session_state.pop("coach_rename_sid", None)
                                st.rerun()
            else:
                st.markdown(
                    '<div style="font-size:0.8rem; color:var(--text-dim);">No conversations yet.</div>',
                    unsafe_allow_html=True,
                )

        # ── M1 model / system info cards (dashboard page only) ────────────────
        if current == "dashboard":
            from modules.icons import icon as _icon2
            st.markdown(f"""
            <div style='margin-top: 0.8rem; padding: 1.1rem; background: var(--bg-card);
                        border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow);'>
                <div style='font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
                            letter-spacing: 0.12em; color: var(--accent); margin-bottom: 0.6rem;
                            border-bottom: 1px solid var(--border); padding-bottom: 0.3rem;'>Model Performance</div>
                <div style='font-size: 0.82rem; color: var(--text); font-weight: 600; margin-bottom: 0.5rem;
                            display:flex; justify-content:space-between;'><span>Accuracy</span><span>91.76%</span></div>
                <div style='font-size: 0.82rem; color: var(--text); font-weight: 600; margin-bottom: 0.5rem;
                            display:flex; justify-content:space-between;'><span>R&#178; Score</span><span>0.9397</span></div>
                <div style='font-size: 0.82rem; color: var(--text); font-weight: 600; margin-bottom: 0.5rem;
                            display:flex; justify-content:space-between;'><span>F1 Score</span><span>0.9176</span></div>
                <div style='font-size: 0.82rem; color: var(--text); font-weight: 600;
                            display:flex; justify-content:space-between;'><span>MAE</span><span>3.04 marks</span></div>
            </div>
            <div style='margin-top: 0.8rem; padding: 1.1rem; background: var(--bg-card);
                        border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow);'>
                <div style='font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
                            letter-spacing: 0.12em; color: var(--accent); margin-bottom: 0.6rem;
                            border-bottom: 1px solid var(--border); padding-bottom: 0.3rem;'>System Info</div>
                <div style='font-size: 0.82rem; color: var(--text); font-weight: 500; margin-bottom: 0.5rem;
                            display:flex; align-items:center; gap:0.5rem;'>
                    {_icon2("database", "#b07d4e")} 30,640 records loaded
                </div>
                <div style='font-size: 0.82rem; color: var(--text); font-weight: 500; margin-bottom: 0.5rem;
                            display:flex; align-items:center; gap:0.5rem;'>
                    {_icon2("cpu", "#b07d4e")} 3 models active
                </div>
                <div style='font-size: 0.82rem; color: var(--text); font-weight: 500;
                            display:flex; align-items:center; gap:0.5rem;'>
                    {_icon2("settings", "#b07d4e")} 11 features enabled
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_fresh_agent_state() -> dict:
    return {
        "messages":             [],
        "student_data":         {},
        "ml_results":           {},
        "retrieved_docs":       [],
        "last_retrieval_query": "",
        "quiz_questions":       [],
        "current_q_idx":        0,
        "quiz_score":           0,
        "quiz_active":          False,
        "awaiting_answer":      False,
        "study_plan":           "",
        "plan":                 [],
        "task_plan":            [],
        "response_parts":       [],
        "response_mode":        "academic",
        "direct_response":      "",
        "current_step_index":   0,
        "next_node":            "",
    }


_PERSISTED_AGENT_KEYS = (
    "student_data",
    "ml_results",
    "retrieved_docs",
    "last_retrieval_query",
    "quiz_questions",
    "current_q_idx",
    "quiz_score",
    "quiz_active",
    "awaiting_answer",
    "study_plan",
    "plan",
    "task_plan",
    "response_parts",
    "response_mode",
    "direct_response",
    "current_step_index",
    "next_node",
)


def _serializable_agent_state(state: dict) -> dict:
    defaults = _build_fresh_agent_state()
    return {key: state.get(key, defaults.get(key)) for key in _PERSISTED_AGENT_KEYS}


def _start_new_session():
    new_sid = str(uuid.uuid4())
    create_session(new_sid)
    st.session_state.session_id   = new_sid
    st.session_state.chat_display = []
    st.session_state.agent_state  = _build_fresh_agent_state()


def _load_session(session_id: str):
    from langchain_core.messages import HumanMessage, AIMessage as AI

    msgs_db = load_messages(session_id)
    display = [{"role": m["role"], "content": m["content"]} for m in msgs_db]

    lc_msgs = []
    for m in msgs_db:
        if m["role"] == "user":
            lc_msgs.append(HumanMessage(content=m["content"]))
        else:
            lc_msgs.append(AI(content=m["content"]))

    new_state   = _build_fresh_agent_state()
    saved_state = load_agent_state(session_id) or {}
    for key in _PERSISTED_AGENT_KEYS:
        if key in saved_state:
            new_state[key] = saved_state[key]
    new_state["messages"] = trim_messages(lc_msgs)

    st.session_state.session_id   = session_id
    st.session_state.chat_display = display
    st.session_state.agent_state  = new_state


def _init_coach_session():
    if "session_id" not in st.session_state:
        _start_new_session()
    elif not session_exists(st.session_state.session_id):
        create_session(st.session_state.session_id)

    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []

    if "agent_state" not in st.session_state:
        st.session_state.agent_state = _build_fresh_agent_state()


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def show_dashboard():
    from modules.styling import apply_global_styles
    from modules.home import render as render_home
    from modules.performance import render as render_performance
    from modules.predict import render as render_predict

    if "prediction_run" not in st.session_state:
        st.session_state.prediction_run = False
    if "page" not in st.session_state:
        st.session_state.page = "home"

    apply_global_styles()

    selected_page = st.session_state.get("page", "home")
    if selected_page == "home":
        render_home()
    elif selected_page == "performance":
        render_performance()
    elif selected_page == "predict":
        render_predict()


# ═══════════════════════════════════════════════════════════════════════════════
# AI STUDY COACH PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _get_last_ai_content(result_state: dict) -> str:
    from langchain_core.messages import AIMessage
    ai_msgs = [m for m in result_state["messages"] if isinstance(m, AIMessage)]
    return ai_msgs[-1].content if ai_msgs else "*(no response)*"


def _patch_last_ai_message(msgs, new_content: str):
    from langchain_core.messages import AIMessage
    out = list(msgs)
    for i in range(len(out) - 1, -1, -1):
        if isinstance(out[i], AIMessage):
            out[i] = AIMessage(content=new_content)
            break
    return out


def _log_exception(context: str, error: Exception) -> str:
    error_id = str(uuid.uuid4())[:8]
    log_path = os.path.join(os.path.dirname(__file__), "app_errors.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"\n[{datetime.utcnow().isoformat()}] error_id={error_id} context={context}\n"
            f"{''.join(traceback.format_exception(type(error), error, error.__traceback__))}\n"
        )
    return error_id


def _friendly_error_message(error_id: str) -> str:
    return (
        "I hit a system issue while processing that request. Your conversation is still safe.\n\n"
        f"Please notify the project owner with this error ID: `{error_id}`. "
        "They can check the local application error log and fix the underlying issue."
    )


def _is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return error.__class__.__name__ == "RateLimitError" or "rate limit" in text or "429" in text


def _save_assistant_failure(session_id: str, content: str) -> None:
    save_message(session_id, "assistant", content)
    st.session_state.chat_display.append({"role": "assistant", "content": content})


def _handle_user_message(user_input: str, coach_app, status_box=None):
    from langchain_core.messages import HumanMessage

    sid = st.session_state.session_id

    if status_box:
        status_box.update(label="Saving your message and preparing context...", state="running")
    save_message(sid, "user", user_input)
    st.session_state.chat_display.append({"role": "user", "content": user_input})

    if status_box:
        status_box.update(label="Checking intent, student memory, and available academic sources...", state="running")
    sessions = list_sessions()
    current  = next((s for s in sessions if s["session_id"] == sid), None)
    if current and (current.get("title") in ("New Conversation", "") or int(current.get("message_count") or 0) <= 1):
        update_session_title(sid, user_input[:55] + ("…" if len(user_input) > 55 else ""))

    current_state = dict(st.session_state.agent_state)
    merged        = list(current_state["messages"]) + [HumanMessage(content=user_input)]
    current_state["messages"] = trim_messages(merged)

    if status_box:
        status_box.update(label="Running the study-coach graph...", state="running")
    try:
        result_state = coach_app.invoke(current_state)
    except Exception as e:
        if _is_rate_limit_error(e):
            ai_response = (
                "The AI provider token quota is temporarily exhausted, so the study coach cannot generate "
                "a full response right now.\n\nPlease wait a few minutes and retry, or use a shorter prompt."
            )
            save_message(sid, "assistant", ai_response)
            st.session_state.chat_display.append({"role": "assistant", "content": ai_response})
            if status_box:
                status_box.update(label="Provider quota limit reached. Please retry shortly.", state="error")
            return ai_response
        raise

    if status_box:
        status_box.update(label="Formatting and saving the response...", state="running")
    ai_raw      = _get_last_ai_content(result_state)
    ai_response = polish_assistant_markdown(ai_raw)

    result_state = dict(result_state)
    result_state["messages"] = trim_messages(_patch_last_ai_message(result_state["messages"], ai_response))

    save_message(sid, "assistant", ai_response)
    save_agent_state(sid, _serializable_agent_state(result_state))
    st.session_state.chat_display.append({"role": "assistant", "content": ai_response})
    st.session_state.agent_state = result_state

    if status_box:
        status_box.update(label="Done.", state="complete")

    return ai_response


def show_ai_study_coach():
    if not os.getenv("GROQ_API_KEY"):
        st.markdown(
            '<div class="chat-page-header">'
            '<div class="chat-page-title">AI Study Coach</div>'
            '<div class="chat-page-sub">Configuration required</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.error(
            "**GROQ_API_KEY not found.**\n\n"
            "Add `GROQ_API_KEY=your_key` to a `.env` file in the project root, then restart."
        )
        return

    try:
        from agent import coach_app
    except Exception as e:
        error_id = _log_exception("agent_import", e)
        st.error(
            "The AI Study Coach could not start because a system dependency or configuration failed.\n\n"
            f"Please notify the project owner with this error ID: `{error_id}`."
        )
        return

    _init_coach_session()

    st.markdown(
        """
        <div class="chat-page-header">
            <div class="chat-page-title">AI Study Coach</div>
            <div class="chat-page-sub">
                Ask me anything — performance analysis, concept explanations,
                7-day study plans, or a quick quiz.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.chat_display:
        st.markdown(
            '<div style="font-size:0.82rem; font-weight:700; letter-spacing:0.1em; '
            'text-transform:uppercase; color:var(--text-dim); margin-bottom:0.8rem;">'
            'Try asking…</div>',
            unsafe_allow_html=True,
        )
        starters = [
            ("Analyse my performance", ":material/analytics:",
             "I scored 68 in math and 75 in reading. Am I performing well?"),
            ("Explain a concept", ":material/menu_book:",
             "Can you explain quadratic equations in simple terms?"),
            ("Build a study plan", ":material/event_note:",
             "Give me a 7-day plan to improve my algebra skills."),
            ("Take a quiz", ":material/quiz:",
             "Give me a quiz on trigonometry basics."),
        ]
        cols = st.columns(4)
        for i, (label, icon_val, prompt_text) in enumerate(starters):
            with cols[i]:
                st.markdown('<div class="starter-chip">', unsafe_allow_html=True)
                if st.button(label, icon=icon_val, key=f"starter_{i}", use_container_width=True):
                    try:
                        with st.status("Starting AI Study Coach...", expanded=True) as status:
                            _handle_user_message(prompt_text, coach_app, status)
                    except Exception as e:
                        error_id = _log_exception("starter_prompt", e)
                        msg = _friendly_error_message(error_id)
                        _save_assistant_failure(st.session_state.session_id, msg)
                        st.error(msg)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    for entry in st.session_state.chat_display:
        with st.chat_message(entry["role"]):
            shown = (
                polish_assistant_markdown(entry["content"])
                if entry["role"] == "assistant"
                else entry["content"]
            )
            st.markdown(shown)

    user_input = st.chat_input("Message your AI Study Coach…")

    if user_input:
        try:
            with st.status("Starting AI Study Coach...", expanded=True) as status:
                _handle_user_message(user_input, coach_app, status)
        except Exception as e:
            error_id = _log_exception("chat_message", e)
            msg = _friendly_error_message(error_id)
            _save_assistant_failure(st.session_state.session_id, msg)
            st.error(msg)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

if "current_page" not in st.session_state:
    st.session_state.current_page = "dashboard"

_sidebar_nav()

if st.session_state.current_page == "dashboard":
    show_dashboard()
else:
    show_ai_study_coach()
