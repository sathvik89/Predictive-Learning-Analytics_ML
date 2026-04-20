"""
chat_history.py - PostgreSQL-backed (Neon) chat sessions and messages.

Drop-in replacement for the SQLite version.
Every function name, parameter, and return type is identical.
Only the internal connection and SQL syntax changed.

Tables created:
  chat_sessions      - one row per conversation
  chat_messages      - all messages for every conversation
  chat_agent_state   - serialised LangGraph state per conversation
"""

import os
import json
from datetime import datetime

import psycopg2
import psycopg2.extras


# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────

def _get_conn():
    """
    Returns a psycopg2 connection using DATABASE_URL.

    Priority:
      1. os.environ["DATABASE_URL"]          (set by python-dotenv / system env)
      2. st.secrets["DATABASE_URL"]          (Streamlit Cloud secrets)

    Format expected:
      postgresql://user:password@host/dbname?sslmode=require
    """
    url = os.environ.get("DATABASE_URL", "").strip()

    if not url:
        # Streamlit secrets fallback (only import streamlit when needed)
        try:
            import streamlit as st
            url = st.secrets.get("DATABASE_URL", "").strip()
        except Exception:
            pass

    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set.\n"
            "• Local dev: add DATABASE_URL=... to your .env file.\n"
            "• Streamlit Cloud: add it under App Settings → Secrets.\n"
            "• Format: postgresql://user:password@host/dbname?sslmode=require"
        )

    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


# ─────────────────────────────────────────────
# SCHEMA INIT
# ─────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables if they do not exist.
    Safe to call on every app startup — uses IF NOT EXISTS throughout.
    Neon (PostgreSQL) equivalent of the SQLite schema in the original file.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:

            # ── chat_sessions ─────────────────────────────────────────────────
            # SQLite used INTEGER PRIMARY KEY AUTOINCREMENT → SERIAL here
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id          SERIAL PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    started_at  TEXT NOT NULL,
                    title       TEXT NOT NULL DEFAULT 'New Conversation'
                );
            """)

            # ── chat_messages ─────────────────────────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id          SERIAL PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON chat_messages(session_id);
            """)

            # ── chat_agent_state ──────────────────────────────────────────────
            # Uses TEXT PRIMARY KEY (same as SQLite version)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_agent_state (
                    session_id  TEXT PRIMARY KEY,
                    state_json  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );
            """)

        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────
# SESSION OPERATIONS
# ─────────────────────────────────────────────

def create_session(session_id: str, title: str = "New Conversation") -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_sessions (session_id, started_at, title) VALUES (%s, %s, %s)",
                (session_id, datetime.utcnow().isoformat(), title),
            )
        conn.commit()
    finally:
        conn.close()


def session_exists(session_id: str) -> bool:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM chat_sessions WHERE session_id = %s",
                (session_id,),
            )
            return cur.fetchone() is not None
    finally:
        conn.close()


def get_session(session_id: str) -> dict | None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT session_id, title, started_at FROM chat_sessions WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def update_session_title(session_id: str, title: str) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET title = %s WHERE session_id = %s",
                (title[:60], session_id),
            )
        conn.commit()
    finally:
        conn.close()


def list_sessions(limit: int = 30) -> list[dict]:
    """Most recently active conversations first (by last message time)."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.session_id,
                       s.title,
                       s.started_at,
                       COUNT(m.id)                                   AS message_count,
                       COALESCE(MAX(m.created_at), s.started_at)    AS last_at
                FROM   chat_sessions  s
                LEFT JOIN chat_messages m ON s.session_id = m.session_id
                GROUP BY s.session_id, s.title, s.started_at
                ORDER BY last_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def delete_session(session_id: str) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_messages    WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM chat_agent_state WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM chat_sessions    WHERE session_id = %s", (session_id,))
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────
# MESSAGE OPERATIONS
# ─────────────────────────────────────────────

def save_message(session_id: str, role: str, content: str) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (%s, %s, %s, %s)",
                (session_id, role, content, datetime.utcnow().isoformat()),
            )
        conn.commit()
    finally:
        conn.close()


def load_messages(session_id: str) -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content, created_at FROM chat_messages WHERE session_id = %s ORDER BY id",
                (session_id,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ─────────────────────────────────────────────
# AGENT STATE OPERATIONS
# ─────────────────────────────────────────────

def save_agent_state(session_id: str, state: dict) -> None:
    """
    Upsert the serialised LangGraph agent state for a session.
    PostgreSQL ON CONFLICT syntax (same as SQLite in this case).
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_agent_state (session_id, state_json, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE
                    SET state_json = EXCLUDED.state_json,
                        updated_at = EXCLUDED.updated_at
                """,
                (session_id, json.dumps(state), datetime.utcnow().isoformat()),
            )
        conn.commit()
    finally:
        conn.close()


def load_agent_state(session_id: str) -> dict | None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT state_json FROM chat_agent_state WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(row["state_json"])
        except json.JSONDecodeError:
            return None
    finally:
        conn.close()