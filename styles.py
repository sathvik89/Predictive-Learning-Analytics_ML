"""
styles.py – Global styling and theme configuration for the Streamlit app.
Matches Milestone 1 design language exactly.
"""

import streamlit as st


def apply_global_styles():
    """
    Apply all global CSS styles and theme configuration.
    Call once at app startup.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=DM+Sans:wght@300;400;500;600;700&display=swap');

    :root {
        --bg:        #f5f0e8;
        --bg-deep:   #ede8de;
        --bg-card:   #faf7f2;
        --accent:    #b07d4e;
        --accent-lt: #d4a574;
        --accent-dk: #8d6035;
        --text:      #4a3621;
        --text-dim:  #8b7b68;
        --border:    #e8ddc8;
        --shadow:    0 2px 20px rgba(100,70,30,0.06);
        --radius:    14px;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-size: 15px !important;
    }

    /* Hide chrome only — NOT stHeader: that region contains the sidebar collapse/expand control */
    #MainMenu, footer { visibility: hidden; }

    /* Fix Streamlit default header/icons to be brown instead of black */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    header[data-testid="stHeader"] svg, 
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] svg {
        fill: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* ── SIDEBAR ─────────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-deep) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    /* Remove Streamlit's default top gap inside the sidebar so our heading sits flush */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
    }
    /* The inner block that wraps all sidebar widgets */
    section[data-testid="stSidebar"] > div > div > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Nav buttons in sidebar */
    .stSidebar button {
        width: 100% !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.2rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em !important;
        text-align: left !important;
        margin-bottom: 0.4rem !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--shadow) !important;
    }
    .stSidebar button:hover {
        background: var(--accent) !important;
        color: white !important;
        border-color: var(--accent) !important;
        transform: translateX(3px) !important;
    }
    .stSidebar button:focus, .stSidebar button:active {
        box-shadow: none !important;
        outline: none !important;
    }

    /* Dashboard sub-nav specific styling */
    .st-key-dash_home button,
    .st-key-dash_perf button,
    .st-key-dash_pred button,
    .st-key-nav_dash button,
    .st-key-nav_coach button {
        text-align: center !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 0.5rem !important;
        padding: 1.1rem !important;
        border-radius: 14px !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.08em !important;
    }
    .st-key-dash_home button:hover,
    .st-key-dash_perf button:hover,
    .st-key-dash_pred button:hover,
    .st-key-nav_dash button:hover,
    .st-key-nav_coach button:hover {
        transform: translateY(-2px) !important;
    }

    /* Active nav button state */
    .nav-btn-active button {
        background: var(--accent) !important;
        color: white !important;
        border-color: var(--accent) !important;
    }
    .nav-btn-active button:hover {
        background: var(--accent-dk) !important;
        border-color: var(--accent-dk) !important;
    }

    /* ── MAIN LAYOUT ─────────────────────────────────────────────────────────── */
    .stApp { background-color: var(--bg) !important; }
    .block-container { padding: 2rem 3rem !important; }

    /* ── INPUTS ──────────────────────────────────────────────────────────────── */
    div[data-baseweb="slider"] > div > div > div { background-color: var(--accent) !important; }
    .stSlider [role="slider"] {
        background-color: var(--accent) !important;
        border: 2px solid white !important;
    }
    div[data-baseweb="select"] > div {
        background: white !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="select"] > div:hover { border-color: var(--accent) !important; }
    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(176,125,78,0.2) !important;
    }
    div[data-baseweb="select"] * { color: var(--text) !important; }
    div[data-baseweb="select"] svg { fill: var(--accent) !important; }
    .stRadio [data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
    label[data-testid="stWidgetLabel"] p {
        color: var(--text-dim) !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }

    /* ── BUTTONS ─────────────────────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        border: none !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        padding: 0.9rem 1.5rem !important;
        font-size: 0.85rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--accent-dk) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(176,125,78,0.3) !important;
    }
    .stButton > button:not([kind="primary"]) {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* ── CARDS ───────────────────────────────────────────────────────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1.2rem !important;
    }

    .intel-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 2rem 1.8rem;
        box-shadow: var(--shadow);
        transition: all 0.2s ease;
        height: 100%;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
    }
    .intel-card:hover {
        box-shadow: 0 8px 32px rgba(100,70,30,0.12);
        transform: translateY(-2px);
    }
    .card-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--text-dim);
        margin-bottom: 0.5rem;
    }
    .card-value {
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1;
    }
    .card-body {
        font-size: 0.88rem;
        color: var(--text-dim);
        line-height: 1.6;
        margin-top: 0.5rem;
    }

    /* ── TYPOGRAPHY HELPERS ──────────────────────────────────────────────────── */
    .page-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.3rem;
        line-height: 1.15;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: var(--text-dim);
        font-weight: 400;
        margin-bottom: 2rem;
    }
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--text);
        margin: 2.5rem 0 1.2rem 0;
        padding-left: 0.8rem;
        border-left: 3px solid var(--accent);
    }
    .hero-strip {
        background: linear-gradient(135deg, #ede0cc 0%, #f5efe4 60%, #e8ddc8 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 2.8rem 3.5rem 3rem 3.5rem;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
    }
    .hero-strip::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(176,125,78,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.85rem 0;
        border-bottom: 1px solid var(--border);
        font-size: 0.92rem;
    }
    .metric-row:last-child { border-bottom: none; }
    .metric-key { color: var(--text-dim); font-weight: 500; }
    .metric-val { color: var(--text); font-weight: 700; }

    .conf-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .conf-table th {
        background: var(--bg-deep);
        color: var(--text-dim);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .conf-table td {
        padding: 1rem;
        text-align: center;
        border: 1px solid var(--border);
        font-weight: 600;
        font-size: 1.05rem;
        color: var(--text);
    }
    .conf-hit  { background: rgba(176,125,78,0.15); }
    .conf-miss { background: var(--bg-card); color: var(--text-dim); }

    .rec-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: var(--radius);
        padding: 1.1rem 1.4rem;
        font-size: 0.88rem;
        color: var(--text);
        line-height: 1.6;
        margin-bottom: 0.8rem;
        box-shadow: var(--shadow);
    }
    .input-group-header {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    input[type="number"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }
    .metric-card-equal {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.6rem 1.2rem;
        box-shadow: var(--shadow);
        text-align: center;
        height: 160px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .exec-card-equal {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.4rem 1rem;
        box-shadow: var(--shadow);
        text-align: center;
        height: 130px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    /* ══════════════════════════════════════════════════════════════════════════
       AI STUDY COACH – CHAT PAGE STYLES
    ══════════════════════════════════════════════════════════════════════════ */

    /* Page header banner */
    .chat-page-header {
        background: linear-gradient(135deg, #ede0cc 0%, #f5efe4 60%, #e8ddc8 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .chat-page-header::before {
        content: '🤖';
        position: absolute;
        right: 2.5rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 3.5rem;
        opacity: 0.18;
    }
    .chat-page-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--text);
        margin: 0 0 0.3rem 0;
    }
    .chat-page-sub {
        font-size: 0.9rem;
        color: var(--text-dim);
        margin: 0;
    }

    /* ── Message bubbles ─────────────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.3rem 0 !important;
        opacity: 1 !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
        color: var(--text) !important;
        opacity: 1 !important;
    }
    [data-testid="stChatMessage"][data-role="user"] [data-testid="stMarkdownContainer"] {
        background: var(--accent) !important;
        color: white !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 0.9rem 1.2rem !important;
        max-width: 78% !important;
        margin-left: auto !important;
        font-size: 0.92rem !important;
        line-height: 1.6 !important;
        box-shadow: 0 2px 12px rgba(176,125,78,0.25) !important;
    }
    [data-testid="stChatMessage"][data-role="user"] [data-testid="stMarkdownContainer"] * {
        color: white !important;
    }
    [data-testid="stChatMessage"][data-role="assistant"] [data-testid="stMarkdownContainer"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 1rem 1.3rem !important;
        max-width: 85% !important;
        margin-right: auto !important;
        font-size: 0.9rem !important;
        line-height: 1.7 !important;
        box-shadow: var(--shadow) !important;
        color: var(--text) !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        padding: 1rem 1.2rem !important;
        max-width: 88% !important;
        box-shadow: var(--shadow) !important;
    }
    [data-testid="stChatMessage"][data-role="assistant"] [data-testid="stMarkdownContainer"] p {
        margin: 0 0 0.75em 0 !important;
    }
    [data-testid="stChatMessage"][data-role="assistant"] [data-testid="stMarkdownContainer"] p:last-child {
        margin-bottom: 0 !important;
    }
    [data-testid="stChatMessage"][data-role="assistant"] [data-testid="stMarkdownContainer"] h3 {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.05rem !important;
        margin: 0.9em 0 0.45em 0 !important;
        color: var(--text) !important;
    }
    [data-testid="stChatMessage"][data-role="assistant"] [data-testid="stMarkdownContainer"] ul,
    [data-testid="stChatMessage"][data-role="assistant"] [data-testid="stMarkdownContainer"] ol {
        margin: 0.35em 0 0.75em 0 !important;
        padding-left: 1.25rem !important;
    }
    [data-testid="stChatMessage"][data-role="assistant"] [data-testid="stMarkdownContainer"] li {
        margin-bottom: 0.35em !important;
    }
    [data-testid="stChatMessage"][data-role="assistant"] [data-testid="stMarkdownContainer"] hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1rem 0 !important;
    }
    [data-testid="stChatMessage"][data-role="assistant"] [data-testid="stMarkdownContainer"] code {
        background: rgba(176,125,78,0.12) !important;
        padding: 0.12em 0.35em !important;
        border-radius: 4px !important;
        font-size: 0.88em !important;
    }

    /* ── FIX 2: Chat input box – NO BLACK, NO GREY ──────────────────────────── */
    /* This targets the full-width fixed container that usually sticks as a black bar */
    .stChatInput {
        background-color: var(--bg) !important;
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        border-top: 1px solid var(--border) !important;
    }

    /* The input box wrapper - switching from dark grey to Cream */
    [data-testid="stChatInputContainer"] {
        background-color: var(--bg-card) !important;
        border: 2px solid var(--accent) !important;
        border-radius: 16px !important;
        padding: 0.4rem 0.8rem !important;
        box-shadow: 0 4px 20px rgba(100,70,30,0.08) !important;
    }

    /* Target the textarea background and text color directly */
    [data-testid="stChatInputContainer"] textarea {
        background-color: transparent !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        caret-color: var(--accent) !important;
    }

    /* Ensure placeholder is brown-tinted */
    [data-testid="stChatInputContainer"] textarea::placeholder {
        color: var(--text-dim) !important;
        opacity: 0.7 !important;
    }

    /* The circular send button - Brown matching the theme */
    [data-testid="stChatInputContainer"] button {
        background-color: var(--accent) !important;
        color: white !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
        margin-bottom: 2px !important;
    }

    [data-testid="stChatInputContainer"] button:hover {
        background-color: var(--accent-dk) !important;
        transform: scale(1.08) !important;
    }

    /* Force the arrow icon inside the button to be white */
    [data-testid="stChatInputContainer"] button svg {
        fill: white !important;
    }

    /* ── FIX 3: st.status "thinking" box – borderless, seamless, dark text ──── */
    /* Streamlit renders st.status as a <details> element */
    details {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0.2rem 0 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    details summary {
        background: transparent !important;
        border: none !important;
        list-style: none !important;
        padding: 0.3rem 0 !important;
        cursor: default !important;
    }
    details summary::-webkit-details-marker { display: none !important; }
    details summary,
    details summary *,
    details p,
    details div,
    details span,
    details li {
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.88rem !important;
        background: transparent !important;
        border: none !important;
    }
    /* Streamlit also uses data-testid wrappers */
    [data-testid="stStatusWidget"],
    [data-testid="stStatusWidget"] * {
        color: var(--text) !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* ── Starter prompt chips ─────────────────────────────────────────────────── */
    .starter-chip button {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 50px !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: var(--text-dim) !important;
        transition: all 0.2s !important;
        white-space: nowrap !important;
        width: 100% !important;
    }
    .starter-chip button:hover {
        background: var(--accent) !important;
        color: white !important;
        border-color: var(--accent) !important;
        transform: translateY(-1px) !important;
    }

    /* ── FIX 4: Coach sidebar – sleek past-conversation rows ─────────────────── */

    /* Section label above past conversations */
    .sidebar-section-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin: 1.4rem 0 0.5rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }

    /* Session row buttons — make them look like compact list items, not chunky buttons */
    [data-testid="stSidebar"] div[class*="st-key-open_sess_"] button[kind="secondary"] {
        background: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.45rem 0.7rem !important;
        font-size: 0.79rem !important;
        font-weight: 500 !important;
        color: var(--text-dim) !important;
        text-align: left !important;
        letter-spacing: 0 !important;
        box-shadow: none !important;
        margin-bottom: 0.1rem !important;
        transition: background 0.15s, color 0.15s !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    [data-testid="stSidebar"] div[class*="st-key-open_sess_"] button[kind="secondary"]:hover {
        background: rgba(176,125,78,0.10) !important;
        color: var(--accent) !important;
        transform: none !important;
        border: none !important;
    }

    /* Active session row (primary button) */
    [data-testid="stSidebar"] div[class*="st-key-open_sess_"] button[kind="primary"] {
        background: rgba(176,125,78,0.14) !important;
        border: 1px solid rgba(176,125,78,0.35) !important;
        border-radius: 8px !important;
        padding: 0.45rem 0.7rem !important;
        font-size: 0.79rem !important;
        font-weight: 700 !important;
        color: var(--accent-dk) !important;
        text-align: left !important;
        letter-spacing: 0 !important;
        box-shadow: none !important;
        margin-bottom: 0.1rem !important;
    }
    [data-testid="stSidebar"] div[class*="st-key-open_sess_"] button[kind="primary"]:hover {
        background: rgba(176,125,78,0.22) !important;
        transform: none !important;
    }

    /* ⋮ popover trigger button – tiny and unobtrusive */
    [data-testid="stSidebar"] [data-testid="stPopover"] > button,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"][aria-label="⋮"] {
        background: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.3rem 0.5rem !important;
        font-size: 0.85rem !important;
        color: var(--text-dim) !important;
        box-shadow: none !important;
        min-width: 0 !important;
        width: auto !important;
        margin-bottom: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stPopover"] > button:hover {
        background: rgba(176,125,78,0.10) !important;
        color: var(--accent) !important;
        transform: none !important;
    }

    /* New Conversation button — keep it accent-styled */
    [data-testid="stSidebar"] div.st-key-new_conv button {
        background: var(--accent) !important;
        color: white !important;
        border-color: var(--accent) !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] div.st-key-new_conv button:hover {
        background: var(--accent-dk) !important;
    }

    /* Quiz progress bar */
    .quiz-progress-bar {
        height: 5px;
        background: var(--border);
        border-radius: 3px;
        overflow: hidden;
        margin-top: 0.4rem;
    }
    .quiz-progress-fill {
        height: 100%;
        background: var(--accent);
        border-radius: 3px;
        transition: width 0.4s ease;
    }

    /* Quiz active badge */
    .quiz-active-badge {
        display: inline-block;
        background: rgba(176,125,78,0.15);
        color: var(--accent);
        border: 1px solid rgba(176,125,78,0.3);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
    }
    </style>
    """, unsafe_allow_html=True)