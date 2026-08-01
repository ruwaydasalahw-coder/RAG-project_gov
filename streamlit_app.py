"""
streamlit_app_premium.py

Knowledge Transfer Between Nations — Cross-Lingual RAG Framework
Enterprise console frontend.

IMPORTANT: This file only redesigns the presentation layer. It calls the
existing, unmodified backend exactly as the original streamlit_app.py did:
    - import_module("07_prompting") for rag.answer_question(), rag.OPENROUTER_*
    - import_module("01_documents") / ("03_chunking") / ("04_vector_representation")
      / ("06_retrieve_context") are used READ-ONLY, to introspect real counts
      and real configuration for display (document/chunk/country counts,
      embedding model name, hybrid-search alpha, etc). No pipeline, retrieval,
      embedding, or prompting logic is defined, changed, or duplicated here.
"""


from __future__ import annotations


import time
from importlib import import_module

import streamlit as st
import os

# التحقق من وجود chroma_db وبنائها تلقائياً إذا كانت مفقودة
if not os.path.exists("chroma_db") or len(os.listdir("chroma_db")) == 0:
    os.system("python 05_create_chroma_store.py")


# =============================================================================
# PAGE CONFIG (must be the first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="Knowledge Transfer Between Nations",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# st.markdown patch: CommonMark terminates a raw-HTML block (e.g. <div>...)
# at the first blank/whitespace-only line, which breaks multi-line, indented
# HTML built from Python triple-quoted f-strings (indentation + blank lines
# between concatenated fragments read as "end of HTML block" to the parser).
# <style>/<script> blocks are exempt from that rule, which is why CSS
# injection worked untouched. This wrapper flattens HTML payloads (strips
# per-line indentation, drops blank lines) before handing them to Streamlit,
# so every unsafe_allow_html=True call in this file renders correctly
# regardless of how it's pretty-printed in the source.
# -----------------------------------------------------------------------------
_st_markdown = st.markdown


def _flatten_html(body: str) -> str:
    lines = [line.strip() for line in body.split("\n")]
    return "\n".join(line for line in lines if line != "")


def _patched_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str) and "<" in body:
        body = _flatten_html(body)
    return _st_markdown(body, *args, **kwargs)


st.markdown = _patched_markdown

# =============================================================================
# DESIGN TOKENS
# =============================================================================
INK = "#10142B"
INK_2 = "#181D3D"
INK_LINE = "#2B315C"
PAPER = "#F5F6FA"
SURFACE = "#FFFFFF"
BRASS = "#C6963E"
BRASS_DEEP = "#9C7327"
BRASS_SOFT = "#F3E7CE"
SIGNAL = "#4C5FD5"
SIGNAL_SOFT = "#EEF0FC"
TEXT = "#12172B"
MUTED = "#667085"
LINE = "#E6E8F0"
SUCCESS = "#1F9D6C"
SUCCESS_SOFT = "#E7F7EF"
DANGER = "#C23B3B"
DANGER_SOFT = "#FBEAEA"

# Country / organization display metadata. Purely presentational (flag +
# grouping label) — does not alter which countries exist; that still comes
# entirely from the real backend data.
COUNTRY_META = {
    "estonia": {"flag": "https://flagcdn.com/w40/ee.png", "label": "Estonia", "kind": "Country"},
    "singapore": {"flag": "https://flagcdn.com/w40/sg.png", "label": "Singapore", "kind": "Country"},
    "south korea": {"flag": "https://flagcdn.com/w40/kr.png", "label": "South Korea", "kind": "Country"},
    "oecd": {"flag": "https://static.cdnlogo.com/logos/o/67/oecd_thumb.png", "label": "OECD", "kind": "Organization"},
    "un": {"flag": "https://flagcdn.com/w40/un.png", "label": "United Nations", "kind": "Organization"},
    "united nations": {"flag": "https://flagcdn.com/w40/un.png", "label": "United Nations", "kind": "Organization"},
}

FUTURE_DOMAINS = [
    ("💧", "Water Management"),
    ("🎓", "Education"),
    ("🌾", "Agriculture"),
    ("⚡", "Energy"),
    ("🏥", "Healthcare"),
]

NAV_ITEMS = [
    ("dashboard", "◆", "Dashboard"),
    ("ask", "◈", "Ask a Question"),
    ("sources", "◇", "Knowledge Sources"),
    ("pipeline", "◎", "Pipeline & System"),
]


def country_meta(raw_name: str) -> dict:
    key = (raw_name or "").strip().lower()
    # Backend may send ISO-2 codes instead of full names; map them onto the
    # existing COUNTRY_META keys before lookup (no new COUNTRY_META entries).
    iso2_to_key = {"ee": "estonia", "sg": "singapore", "kr": "south korea"}
    key = iso2_to_key.get(key, key)
    if key in COUNTRY_META:
        return COUNTRY_META[key]
    return {"flag": "🌐", "label": (raw_name or "Unknown").title(), "kind": "Source"}


# -----------------------------------------------------------------------------
# CHANGE: COUNTRY_META["flag"] values can now be either a plain emoji (e.g.
# "🇪🇪") or an image URL (SVG/PNG, e.g. flagcdn.com or a logo URL). Every place
# that renders `meta['flag']` used to just interpolate it as text, so URLs
# showed up as literal strings instead of images. These two helpers are the
# only new logic added: they detect a URL and turn it into an <img> tag,
# while leaving emoji values completely untouched.
# -----------------------------------------------------------------------------
def _is_image_url(value: str) -> bool:
    """
    True if the flag/logo value is an image reference (a remote http(s) URL
    or a local file path) rather than an emoji/text.

    Detected by file extension rather than only an http(s) prefix, so this
    covers both remote URLs (e.g. flagcdn.com, static.cdnlogo.com) and local
    paths (e.g. "assets/oecd.png") the same way.
    """
    if not isinstance(value, str):
        return False
    lowered = value.strip().lower()
    return lowered.endswith((".png", ".jpg", ".jpeg", ".svg", ".webp"))


def flag_markup(flag_value: str, size: str = "1em") -> str:
    """
    Return HTML for a COUNTRY_META['flag'] value.

    - Emoji/plain text (e.g. "🇪🇪", "🏛️") is returned unchanged, exactly as
      before.
    - Image URLs (SVG or PNG both work fine in an <img> tag) are wrapped in
      an <img> tag sized via `size`, so they render as images instead of raw
      URL text. This is the only change needed to fix the display issue.
    """
    if _is_image_url(flag_value):
        return f'<img src="{flag_value}" style="height:{size}; width:auto; vertical-align:middle;">'
    return flag_value


# =============================================================================
# GLOBAL CSS
# =============================================================================
def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,480;9..144,560;9..144,640&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {{
            --ink: {INK};
            --ink-2: {INK_2};
            --ink-line: {INK_LINE};
            --paper: {PAPER};
            --surface: {SURFACE};
            --brass: {BRASS};
            --brass-deep: {BRASS_DEEP};
            --brass-soft: {BRASS_SOFT};
            --signal: {SIGNAL};
            --signal-soft: {SIGNAL_SOFT};
            --text: {TEXT};
            --muted: {MUTED};
            --line: {LINE};
            --success: {SUCCESS};
            --success-soft: {SUCCESS_SOFT};
            --danger: {DANGER};
            --danger-soft: {DANGER_SOFT};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        /* ---------- base canvas ---------- */
        [data-testid="stAppViewContainer"] {{
            background: var(--paper);
        }}
        [data-testid="stHeader"] {{
            background: transparent;
            height: 0;
        }}
        #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{
            visibility: hidden;
            height: 0;
        }}
        .main .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 4rem;
            max-width: 1180px;
        }}

        * {{
            scroll-behavior: smooth;
        }}
        :focus-visible {{
            outline: 2px solid var(--signal) !important;
            outline-offset: 2px !important;
            border-radius: 6px;
        }}
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                animation-duration: 0.001ms !important;
                transition-duration: 0.001ms !important;
            }}
        }}

        h1, h2, h3, .display-font {{
            font-family: 'Fraunces', Georgia, serif;
            color: var(--text);
            letter-spacing: -0.01em;
        }}

        /* ---------- sidebar ---------- */
        [data-testid="stSidebar"] {{
            background: linear-gradient(185deg, var(--ink) 0%, var(--ink-2) 100%);
            border-right: 1px solid var(--ink-line);
        }}
        [data-testid="stSidebar"] * {{
            color: #E7E8F5;
        }}
        [data-testid="stSidebar"] .block-container {{
            padding-top: 1.4rem;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: var(--ink-line);
            margin: 1.1rem 0;
        }}

        .brand-row {{
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.2rem 0.1rem 0.4rem 0.1rem;
        }}
        .brand-title {{
            font-family: 'Fraunces', serif;
            font-size: 1.02rem;
            font-weight: 560;
            line-height: 1.15;
            color: #FFFFFF;
            margin: 0;
        }}
        .brand-subtitle {{
            font-size: 0.68rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--brass);
            margin: 0.1rem 0 0 0;
        }}

        .nav-eyebrow {{
            font-size: 0.66rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #8B90B8;
            margin: 1.0rem 0 0.35rem 0.15rem;
            font-weight: 600;
        }}

        [data-testid="stSidebar"] .stButton > button {{
            width: 100%;
            text-align: left;
            background: transparent;
            border: 1px solid transparent;
            color: #C7CAE6;
            font-weight: 500;
            font-size: 0.86rem;
            padding: 0.5rem 0.7rem;
            border-radius: 9px;
            transition: all 0.15s ease;
            box-shadow: none;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            background: rgba(198, 150, 62, 0.12);
            border-color: rgba(198, 150, 62, 0.35);
            color: #FFFFFF;
        }}
        [data-testid="stSidebar"] .stButton > button:focus {{
            box-shadow: none;
        }}
        [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] div:has(> .nav-active-marker) .stButton > button {{
            background: rgba(198, 150, 62, 0.16);
            border-color: var(--brass);
            color: #FFFFFF;
        }}

        .domain-chip {{
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--ink-line);
            border-radius: 12px;
            padding: 0.65rem 0.75rem;
            margin-top: 0.4rem;
        }}
        .domain-chip .chip-icon {{ font-size: 1.15rem; }}
        .domain-chip .chip-label {{
            font-size: 0.82rem;
            font-weight: 600;
            color: #FFFFFF;
            margin: 0.25rem 0 0.1rem 0;
        }}
        .domain-chip .chip-sub {{
            font-size: 0.72rem;
            color: #9296C0;
            line-height: 1.3;
        }}

        .future-row {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            color: #9296C0;
            padding: 0.28rem 0.15rem;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.32rem 0.65rem;
            border-radius: 999px;
        }}
        .status-dot {{
            width: 7px; height: 7px; border-radius: 50%;
            display: inline-block;
        }}
        .status-ok {{ background: rgba(31,157,108,0.15); color: #7FE3B4; }}
        .status-ok .status-dot {{ background: #3FE0A0; box-shadow: 0 0 6px #3FE0A0; }}
        .status-bad {{ background: rgba(194,59,59,0.15); color: #F0A3A3; }}
        .status-bad .status-dot {{ background: #E45B5B; }}

        /* ---------- animations ---------- */
        @keyframes fadeSlideUp {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }} to {{ opacity: 1; }}
        }}
        @keyframes slowSpin {{
            from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }}
        }}
        @keyframes pulseGlow {{
            0%, 100% {{ opacity: 0.55; }} 50% {{ opacity: 1; }}
        }}
        .anim-in {{ animation: fadeSlideUp 0.55s cubic-bezier(.21,.85,.36,1) both; }}
        .anim-in-1 {{ animation-delay: 0.05s; }}
        .anim-in-2 {{ animation-delay: 0.12s; }}
        .anim-in-3 {{ animation-delay: 0.19s; }}
        .anim-in-4 {{ animation-delay: 0.26s; }}
        .mark-spin {{ animation: slowSpin 38s linear infinite; transform-origin: center; }}
        .mark-pulse {{ animation: pulseGlow 2.6s ease-in-out infinite; }}

        /* ---------- hero ---------- */
        .hero {{
            background: radial-gradient(120% 140% at 15% 0%, #1D2454 0%, var(--ink) 55%, #0B0E22 100%);
            border-radius: 22px;
            padding: 2.6rem 2.6rem 2.3rem 2.6rem;
            position: relative;
            overflow: hidden;
            border: 1px solid var(--ink-line);
        }}
        .hero::after {{
            content: "";
            position: absolute;
            inset: 0;
            background-image: radial-gradient(rgba(198,150,62,0.15) 1px, transparent 1px);
            background-size: 22px 22px;
            opacity: 0.4;
            pointer-events: none;
        }}
        .hero-eyebrow {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--brass);
            margin-bottom: 0.9rem;
            position: relative;
        }}
        .hero-title {{
            font-family: 'Fraunces', serif;
            font-weight: 560;
            font-size: 2.5rem;
            line-height: 1.08;
            color: #FFFFFF;
            margin: 0 0 0.9rem 0;
            max-width: 700px;
            position: relative;
        }}
        .hero-sub {{
            font-size: 1.02rem;
            line-height: 1.55;
            color: #C4C8EA;
            max-width: 620px;
            margin: 0;
            position: relative;
        }}

        /* ---------- section headers ---------- */
        .section-eyebrow {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--brass-deep);
            margin: 0 0 0.35rem 0;
            font-weight: 600;
        }}
        .section-title {{
            font-family: 'Fraunces', serif;
            font-size: 1.55rem;
            font-weight: 560;
            color: var(--text);
            margin: 0 0 0.2rem 0;
        }}
        .section-desc {{
            color: var(--muted);
            font-size: 0.92rem;
            margin: 0 0 1.1rem 0;
        }}

        /* ---------- stat cards ---------- */
        .stat-card {{
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.15rem 1.3rem;
            transition: all 0.2s ease;
            height: 100%;
        }}
        .stat-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 12px 28px -12px rgba(16,20,43,0.18);
            border-color: rgba(198,150,62,0.4);
        }}
        .stat-label {{
            font-size: 0.68rem;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: var(--muted);
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        .stat-value {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.9rem;
            font-weight: 500;
            color: var(--text);
            line-height: 1;
        }}
        .stat-sub {{
            font-size: 0.78rem;
            color: var(--muted);
            margin-top: 0.45rem;
        }}

        /* ---------- generic card ---------- */
        .card {{
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            transition: all 0.2s ease;
        }}
        .card:hover {{
            box-shadow: 0 14px 30px -16px rgba(16,20,43,0.16);
        }}

        /* ---------- country cards ---------- */
        .country-card {{
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.1rem 1.1rem 1.0rem 1.1rem;
            text-align: center;
            transition: all 0.2s cubic-bezier(.21,.85,.36,1);
            height: 100%;
        }}
        .country-card:hover {{
            transform: translateY(-4px);
            border-color: var(--brass);
            box-shadow: 0 16px 32px -16px rgba(198,150,62,0.35);
        }}
        .country-flag {{ font-size: 2.1rem; line-height: 1; margin-bottom: 0.55rem; }}
        .country-name {{ font-weight: 650; color: var(--text); font-size: 0.94rem; }}
        .country-kind {{
            font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.07em;
            color: var(--brass-deep); font-weight: 600; margin: 0.15rem 0 0.5rem 0;
        }}
        .country-count {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.76rem; color: var(--muted);
            border-top: 1px dashed var(--line);
            padding-top: 0.5rem; margin-top: 0.15rem;
        }}

        /* ---------- pipeline timeline ---------- */
        .pipeline-wrap {{ position: relative; padding-left: 2px; }}
        .pipeline-step {{
            display: flex;
            gap: 1rem;
            position: relative;
            padding-bottom: 1.7rem;
        }}
        .pipeline-step:last-child {{ padding-bottom: 0; }}
        .pipeline-step::before {{
            content: "";
            position: absolute;
            left: 17px; top: 38px; bottom: -2px;
            width: 2px;
            background: linear-gradient(var(--line), var(--line));
        }}
        .pipeline-step:last-child::before {{ display: none; }}
        .pipeline-node {{
            flex-shrink: 0;
            width: 36px; height: 36px;
            border-radius: 50%;
            background: var(--signal-soft);
            border: 1.5px solid var(--signal);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.0rem;
            z-index: 1;
        }}
        .pipeline-node.brass {{
            background: var(--brass-soft);
            border-color: var(--brass);
        }}
        .pipeline-title {{
            font-weight: 650;
            color: var(--text);
            font-size: 0.94rem;
            margin-bottom: 0.15rem;
        }}
        .pipeline-desc {{
            font-size: 0.82rem;
            color: var(--muted);
            line-height: 1.4;
        }}
        .pipeline-tag {{
            display: inline-block;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            color: var(--brass-deep);
            background: var(--brass-soft);
            border-radius: 5px;
            padding: 0.08rem 0.4rem;
            margin-top: 0.35rem;
        }}

        /* ---------- badges / chips ---------- */
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.22rem 0.6rem;
            border-radius: 999px;
            background: var(--signal-soft);
            color: var(--signal);
            margin: 0 0.3rem 0.3rem 0;
        }}
        .badge.gold {{ background: var(--brass-soft); color: var(--brass-deep); }}

        .example-chip {{
            display: inline-block;
            font-size: 0.82rem;
            color: var(--text);
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.4rem 0.9rem;
            margin: 0 0.4rem 0.5rem 0;
            transition: all 0.15s ease;
        }}

        /* ---------- answer card ---------- */
        .answer-card {{
            background: linear-gradient(180deg, #FFFFFF 0%, #FCFBF7 100%);
            border: 1px solid var(--line);
            border-left: 4px solid var(--brass);
            border-radius: 14px;
            padding: 1.5rem 1.7rem;
            font-size: 1.0rem;
            line-height: 1.65;
            color: var(--text);
        }}
        .answer-label {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--brass-deep);
            font-weight: 600;
            margin-bottom: 0.7rem;
        }}

        .source-card {{
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1.0rem 1.15rem;
            height: 100%;
            transition: all 0.2s ease;
        }}
        .source-card:hover {{
            border-color: var(--signal);
            box-shadow: 0 10px 24px -14px rgba(76,95,213,0.35);
        }}
        .source-head {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.55rem;
        }}
        .source-num {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            color: var(--signal);
            background: var(--signal-soft);
            border-radius: 5px;
            padding: 0.1rem 0.4rem;
        }}
        .source-country {{ font-weight: 650; font-size: 0.86rem; color: var(--text); }}
        .source-title {{ font-size: 0.76rem; color: var(--muted); margin-bottom: 0.5rem; }}
        .source-snippet {{
            font-size: 0.82rem;
            color: #3C4257;
            line-height: 1.5;
            max-height: 6.2em;
            overflow: hidden;
        }}

        /* ---------- footer ---------- */
        .app-footer {{
            margin-top: 3rem;
            padding-top: 1.4rem;
            border-top: 1px solid var(--line);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.78rem;
            color: var(--muted);
            flex-wrap: wrap;
            gap: 0.6rem;
        }}

        /* ---------- misc streamlit overrides ---------- */
        .stTextArea textarea {{
            border-radius: 12px !important;
            border: 1.5px solid var(--line) !important;
            font-size: 0.98rem !important;
            padding: 0.9rem 1rem !important;
        }}
        .stTextArea textarea:focus {{
            border-color: var(--signal) !important;
            box-shadow: 0 0 0 3px rgba(76,95,213,0.14) !important;
        }}
        div[data-testid="stTextArea"] label {{
            font-weight: 600;
            color: var(--text);
        }}
        .main .stButton > button[kind="primary"] {{
            background: var(--ink);
            border: 1px solid var(--ink);
            color: #FFFFFF;
            font-weight: 600;
            border-radius: 10px;
            padding: 0.55rem 1.4rem;
            transition: all 0.15s ease;
        }}
        .main .stButton > button[kind="primary"]:hover {{
            background: var(--brass-deep);
            border-color: var(--brass-deep);
            transform: translateY(-1px);
            box-shadow: 0 8px 18px -8px rgba(156,115,39,0.55);
        }}
        .main .stButton > button:not([kind="primary"]) {{
            border-radius: 10px;
            border: 1px solid var(--line);
            font-weight: 500;
            color: var(--text);
        }}
        [data-testid="stExpander"] {{
            border: 1px solid var(--line) !important;
            border-radius: 12px !important;
            background: var(--surface);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SIGNATURE SVG MARK — a meridian / great-circle emblem
# (three nodes connected by arcing "knowledge routes", evoking cross-border
# exchange rather than a generic globe icon)
# =============================================================================
def brand_mark(size: int = 34, spin: bool = True, color: str = "#C6963E") -> str:
    spin_class = "mark-spin" if spin else ""
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="50" cy="50" r="47" stroke="{color}" stroke-width="1.4" opacity="0.35"/>
        <g class="{spin_class}">
            <path d="M15 60 Q50 10 85 55" stroke="{color}" stroke-width="1.6" fill="none" opacity="0.85"/>
            <path d="M20 30 Q50 85 80 35" stroke="{color}" stroke-width="1.6" fill="none" opacity="0.6"/>
            <path d="M10 45 Q50 50 90 48" stroke="{color}" stroke-width="1.6" fill="none" opacity="0.45"/>
        </g>
        <circle cx="15" cy="60" r="3.4" fill="{color}"/>
        <circle cx="85" cy="55" r="3.4" fill="{color}"/>
        <circle cx="50" cy="16" r="3.4" fill="{color}"/>
        <circle cx="50" cy="50" r="4.2" fill="{color}"/>
    </svg>
    """


# =============================================================================
# BACKEND INTROSPECTION (read-only — no pipeline logic defined here)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_rag():
    """Identical contract to the original app: import the real pipeline
    module. Raises RuntimeError if the Chroma store isn't built yet — the
    exact same failure mode the original frontend handled."""
    return import_module("07_prompting")


@st.cache_data(show_spinner=False)
def load_corpus_stats():
    """Read-only introspection of the real, already-built corpus. Returns
    counts only — never duplicates retrieval/embedding logic. Falls back to
    an empty-but-valid shape if the corpus isn't built yet, so the Dashboard
    and Knowledge Sources pages can still render a helpful empty state."""
    try:
        documents = import_module("01_documents").get_documents()
        chunks = import_module("03_chunking").build_chunks()
    except Exception as error:
        return {"ok": False, "error": str(error), "documents": [], "chunks": [], "by_country": {}}

    by_country = {}
    for doc in documents:
        key = (doc.get("country") or "unknown").strip().lower()
        by_country.setdefault(key, {"documents": 0, "chunks": 0, "raw_name": doc.get("country", "Unknown")})
        by_country[key]["documents"] += 1
    for chunk in chunks:
        key = (chunk.get("country") or "unknown").strip().lower()
        by_country.setdefault(key, {"documents": 0, "chunks": 0, "raw_name": chunk.get("country", "Unknown")})
        by_country[key]["chunks"] += 1

    return {
        "ok": True,
        "error": None,
        "documents": documents,
        "chunks": chunks,
        "by_country": by_country,
    }


@st.cache_data(show_spinner=False)
def load_system_config():
    """Read-only introspection of real configuration constants already
    defined in the backend (embedding model name, fusion weight, LLM model,
    etc). Nothing here is invented — every value is pulled from the actual
    module attributes."""
    config = {
        "embedding_model": None,
        "alpha": None,
        "country_boost": None,
        "llm_model": None,
        "api_key_set": False,
        "ground_truth_count": None,
        "error": None,
    }
    try:
        vectors = import_module("04_vector_representation")
        config["embedding_model"] = getattr(vectors, "MODEL_NAME", None)
        config["alpha"] = getattr(vectors, "ALPHA", None)
    except Exception as error:
        config["error"] = str(error)

    try:
        retrieve = import_module("06_retrieve_context")
        config["country_boost"] = getattr(retrieve, "COUNTRY_BOOST", None)
    except Exception:
        pass

    try:
        rag = import_module("07_prompting")
        config["llm_model"] = getattr(rag, "OPENROUTER_MODEL", None)
        config["api_key_set"] = bool(getattr(rag, "OPENROUTER_API_KEY", None))
        config["ground_truth_count"] = len(getattr(rag, "GROUND_TRUTH", []))
    except Exception:
        pass

    return config


# =============================================================================
# SESSION STATE
# =============================================================================
def init_state():
    st.session_state.setdefault("page", "dashboard")
    st.session_state.setdefault("history", [])  # list of dicts: question, answer, sources, seconds
    st.session_state.setdefault("pending_question", None)


def go_to(page_key: str, prefill_question: str | None = None):
    st.session_state["page"] = page_key
    if prefill_question is not None:
        st.session_state["pending_question"] = prefill_question


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar(stats: dict, config: dict, rag_ok: bool):
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand-row">
                {brand_mark(34)}
                <div>
                    <p class="brand-title">Knowledge Transfer<br/>Between Nations</p>
                    <p class="brand-subtitle">Cross-Lingual RAG Framework</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown('<div class="nav-eyebrow">Console</div>', unsafe_allow_html=True)
        for key, icon, label in NAV_ITEMS:
            active = st.session_state["page"] == key
            if active:
                st.markdown('<div class="nav-active-marker"></div>', unsafe_allow_html=True)
            if st.button(f"{icon}   {label}", key=f"nav_{key}", use_container_width=True):
                go_to(key)
                st.rerun()

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown('<div class="nav-eyebrow">Current Domain</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="domain-chip">
                <div class="chip-icon">🏛️</div>
                <div class="chip-label">Digital Government</div>
                <div class="chip-sub">Sharing digital-transformation practice across governments</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="nav-eyebrow">Future Domains</div>', unsafe_allow_html=True)
        future_html = "".join(
            f'<div class="future-row"><span>{icon}</span><span>{label}</span></div>'
            for icon, label in FUTURE_DOMAINS
        )
        st.markdown(future_html, unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown('<div class="nav-eyebrow">System Status</div>', unsafe_allow_html=True)
        if rag_ok:
            st.markdown(
                '<span class="status-pill status-ok"><span class="status-dot"></span>All systems operational</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill status-bad"><span class="status-dot"></span>Index not ready</span>',
                unsafe_allow_html=True,
            )
        if config.get("embedding_model"):
            st.markdown(
                f'<div class="future-row" style="margin-top:0.6rem;">🧬 <span>{config["embedding_model"]}</span></div>',
                unsafe_allow_html=True,
            )
        if config.get("llm_model"):
            st.markdown(
                f'<div class="future-row">🤖 <span>{config["llm_model"]}</span></div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# SHARED UI PIECES
# =============================================================================
def section_header(eyebrow: str, title: str, desc: str = ""):
    desc_html = f'<p class="section-desc">{desc}</p>' if desc else ""
    st.markdown(
        f"""
        <div class="anim-in">
            <p class="section-eyebrow">{eyebrow}</p>
            <p class="section-title">{title}</p>
            {desc_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, sub: str = "", delay_class: str = ""):
    st.markdown(
        f"""
        <div class="stat-card anim-in {delay_class}">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def country_card(raw_name: str, doc_count: int, chunk_count: int):
    meta = country_meta(raw_name)
    st.markdown(
        f"""
        <div class="country-card anim-in">
            <div class="country-flag">{flag_markup(meta['flag'], size='2.1rem')}</div>
            <div class="country-name">{meta['label']}</div>
            <div class="country-kind">{meta['kind']}</div>
            <div class="country-count">{doc_count} report{'s' if doc_count != 1 else ''} · {chunk_count} chunks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_timeline(highlight_last: bool = False):
    steps = [
        ("📄", "PDF Ingestion", "Government and institutional reports are parsed per country/organization.", "01_documents.py"),
        ("🧹", "Preprocessing", "Text is cleaned and normalized while protecting entities like IDs and country names.", "02_preprocessing.py"),
        ("🧩", "Chunking", "Documents are split into overlapping passages sized for retrieval precision.", "03_chunking.py"),
        ("🧬", "Embeddings", "Each chunk is encoded into a dense semantic vector.", "04_vector_representation.py"),
        ("🗄️", "Vector Indexing", "Vectors are persisted in ChromaDB alongside BM25 keyword indexing.", "05_create_chroma_store.py"),
        ("🔎", "Hybrid Retrieval", "BM25 and dense similarity are fused, with a country-aware relevance boost.", "06_retrieve_context.py"),
        ("🤖", "Grounded Generation", "An LLM answers strictly from retrieved passages, citing every source used.", "07_prompting.py"),
    ]
    html = '<div class="pipeline-wrap">'
    for i, (icon, title, desc, tag) in enumerate(steps):
        node_cls = "brass" if (highlight_last and i == len(steps) - 1) else ""
        html += f"""
        <div class="pipeline-step anim-in">
            <div class="pipeline-node {node_cls}">{icon}</div>
            <div>
                <div class="pipeline-title">{title}</div>
                <div class="pipeline-desc">{desc}</div>
                <span class="pipeline-tag">{tag}</span>
            </div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_footer(stats_ok: bool):
    st.markdown(
        f"""
        <div class="app-footer">
            <div>{brand_mark(16, spin=False)} &nbsp;Knowledge Transfer Between Nations — Cross-Lingual RAG Framework</div>
            <div>Hybrid Retrieval (BM25 + Dense) · ChromaDB · Grounded Generation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE: DASHBOARD
# =============================================================================
def render_dashboard(stats: dict, config: dict, rag_ok: bool):
    st.markdown(
        f"""
        <div class="hero anim-in">
            <div class="hero-eyebrow">Cross-Lingual Retrieval-Augmented Generation</div>
            <h1 class="hero-title">Knowledge Transfer<br/>Between Nations</h1>
            <p class="hero-sub">
                Ask one question in plain language and get a grounded answer synthesized
                from government and institutional reports across countries — every claim
                traceable back to its exact source passage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    documents = stats.get("documents", [])
    chunks = stats.get("chunks", [])
    by_country = stats.get("by_country", {})
    n_countries = sum(1 for k in by_country if country_meta(k)["kind"] == "Country")
    n_orgs = sum(1 for k in by_country if country_meta(k)["kind"] != "Country")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Knowledge Sources", str(n_countries + n_orgs), f"{n_countries} countries · {n_orgs} organizations", "anim-in-1")
    with c2:
        stat_card("Reports Indexed", str(len(documents)) if stats.get("ok") else "—", "Source PDF documents", "anim-in-2")
    with c3:
        stat_card("Retrievable Chunks", str(len(chunks)) if stats.get("ok") else "—", "Passages available to the retriever", "anim-in-3")
    with c4:
        this_session = len(st.session_state["history"])
        stat_card("Questions Asked", str(this_session), "This session", "anim-in-4")

    st.write("")
    st.write("")

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        section_header("Try It", "Ask anything", "Type a question or start from an example.")
        with st.form("dashboard_ask_form", clear_on_submit=False):
            q = st.text_area(
                "Question",
                placeholder="e.g. How did Estonia build X-Road, and how is it governed?",
                height=100,
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Ask a Question →", type="primary")
        if submitted and q.strip():
            go_to("ask", prefill_question=q.strip())
            st.rerun()

        examples = [
            "How does Estonia's digital identity system work?",
            "What does the OECD recommend for digital government?",
            "How does Singapore approach e-governance?",
        ]
        chip_cols = st.columns(len(examples))
        for col, ex in zip(chip_cols, examples):
            with col:
                if st.button(ex, key=f"dash_ex_{ex}", use_container_width=True):
                    go_to("ask", prefill_question=ex)
                    st.rerun()

        st.write("")
        section_header("Overview", "How the pipeline works", "Seven stages, from raw PDFs to a cited answer.")
        pipeline_timeline()

    with right:
        section_header("Coverage", "Knowledge sources", "")
        if not by_country:
            st.markdown(
                '<div class="card">Index not built yet. Run <code>05_create_chroma_store.py</code> to populate the knowledge base.</div>',
                unsafe_allow_html=True,
            )
        else:
            for key, data in sorted(by_country.items(), key=lambda kv: -kv[1]["documents"]):
                meta = country_meta(key)
                st.markdown(
                    f"""
                    <div class="card anim-in" style="margin-bottom:0.7rem; padding:0.9rem 1.1rem; display:flex; align-items:center; justify-content:space-between;">
                        <div style="display:flex; align-items:center; gap:0.6rem;">
                            <span style="font-size:1.35rem;">{flag_markup(meta['flag'], size='1.35rem')}</span>
                            <div>
                                <div style="font-weight:650; font-size:0.88rem; color:var(--text);">{meta['label']}</div>
                                <div style="font-size:0.72rem; color:var(--muted);">{meta['kind']}</div>
                            </div>
                        </div>
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:var(--brass-deep); font-weight:600;">
                            {data['documents']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("")
        section_header("Recent Activity", "Questions this session", "")
        if not st.session_state["history"]:
            st.markdown(
                '<div class="card" style="color:var(--muted); font-size:0.86rem;">No questions asked yet in this session — try the box on the left.</div>',
                unsafe_allow_html=True,
            )
        else:
            for item in reversed(st.session_state["history"][-4:]):
                st.markdown(
                    f"""
                    <div class="card anim-in" style="margin-bottom:0.6rem; padding:0.8rem 1.0rem;">
                        <div style="font-size:0.85rem; font-weight:600; color:var(--text); margin-bottom:0.2rem;">{item['question']}</div>
                        <div style="font-size:0.72rem; color:var(--muted);">{len(item['sources'])} sources · {item['seconds']:.1f}s</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =============================================================================
# PAGE: ASK A QUESTION
# =============================================================================
def run_query(rag, question: str):
    """Calls the real, unmodified backend. The status steps below narrate
    the real pipeline stages the backend performs inside answer_question();
    they are presentational pacing around one real blocking call, not a
    fabrication of separate calls."""
    with st.status("Running the hybrid RAG pipeline...", expanded=True) as status:
        st.write("🧹 Preprocessing query...")
        time.sleep(0.25)
        st.write("🔎 Hybrid retrieval — BM25 + dense embeddings...")
        time.sleep(0.25)
        st.write("🧭 Applying country-aware relevance boost...")
        time.sleep(0.2)
        st.write("🤖 Generating grounded, cited answer...")
        start = time.time()
        answer, sources = rag.answer_question(question)
        elapsed = time.time() - start
        status.update(label="Answer ready", state="complete", expanded=False)
    return answer, sources, elapsed


def render_ask(rag, rag_ok: bool):
    section_header("Ask", "Ask a question", "Answers are grounded only in retrieved passages and cite every source used.")

    prefill = st.session_state.pop("pending_question", None) or ""

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            value=prefill,
            placeholder="e.g. What is X-Road, and what technical architecture does it use?",
            height=110,
            label_visibility="collapsed",
        )
        col_submit, col_hint = st.columns([1, 3])
        with col_submit:
            submitted = st.form_submit_button("Ask a Question →", type="primary", use_container_width=True)
        with col_hint:
            st.markdown(
                '<div style="color:var(--muted); font-size:0.82rem; padding-top:0.6rem;">Answers cite sources like [Source 1] — expand them below.</div>',
                unsafe_allow_html=True,
            )

    examples = [
        "How does Estonia's digital identity system work?",
        "What does the OECD recommend for digital government?",
        "How does Singapore approach e-governance?",
        "What is X-Road's technical architecture?",
    ]
    st.markdown(
        "".join(f'<span class="example-chip">{e}</span>' for e in examples),
        unsafe_allow_html=True,
    )

    if not rag_ok:
        st.markdown(
            f"""
            <div class="card" style="border-left:4px solid var(--danger); margin-top:1.2rem;">
                <div style="font-weight:650; color:var(--text); margin-bottom:0.3rem;">Knowledge index isn't ready yet</div>
                <div style="color:var(--muted); font-size:0.88rem;">
                    The vector store hasn't been built (or is empty). Run
                    <code>05_create_chroma_store.py</code> to index the corpus, then reload this page.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if submitted and question.strip():
        answer, sources, elapsed = run_query(rag, question.strip())
        st.session_state["history"].append(
            {"question": question.strip(), "answer": answer, "sources": sources, "seconds": elapsed}
        )

    if st.session_state["history"]:
        latest = st.session_state["history"][-1]

        st.write("")
        st.markdown(
            f"""
            <div class="answer-card anim-in">
                <div class="answer-label">Answer &nbsp;·&nbsp; {latest['seconds']:.1f}s</div>
                {latest['answer'].replace(chr(10), '<br/>')}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if latest["sources"]:
            st.write("")
            section_header("Evidence", f"Retrieved sources ({len(latest['sources'])})", "")
            cols = st.columns(min(3, len(latest["sources"])) or 1)
            for i, source in enumerate(latest["sources"]):
                meta = country_meta(source.get("country", ""))
                with cols[i % len(cols)]:
                    st.markdown(
                        f"""
                        <div class="source-card anim-in">
                            <div class="source-head">
                                <span class="source-num">Source {i + 1}</span>
                                <span>{flag_markup(meta['flag'], size='1rem')}</span>
                                <span class="source-country">{meta['label']}</span>
                            </div>
                            <div class="source-title">{source.get('title', '')}</div>
                            <div class="source-snippet">{source.get('chunk_text', '')[:260]}…</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No sources were retrieved above the relevance threshold for this question.")

    if len(st.session_state["history"]) > 1:
        st.write("")
        with st.expander(f"Previous questions this session ({len(st.session_state['history']) - 1})"):
            for item in reversed(st.session_state["history"][:-1]):
                st.markdown(f"**{item['question']}**")
                st.write(item["answer"])
                st.markdown("---")


# =============================================================================
# PAGE: KNOWLEDGE SOURCES
# =============================================================================
def render_sources(stats: dict):
    section_header("Coverage", "Knowledge sources", "Every country and organization currently indexed, with live document and chunk counts.")

    by_country = stats.get("by_country", {})
    if not stats.get("ok"):
        st.markdown(
            f"""
            <div class="card" style="border-left:4px solid var(--danger);">
                <div style="font-weight:650; margin-bottom:0.3rem;">Corpus data isn't available</div>
                <div style="color:var(--muted); font-size:0.88rem;">{stats.get('error', 'Unknown error')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    ordered = sorted(by_country.items(), key=lambda kv: -kv[1]["documents"])
    n_cols = 5
    rows = [ordered[i:i + n_cols] for i in range(0, len(ordered), n_cols)]
    for row in rows:
        cols = st.columns(n_cols)
        for col, (key, data) in zip(cols, row):
            with col:
                country_card(data.get("raw_name", key), data["documents"], data["chunks"])

    st.write("")
    st.write("")
    section_header("Documents", "Full report index", "Every source document currently retrievable, grouped by source.")

    documents = stats.get("documents", [])
    grouped: dict[str, list] = {}
    for doc in documents:
        key = (doc.get("country") or "unknown").strip().lower()
        grouped.setdefault(key, []).append(doc)

    for key, docs in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        meta = country_meta(key)
        count_label = f"{meta['label']} — {len(docs)} report{'s' if len(docs) != 1 else ''}"
        # st.expander() labels are plain text and cannot render an <img> tag,
        # so when the flag is an image URL we render it as a small inline
        # image right above the expander instead, and leave it out of the
        # label text (which stays exactly as before for emoji flags).
        if _is_image_url(meta['flag']):
            st.markdown(flag_markup(meta['flag'], size='1.1rem'), unsafe_allow_html=True)
            expander_label = count_label
        else:
            expander_label = f"{meta['flag']}  {count_label}"
        with st.expander(expander_label):
            for doc in docs:
                st.markdown(
                    f"""
                    <div style="padding:0.5rem 0; border-bottom:1px solid var(--line); font-size:0.86rem;">
                        <span style="font-weight:600; color:var(--text);">{doc.get('title', doc.get('id',''))}</span>
                        <span style="color:var(--muted); font-family:'IBM Plex Mono',monospace; font-size:0.74rem;"> · {len(doc.get('text',''))} chars</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =============================================================================
# PAGE: PIPELINE & SYSTEM
# =============================================================================
def render_pipeline(stats: dict, config: dict, rag_ok: bool):
    section_header("Architecture", "Pipeline & system", "How a question becomes a grounded, cited answer — and the live configuration behind it.")

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        pipeline_timeline(highlight_last=True)

    with right:
        st.markdown(
            f"""
            <div class="card anim-in">
                <div class="section-eyebrow" style="margin-bottom:0.8rem;">Live Configuration</div>
            """,
            unsafe_allow_html=True,
        )
        rows = [
            ("Retriever", "Hybrid — BM25 + Dense"),
            ("Embedding model", config.get("embedding_model") or "—"),
            ("Fusion weight (α)", config.get("alpha") if config.get("alpha") is not None else "—"),
            ("Country boost", config.get("country_boost") if config.get("country_boost") is not None else "—"),
            ("Vector database", "ChromaDB (persistent)"),
            ("Language model", config.get("llm_model") or "—"),
            ("LLM provider", "OpenRouter"),
            ("API key configured", "Yes" if config.get("api_key_set") else "No"),
            ("Ground-truth suite", f"{config.get('ground_truth_count')} questions" if config.get("ground_truth_count") else "—"),
        ]
        rows_html = "".join(
            f"""
            <div style="display:flex; justify-content:space-between; padding:0.5rem 0; border-bottom:1px solid var(--line); font-size:0.85rem;">
                <span style="color:var(--muted);">{label}</span>
                <span style="font-family:'IBM Plex Mono',monospace; font-weight:600; color:var(--text);">{value}</span>
            </div>
            """
            for label, value in rows
        )
        st.markdown(rows_html + "</div>", unsafe_allow_html=True)

        st.write("")
        status_label = "All systems operational" if rag_ok else "Index not ready"
        status_cls = "status-ok" if rag_ok else "status-bad"
        st.markdown(
            f"""
            <div class="card anim-in">
                <div class="section-eyebrow" style="margin-bottom:0.7rem;">System Status</div>
                <span class="status-pill {status_cls}"><span class="status-dot"></span>{status_label}</span>
                <div style="margin-top:0.9rem; font-size:0.82rem; color:var(--muted); line-height:1.5;">
                    {"The vector index is built and the retrieval + generation pipeline is reachable." if rag_ok else
                     "The Chroma vector store is missing or empty. Run <code>05_create_chroma_store.py</code>, then reload."}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# MAIN
# =============================================================================
def main():
    inject_css()
    init_state()

    stats = load_corpus_stats()
    config = load_system_config()

    rag = None
    rag_ok = False
    try:
        rag = load_rag()
        rag_ok = True
    except Exception:
        rag_ok = False

    # Keep original secrets-handling behavior for deployed environments.
    if rag is not None:
        try:
            if not rag.OPENROUTER_API_KEY:
                rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
            rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
        except Exception:
            pass

    render_sidebar(stats, config, rag_ok)

    page = st.session_state["page"]
    if page == "dashboard":
        render_dashboard(stats, config, rag_ok)
    elif page == "ask":
        render_ask(rag, rag_ok)
    elif page == "sources":
        render_sources(stats)
    elif page == "pipeline":
        render_pipeline(stats, config, rag_ok)
    else:
        render_dashboard(stats, config, rag_ok)

    render_footer(stats.get("ok", False))


if __name__ == "__main__":
    main()
