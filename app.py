"""
app.py — Streamlit interface for SEC Filings Analyst

WHAT THIS IS
  A retrieval-augmented Q&A interface over SEC 10-K and 10-Q filings for
  Apple, Alphabet/Google, Meta, Microsoft, and NVIDIA. Covers filings from
  2022 through early 2026.

  Each answer is grounded in actual filing text and includes source citations.
  The system never generates information beyond what is in the retrieved passages.

RUNNING LOCALLY
  streamlit run app.py

DEPLOYMENT (VPS)
  1. Copy data/index/ (the FAISS index) to your server alongside the .py files.
  2. Set OPENAI_API_KEY as an environment variable.
  3. pip install -r requirements.txt
  4. streamlit run app.py --server.port 8501 --server.headless true
  Optionally configure nginx as a reverse proxy on port 80/443.

  Streamlit Secrets (alternative to environment variable):
  Create .streamlit/secrets.toml with:
    OPENAI_API_KEY = "sk-..."
  Then access via st.secrets["OPENAI_API_KEY"] — already handled in load_qa().
"""

import os
import time
from pathlib import Path

import streamlit as st

# ─── Page configuration — must be first Streamlit call ────────────────────
st.set_page_config(
    page_title  = "SEC Filings Analyst",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
    menu_items  = {
        "About": (
            "RAG-based analysis of SEC 10-K and 10-Q filings.\n\n"
            "Companies: AAPL · GOOGL · META · MSFT · NVDA\n"
            "Stack: FAISS · OpenAI Embeddings · GPT-4o-mini · Streamlit\n\n"
            "Built as a portfolio project demonstrating production RAG techniques:\n"
            "two-track ingestion (XBRL + HTML), metadata-filtered retrieval,\n"
            "per-company diversity enforcement, and grounded answer generation."
        )
    }
)

# ─── Company brand colours ─────────────────────────────────────────────────
COMPANY_COLOR = {
    "NVDA" : "#76B900",
    "MSFT" : "#00A4EF",
    "GOOGL": "#4285F4",
    "META" : "#0082FB",
    "AAPL" : "#888888",
}

SECTION_MAP = {
    "All sections"                      : (None, None),
    "Risk Factors"                       : ("Risk Factors",                       "narrative"),
    "Management Discussion & Analysis"   : ("Management Discussion and Analysis", "narrative"),
    "Business Overview"                  : ("Business",                           "narrative"),
    "Financial Summaries (XBRL)"         : ("Financial Summary",                  "financial_data"),
}

EXAMPLE_QUESTIONS = [
    {
        "icon"    : "💰",
        "label"   : "Financial performance",
        "question": "What were NVIDIA's revenue and net income in FY2024, and how did margins compare to the previous year?",
    },
    {
        "icon"    : "📈",
        "label"   : "Trend analysis",
        "question": "How has Apple's gross margin evolved from FY2022 to FY2025?",
    },
    {
        "icon"    : "⚖️",
        "label"   : "Comparative risk",
        "question": "How did Microsoft and Google each describe the competitive threat from AI in their most recent annual reports?",
    },
    {
        "icon"    : "🌐",
        "label"   : "Cross-company synthesis",
        "question": "How are these five companies describing their capital expenditure plans for AI infrastructure?",
    },
    {
        "icon"    : "⚠️",
        "label"   : "Specific risk theme",
        "question": "What geopolitical and macroeconomic risks did these companies flag in their 2024 annual reports?",
    },
    {
        "icon"    : "🔬",
        "label"   : "Deep dive",
        "question": "What specific AI products and partnerships did NVIDIA announce in the Data Center segment in FY2024?",
    },
]


# ─── Minimal custom styling ────────────────────────────────────────────────
st.markdown("""
<style>
  /* Tighten vertical rhythm */
  .block-container { padding-top: 1.5rem; }

  /* Source badge */
  .source-badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
      color: white;
      margin-right: 4px;
  }

  /* Source card */
  .source-card {
      border: 1px solid #e0e0e0;
      border-left: 4px solid #888;
      border-radius: 6px;
      padding: 10px 14px;
      margin-bottom: 8px;
      font-size: 0.85rem;
      background: #fafafa;
  }
  [data-theme="dark"] .source-card {
      background: #1e1e1e;
      border-color: #444;
  }

  /* Example question button — remove default padding */
  div[data-testid="stHorizontalBlock"] button {
      text-align: left;
  }

  /* Metadata line */
  .meta-line {
      font-size: 0.75rem;
      color: #888;
      margin-top: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ─── Cached resource loading ───────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading index and models…")
def load_qa_system():
    """
    Load the QA system once per server session.

    @st.cache_resource means this runs exactly once regardless of how many
    users or interactions hit the app. The FAISS index (~38 MB) and chunk
    metadata (~10 MB) are held in memory for the lifetime of the server.
    Re-loading on every query would add ~1s latency unnecessarily.
    """
    # Resolve API key from environment or Streamlit secrets
    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    from qa import QA
    return QA(verbose=False)


def api_key_available() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY")
        or st.secrets.get("OPENAI_API_KEY", "")
    )


# ─── Helper: build RetrievalFilters from sidebar state ────────────────────

def build_filters(
    companies   : list[str],
    years       : list[str],
    section_key : str,
    source_mode : str,
) -> "RetrievalFilters | None":
    from retrieve import RetrievalFilters

    section_name, section_source = SECTION_MAP[section_key]

    # source_type: prefer section_map override, then sidebar radio
    if section_source:
        resolved_source = section_source
    elif source_mode == "Narrative only":
        resolved_source = "narrative"
    elif source_mode == "Financial data only":
        resolved_source = "financial_data"
    else:
        resolved_source = None

    f = RetrievalFilters(
        company      = companies   or None,
        fiscal_year  = [int(y) for y in years] if years else None,
        section_name = section_name,
        source_type  = resolved_source,
    )

    return None if f.is_empty() else f


# ─── Helper: render a source card ─────────────────────────────────────────

def render_source_card(chunk) -> str:
    color   = COMPANY_COLOR.get(chunk.company, "#888")
    period  = (
        f"FY{chunk.fiscal_year}"
        if chunk.fiscal_period == "FY"
        else f"FY{chunk.fiscal_year} {chunk.fiscal_period}"
    )
    score_pct = int(chunk.score * 100)
    badge  = (
        f'<span class="source-badge" style="background:{color}">'
        f'{chunk.company}</span>'
    )
    return f"""
    <div class="source-card" style="border-left-color:{color}">
      {badge}
      <strong>{chunk.filing_type} {period}</strong> &nbsp;·&nbsp; {chunk.section_name}<br>
      <span style="font-size:0.78rem;color:#888">
        Chunk {chunk.chunk_index+1}/{chunk.chunk_total} &nbsp;·&nbsp;
        Relevance score: {chunk.score:.3f} ({score_pct}%)
      </span>
    </div>
    """


# ─── Sidebar ───────────────────────────────────────────────────────────────

def render_sidebar() -> tuple:
    with st.sidebar:
        st.markdown("## 📊 SEC Filings Analyst")
        st.caption(
            "10-K and 10-Q filings · 2022–2026\n"
            "AAPL · GOOGL · META · MSFT · NVDA"
        )
        st.divider()

        # ── Filters ───────────────────────────────────────────────────
        st.markdown("#### 🔍 Filters")
        st.caption(
            "Leave filters empty to search across all companies, years, and sections. "
            "The system auto-detects company names in your question when no filter is set."
        )

        companies = st.multiselect(
            "Company",
            options=["AAPL", "GOOGL", "META", "MSFT", "NVDA"],
            default=[],
            help="Restrict results to specific companies. Empty = all.",
        )

        years = st.multiselect(
            "Fiscal Year",
            options=["2022", "2023", "2024", "2025", "2026"],
            default=[],
            help="Filter by fiscal year derived from filing period end date.",
        )

        section_key = st.selectbox(
            "Section",
            options=list(SECTION_MAP.keys()),
            index=0,
        )

        source_mode = st.radio(
            "Source type",
            options=["All", "Narrative only", "Financial data only"],
            index=0,
            horizontal=False,
            help=(
                "Narrative: MD&A, Risk Factors, Business sections from the filing HTML.\n"
                "Financial data: structured XBRL summaries with computed margins."
            ),
        )

        st.divider()

        # ── Retrieval options ─────────────────────────────────────────
        st.markdown("#### ⚙️ Retrieval Options")

        top_k = st.slider(
            "Chunks to retrieve",
            min_value=3, max_value=12, value=5,
            help="More chunks = more context for the LLM, higher cost and latency.",
        )

        show_context = st.toggle(
            "Show retrieved passages",
            value=False,
            help=(
                "Display the exact text passages sent to the LLM. "
                "Useful for verifying retrieval quality and understanding "
                "how the answer was grounded."
            ),
        )

        st.divider()

        # ── System information ─────────────────────────────────────────
        st.markdown("#### ℹ️ System")
        st.markdown("""
        | Component | Detail |
        |-----------|--------|
        | Corpus | 160 filings → 6,396 chunks |
        | Embedding | text-embedding-3-small |
        | Index | FAISS IndexFlatIP |
        | LLM | gpt-4o-mini (T=0) |
        """)

        st.caption(
            "Two-track ingestion: structured XBRL data (Track A) "
            "and HTML narrative parsing (Track B). Metadata-filtered "
            "retrieval with per-company diversity enforcement."
        )

    return companies, years, section_key, source_mode, top_k, show_context


# ─── Main content ──────────────────────────────────────────────────────────

def render_header():
    st.title("SEC Filings Analyst")
    st.markdown(
        "Grounded Q&A over SEC 10-K and 10-Q filings for **Apple, Alphabet, "
        "Meta, Microsoft,** and **NVIDIA**. Every answer cites its source passage. "
        "The system will not speculate beyond what is written in the filings."
    )
    st.divider()


def render_example_questions() -> None:
    """Render clickable example questions.
    
    On click: writes the question text directly into st.session_state["question_input"]
    and sets auto_submit=True, then reruns. This is the correct Streamlit pattern —
    setting value= on a text_area only applies at first creation; after that the
    widget's value lives in session state and must be written there directly.
    """
    with st.expander("💡 Example questions — click any to ask immediately", expanded=True):
        cols = st.columns(3)
        for i, ex in enumerate(EXAMPLE_QUESTIONS):
            with cols[i % 3]:
                label = f"{ex['icon']} {ex['label']}"
                if st.button(
                    label,
                    key=f"ex_{i}",
                    use_container_width=True,
                    help=ex["question"],
                ):
                    st.session_state["question_input"] = ex["question"]
                    st.session_state["auto_submit"]    = True
                    st.rerun()


def render_query_input() -> tuple[str, bool]:
    """Render the question input and submit button.
    
    Uses key='question_input' so Streamlit reads/writes the value through
    session state. Never pass value= here — it is ignored after first render.
    """
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_area(
            "Ask a question about the filings:",
            key="question_input",
            height=80,
            placeholder=(
                "e.g. How did NVIDIA describe demand for its data center products in FY2024? "
                "Use the filters on the left to narrow the search."
            ),
            label_visibility="collapsed",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.button("Ask ▶", type="primary", use_container_width=True)

    return question.strip(), submitted


def render_answer(answer, show_context: bool) -> None:
    """Render a complete answer with sources and optional context."""

    # ── Answer text ───────────────────────────────────────────────────
    st.markdown("### Answer")
    st.markdown(answer.answer_text)

    # ── Source cards ──────────────────────────────────────────────────
    if answer.chunks_used:
        st.markdown("### Sources")

        # Deduplicate by (company, filing_type, fiscal_year, fiscal_period, section_name)
        seen   : set[str] = set()
        unique : list     = []
        for c in answer.chunks_used:
            key = f"{c.company}|{c.filing_type}|{c.fiscal_year}|{c.fiscal_period}|{c.section_name}"
            if key not in seen:
                seen.add(key)
                unique.append(c)

        n_cols = min(len(unique), 3)
        cols   = st.columns(n_cols)
        for i, chunk in enumerate(unique):
            with cols[i % n_cols]:
                st.markdown(render_source_card(chunk), unsafe_allow_html=True)

    # ── Metadata line ─────────────────────────────────────────────────
    st.markdown(
        f'<p class="meta-line">'
        f'Model: {answer.model} &nbsp;·&nbsp; '
        f'Tokens: {answer.input_tokens} in + {answer.output_tokens} out &nbsp;·&nbsp; '
        f'Cost: ${answer.cost_usd:.5f} &nbsp;·&nbsp; '
        f'Latency: {answer.elapsed_s:.1f}s'
        f'</p>',
        unsafe_allow_html=True,
    )

    # ── Retrieved passages (optional) ─────────────────────────────────
    if show_context and answer.chunks_used:
        with st.expander(
            f"📄 Retrieved passages sent to LLM ({len(answer.chunks_used)} chunks)",
            expanded=False,
        ):
            st.caption(
                "These are the exact text passages the LLM received as context. "
                "The answer is grounded exclusively in this text."
            )
            for chunk in answer.chunks_used:
                period = (
                    f"FY{chunk.fiscal_year}"
                    if chunk.fiscal_period == "FY"
                    else f"FY{chunk.fiscal_year} {chunk.fiscal_period}"
                )
                color = COMPANY_COLOR.get(chunk.company, "#888")
                st.markdown(
                    f'<div style="border-left:3px solid {color};padding-left:12px;margin-bottom:16px">'
                    f'<small><strong>{chunk.company} · {chunk.filing_type} {period} · '
                    f'{chunk.section_name}</strong> &nbsp; score: {chunk.score:.4f}</small><br><br>'
                    f'{chunk.content}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def render_history(history: list) -> None:
    """Render previous Q&As in collapsible sections."""
    if not history:
        return

    st.divider()
    st.markdown("### Previous questions this session")

    for i, (q, a) in enumerate(reversed(history), start=1):
        with st.expander(f"Q{len(history)-i+1}: {q[:80]}{'…' if len(q)>80 else ''}", expanded=False):
            st.markdown(a.answer_text)
            sources = a.unique_sources()
            if sources:
                st.markdown("**Sources:** " + " · ".join(sources))
            st.caption(
                f"{a.model} · {a.total_tokens} tokens · "
                f"${a.cost_usd:.5f} · {a.elapsed_s:.1f}s"
            )


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Session state initialisation ──────────────────────────────────
    if "history"          not in st.session_state:
        st.session_state.history      = []
    if "question_input"   not in st.session_state:
        st.session_state.question_input = ""
    if "auto_submit"      not in st.session_state:
        st.session_state.auto_submit  = False

    # ── Sidebar ───────────────────────────────────────────────────────
    companies, years, section_key, source_mode, top_k, show_context = render_sidebar()

    # ── Header ────────────────────────────────────────────────────────
    render_header()

    # ── API key check ─────────────────────────────────────────────────
    if not api_key_available():
        st.error(
            "**OPENAI_API_KEY not found.**\n\n"
            "Set it as an environment variable before starting the app:\n\n"
            "```bash\nexport OPENAI_API_KEY=sk-...\nstreamlit run app.py\n```\n\n"
            "Or add it to `.streamlit/secrets.toml`:\n\n"
            "```toml\nOPENAI_API_KEY = \"sk-...\"\n```"
        )
        st.stop()

    # ── Load system ───────────────────────────────────────────────────
    try:
        qa = load_qa_system()
    except Exception as e:
        st.error(
            f"**Failed to load the QA system.**\n\n{e}\n\n"
            "Make sure the pipeline has been run and `data/index/` exists."
        )
        st.stop()

    # ── Example questions ─────────────────────────────────────────────
    render_example_questions()

    # ── Query input ───────────────────────────────────────────────────
    question, submitted = render_query_input()

    # auto_submit is set by example buttons; consume it once
    auto = st.session_state.get("auto_submit", False)
    if auto:
        st.session_state["auto_submit"] = False

    # ── Process question ──────────────────────────────────────────────
    if (submitted or auto) and question:
        filters = build_filters(companies, years, section_key, source_mode)

        # For multi-company queries without explicit filter, auto-detection
        # in the Retriever will handle company identification from the question.
        # For cross-company queries with no company filter, enforce 1 chunk
        # per company so no single company dominates the context.
        max_per_company = (
            2            # 2 not 1: allows both annual + quarterly per company
            if not companies
            else None
        )

        with st.spinner("Searching filings and generating answer…"):
            try:
                from retrieve import RetrievalFilters as RF
                answer = qa.ask(
                    question        = question,
                    top_k           = top_k,
                    filters         = filters,
                    max_per_company = max_per_company,
                    show_context    = False,   # handled by render_answer
                )
            except Exception as e:
                st.error(f"**Error generating answer:** {e}")
                st.stop()

        render_answer(answer, show_context)

        # Store in history (keep last 10)
        st.session_state.history.append((question, answer))
        if len(st.session_state.history) > 10:
            st.session_state.history.pop(0)

    elif submitted and not question:
        st.warning("Please enter a question.")

    # ── Session history ───────────────────────────────────────────────
    if len(st.session_state.history) > 1:
        render_history(st.session_state.history[:-1])


if __name__ == "__main__":
    main()
