# SEC Filings Analyst

Retrieval-augmented Q&A over SEC 10-K and 10-Q filings for five major technology
companies. Every answer is grounded in actual filing text with inline source
citations. The system will not speculate beyond what is written in the filings.

**Companies covered:** Apple (AAPL) · Alphabet/Google (GOOGL) · Meta (META) · Microsoft (MSFT) · NVIDIA (NVDA)  
**Filing types:** 10-K (annual) · 10-Q (quarterly)  
**Period:** FY2022 – early FY2026  
**Stack:** Python · FAISS · OpenAI Embeddings · GPT-4o-mini · Streamlit

---

## What it does

Ask questions in natural language about SEC filings and receive grounded answers
with citations to the exact source passage:

> *"What drove NVIDIA's gross margin expansion between FY2023 and FY2025?"*

> NVIDIA's gross margin expanded from 56.9% in FY2023 to 72.7% in FY2024 and
> 73.0% in FY2025 [NVDA 10-K FY2024 Financial Summary], [NVDA 10-K FY2025
> Financial Summary]. The primary driver was a shift in revenue mix toward
> Data Center products, particularly the Hopper architecture GPU systems...
> [NVDA 10-K FY2024 Management Discussion and Analysis].

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE (run once)            │
│                                                             │
│  SEC EDGAR ──► edgar_api.py ──► XBRL summaries ──┐         │
│    (XBRL)       Track A          formatted text   │         │
│                                                   ├─► document_builder.py
│  SEC EDGAR ──► html_parser.py ──► sections ───────┘         │
│    (HTML)       Track B          Risk/MD&A/Business         │
│                                                             │
│         ingest.py ──► chunk.py ──► embed.py ──► index.py   │
│         documents     chunks       vectors       FAISS      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE (real-time)               │
│                                                             │
│  Question ──► retrieve.py ──► top-k chunks ──► qa.py       │
│               FAISS + filters   with metadata   GPT-4o-mini │
│                                                   │         │
│  app.py (Streamlit) ◄────────────────────────────┘         │
│  Filters · Citations · Context toggle                       │
└─────────────────────────────────────────────────────────────┘
```

### Two-track ingestion

**Track A (XBRL)** fetches certified financial values from the SEC's structured
API at `data.sec.gov`. Numbers are rendered as natural-language summaries with
pre-computed margins before embedding — raw integers do not embed well
semantically. Financial summaries answer "what were the numbers?" queries.

**Track B (HTML)** parses narrative filing sections: Risk Factors, MD&A, and
Business Overview. These sections answer "why did the numbers move?" queries.

The two tracks share a unified metadata schema and are searched together by the
retrieval layer.

---

## Key design decisions

Five decisions distinguish this system from standard RAG tutorials.
The full rationale for all eleven architectural decisions is in
[ARCHITECTURAL_DECISIONS.md](ARCHITECTURAL_DECISIONS.md).

| Decision | What and why |
|----------|-------------|
| Natural-language number rendering | Raw XBRL integers do not embed semantically. Values are formatted in billions and margins are pre-computed so financial data is retrievable by meaning. |
| Exact FAISS search (IndexFlatIP) | At 6,396 vectors, exact search takes <2ms. Approximate indexes trade accuracy for imperceptible speed gains at this scale. |
| ASC 606 combine strategy | The 2018 accounting standard change split the revenue concept across two XBRL tags. Entries from all candidate tags are pooled before deduplication for metrics that are tag renames of the same concept. |
| Per-company retrieval | Multi-company queries with combined filters produce asymmetric results — one company's vocabulary dominates. Separate calibrated searches per company guarantee equal representation. |
| Post-filter with n=ntotal | For IndexFlatIP, all inner products are computed regardless of k. Requesting all candidates costs nothing extra and guarantees sparse categories (financial summaries = 0.6% of index) are always reachable. |

---

## Project structure

```
sec-filings-analyst/
│
├── app.py               Streamlit web interface
├── qa.py                LLM answer generation (GPT-4o-mini)
├── retrieve.py          FAISS retrieval with metadata filtering
├── index.py             Build FAISS index from embeddings
├── embed.py             Embed chunks with text-embedding-3-small
├── chunk.py             Section-aware chunking
├── ingest.py            Pipeline orchestrator
├── document_builder.py  Assemble Track A + Track B documents
├── edgar_api.py         Track A: XBRL financial data from SEC API
├── html_parser.py       Track B: HTML narrative section extraction
│
├── ARCHITECTURAL_DECISIONS.md   All 11 design decisions with rationale
├── NEXT_FEATURES.md             Documented improvement roadmap
├── requirements.txt
├── .env.example
│
└── data/                        Generated by pipeline (git-ignored)
    ├── raw/             Downloaded SEC filings
    ├── processed/       documents.json · chunks.json · embeddings.npy
    └── index/           index.faiss
```

---

## Quick start

### Prerequisites
- Python 3.10+
- OpenAI API key
- ~2GB disk space for downloaded filings
- ~$0.05 to run the full embedding pipeline

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/sec-filings-analyst.git
cd sec-filings-analyst

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

The system reads `OPENAI_API_KEY` from the OS environment. No `.env` file
is needed or recommended — setting the key as a system variable keeps it
out of the filesystem and out of git history entirely.

**Windows (permanent, current user):**
```powershell
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-your-key-here", "User")
# Restart your terminal after setting
```

**Windows (current session only):**
```powershell
$env:OPENAI_API_KEY = "sk-your-key-here"
```

**Linux / macOS (permanent):**
```bash
echo 'export OPENAI_API_KEY=sk-your-key-here' >> ~/.bashrc
source ~/.bashrc
```

`.env.example` documents the required variable name for reference.
`python-dotenv` is not required — the system reads directly from the OS
environment, which is the correct practice for keeping secrets out of
the filesystem and git history.

### Run the pipeline

Each step builds on the previous. Run them in order once to set up the corpus.

```bash
python ingest.py     # Download filings + build XBRL summaries + parse HTML
python chunk.py      # Split documents into retrievable chunks
python embed.py      # Embed chunks with text-embedding-3-small (~$0.03)
python index.py      # Build FAISS index
```

Pipeline outputs are saved to `data/` and git-ignored. Re-run from any step
to update — for example, re-run from `chunk.py` if you change chunk size,
or from `embed.py` if you change the embedding model.

### Use the system

**Web interface (recommended):**
```bash
streamlit run app.py
```

**Interactive command line:**
```bash
python qa.py -i                     # interactive mode
python qa.py -i --show-context      # show retrieved passages
```

**Single question:**
```bash
python qa.py -q "What risks did NVIDIA flag in its FY2025 annual report?"
python qa.py -q "Compare MSFT and GOOGL cloud revenue growth" --show-context
```

**Test suite (5 pre-built test cases):**
```bash
python qa.py                        # answers only
python qa.py --show-context         # context + answers
```

---

## Query examples

The Streamlit interface includes six clickable example questions. Additional
examples covering different query types:

**Financial performance**
- "What were NVIDIA's revenues and margins in FY2024 compared to FY2023?"
- "How has Apple's gross margin evolved from FY2022 to FY2025?"
- "What were the latest quarterly revenues for all five companies?"

**Risk and strategy**
- "What AI-related risks did Microsoft flag in its FY2025 annual report?"
- "How did these companies describe geopolitical risks in FY2024?"
- "How is Meta describing its AI infrastructure investment plans?"

**Comparative analysis**
- "How did Microsoft and Google each describe cloud competition in 2024?"
- "Which company showed the strongest operating margin improvement recently?"

---

## Filters and retrieval options

The Streamlit sidebar exposes the full retrieval configuration:

| Filter | Effect |
|--------|--------|
| Company | Restrict to one or more tickers |
| Fiscal Year | Filter by fiscal year derived from period_end |
| Section | Risk Factors · MD&A · Business · Financial Summaries |
| Source Type | All · Narrative only · Financial data only |
| Chunks to retrieve | 3–12 (default 5) |
| Show retrieved passages | Display exact context sent to LLM |

**For financial figure questions** (revenues, margins, earnings): select
"Financial data only" to route directly to the XBRL-derived summaries.

**For narrative questions** (strategy, risks, competitive positioning): use
the default "All" or "Narrative only."

---

## Known limitations

**Fiscal calendar alignment.** Financial summaries are organised by company
fiscal year, not calendar year. For companies with non-December fiscal year
ends (NVIDIA ends January, Apple ends September, Microsoft ends June),
queries specifying calendar years may not find a direct match. Use fiscal
year terminology — "NVIDIA FY2025" or "Apple FY2025" — for reliable results.

**"Latest" is semantic, not temporal.** The retriever ranks by semantic
similarity, not by recency. Queries for "the latest" figures may return older
periods if they score higher semantically. Adding a year filter or selecting
"Financial data only" with the most recent year mitigates this.

**Corpus boundary.** The system covers five companies over approximately four
years. Questions about other companies, earlier periods, or forward-looking
projections will be correctly declined — the system refuses to speculate beyond
what is in the retrieved passages.

**Segment-level data.** Company-specific XBRL extension tags (e.g.
`nvda:DataCenterRevenue`) are not currently indexed. Revenue figures are
consolidated totals. Segment breakdown questions are partially answered from
narrative MD&A sections.

---

## Costs

| Operation | Cost |
|-----------|------|
| Full pipeline embedding (6,396 chunks) | ~$0.028 |
| Single query (retrieve + GPT-4o-mini) | ~$0.0005 |
| Five-question test suite | ~$0.002 |

All costs are OpenAI API charges. FAISS search and pipeline processing
are local and free.

---

## Roadmap

Eight planned improvements are documented with implementation paths and
effort estimates in [NEXT_FEATURES.md](NEXT_FEATURES.md). Priority order:

1. **Section selection via config file** — move `SECTIONS_TO_INDEX` to
   `config.yaml` for code-free configuration
2. **Cross-encoder re-ranking** — re-rank top-20 candidates with a
   cross-encoder for higher retrieval precision
3. **User feedback loop** — thumbs up/down logging to accumulate
   fine-tuning data
4. **Hybrid retrieval sidecar** — preserve raw XBRL values alongside
   formatted summaries for exact numeric lookup on demand
5. **On-demand chart generation** — Plotly charts for trend and
   comparison queries (depends on #4)

---

## Deployment

### Local
```bash
streamlit run app.py
```

### VPS
```bash
export OPENAI_API_KEY=sk-...
streamlit run app.py --server.port 8501 --server.headless true
```

Configure nginx as a reverse proxy for port 80/443. Only the following
files are required on the server — the full pipeline does not need to run
there:

```
app.py  retrieve.py  qa.py
data/index/index.faiss
data/processed/chunks.json
requirements.txt
```

Transfer the index with:
```bash
rsync -avz data/index/ data/processed/chunks.json \
    user@your-vps:/path/to/app/data/
```

Docker deployment instructions are in progress.

---

## License

MIT
