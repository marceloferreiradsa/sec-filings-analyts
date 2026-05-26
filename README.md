# SEC Filings Analyst

Retrieval-augmented Q&A over SEC 10-K and 10-Q filings for five major technology
companies. Every answer is grounded in actual filing text with inline source
citations. The system will not speculate beyond what is written in the filings.

**Companies covered:** Apple (AAPL) · Alphabet/Google (GOOGL) · Meta (META) · Microsoft (MSFT) · NVIDIA (NVDA)  
**Filing types:** 10-K (annual) · 10-Q (quarterly)  
**Period:** FY2022 – early FY2026  
**Stack:** Python · FAISS · OpenAI Embeddings · GPT-4o-mini · Streamlit  
**Live demo:** https://secrag.ainati.com.br

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
│                    INGESTION PIPELINE                       │
│                    python pipeline.py                       │
│                                                             │
│  SEC EDGAR ──► data_loaders/ ──────────────────────┐       │
│    XBRL         edgar_api.py   financial summaries  │       │
│    HTML         html_parser.py narrative sections   ├─► document_builder.py
│                                                     │       │
│         stages/ingest.py ──► chunk.py ──► embed.py ──► index.py
│         documents              chunks      vectors      FAISS
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE (real-time)               │
│                                                             │
│  Question ──► rag/retrieve.py ──► top-k chunks ──► rag/qa.py
│               FAISS + filters      with metadata    GPT-4o-mini
│                                                      │      │
│  app.py (Streamlit) ◄────────────────────────────────┘      │
│  Filters · Citations · Context toggle                        │
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
The full rationale for all architectural decisions is in
[docs/ARCHITECTURAL_DECISIONS.md](docs/ARCHITECTURAL_DECISIONS.md).

| Decision | What and why |
|----------|-------------|
| Natural-language number rendering | Raw XBRL integers do not embed semantically. Values are formatted in billions and margins are pre-computed so financial data is retrievable by meaning. |
| Exact FAISS search (IndexFlatIP) | At 6,396 vectors, exact search takes <2ms. Approximate indexes trade accuracy for imperceptible speed gains at this scale. |
| ASC 606 combine strategy | The 2018 accounting standard change split the revenue concept across two XBRL tags. Entries from all candidate tags are pooled before deduplication. |
| Per-company retrieval diversity | Multi-company queries can be dominated by one company's vocabulary. Per-company diversity enforcement guarantees equal representation in context. |
| Post-filter with n=ntotal | For IndexFlatIP, all inner products are computed regardless of k. Requesting all candidates costs nothing extra and guarantees sparse categories are always reachable. |

---

## Project structure

```
sec-filings-analyst/
│
├── app.py               Streamlit web interface (Docker entry point)
├── pipeline.py          Build orchestrator — runs the full pipeline or individual stages
│
├── rag/                 Runtime: retrieval and generation
│   ├── retrieve.py      FAISS retrieval with metadata filtering
│   └── qa.py            LLM answer generation (GPT-4o-mini)
│
├── stages/              Build pipeline stages
│   ├── ingest.py        Download filings and build documents
│   ├── chunk.py         Section-aware chunking
│   ├── embed.py         Embed chunks with text-embedding-3-small
│   └── index.py         Build FAISS index from embeddings
│
├── data_loaders/        Raw data parsers
│   ├── edgar_api.py     Track A: XBRL financial data from SEC API
│   ├── html_parser.py   Track B: HTML narrative section extraction
│   └── document_builder.py  Assemble Track A + Track B into unified documents
│
├── eval/
│   └── evaluate.py      RAGAS evaluation · smoke tests · assertion suite
│
├── tests/
│   └── test_deployment.py  Pre-deployment container validation (20 checks)
│
├── docs/
│   ├── ARCHITECTURAL_DECISIONS.md   All design decisions with rationale
│   ├── NEXT_FEATURES.md             Improvement roadmap with effort estimates
│   └── DEPLOYMENT.md                VPS deployment and ops guide
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt         Production (4 packages: openai, faiss-cpu, numpy, streamlit)
├── requirements-dev.txt     Development (adds pipeline and evaluation tools)
│
└── data/                    Generated by pipeline (git-ignored)
    ├── raw/                 Downloaded SEC filings
    ├── processed/           chunks.json · embeddings.npy
    └── index/               index.faiss · chunks.json
```

---

## Quick start

### Prerequisites
- Python 3.10+
- OpenAI API key
- ~2 GB disk space for downloaded filings
- ~$0.05 to run the full embedding pipeline

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/sec-filings-analyst.git
cd sec-filings-analyst

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt   # includes pipeline and evaluation tools
```

### Configuration

The system reads `OPENAI_API_KEY` from the OS environment.

**Windows (permanent):**
```powershell
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-your-key-here", "User")
```

**Linux / macOS:**
```bash
echo 'export OPENAI_API_KEY=sk-your-key-here' >> ~/.bashrc && source ~/.bashrc
```

### Run the pipeline

```bash
python pipeline.py               # full pipeline (ingest → chunk → embed → index)
python pipeline.py --only embed  # single stage
python pipeline.py --from chunk  # from a stage onwards
python pipeline.py --status      # check what has been built
```

Each stage builds on the previous. Re-run from any step to update — for example,
re-run from `embed` if you change the embedding model, or from `chunk` if you
change chunk size.

Approximate cost for the full corpus: ~$0.028 in OpenAI embedding charges.

### Use the system

**Web interface (recommended):**
```bash
streamlit run app.py
```

**Evaluation:**
```bash
python -m eval.evaluate --mode smoke      # 5 health checks (~$0.002)
python -m eval.evaluate --mode all        # smoke + 20 assertion tests
python -m eval.evaluate --mode ragas-real # full RAGAS scoring (~$0.10)
```

---

## Query examples

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

**Fiscal calendar alignment.** NVIDIA ends January, Apple ends September,
Microsoft ends June. Queries specifying calendar years may not find a direct
match. Use fiscal year terminology — "NVIDIA FY2025" — for reliable results.

**"Latest" is semantic, not temporal.** The retriever ranks by semantic
similarity, not recency. Adding a year filter or selecting "Financial data only"
mitigates this. Feature 9 (intelligent query understanding) will resolve this
automatically.

**Corpus boundary.** Covers five companies over approximately four years.
Questions about other companies, earlier periods, or forward-looking projections
are correctly declined.

**Segment-level data.** Company-specific XBRL extension tags
(e.g. `nvda:DataCenterRevenue`) are not currently indexed. Revenue figures
are consolidated totals. Segment breakdown questions are partially answered
from narrative MD&A sections.

---

## Costs

| Operation | Cost |
|-----------|------|
| Full pipeline embedding (6,396 chunks) | ~$0.028 |
| Single query (retrieve + GPT-4o-mini) | ~$0.0005 |
| Smoke test suite (5 questions) | ~$0.002 |
| Full RAGAS evaluation | ~$0.10 |

---

## Evaluation baseline

| Metric | Score |
|--------|-------|
| Faithfulness | 0.962 |
| Answer Relevancy (answerable questions) | 0.928 |
| Smoke tests | 5/5 |
| Deployment tests | 20/20 |

---

## Deployment

The production deployment uses Docker with nginx as a reverse proxy.
Full instructions in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

**Quick start (local Docker):**
```bash
docker build -t sec-filings-analyst .
docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY sec-filings-analyst
```

**Pre-deployment validation:**
```bash
docker run -d -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY --name sec-analyst sec-filings-analyst
python tests/test_deployment.py   # 20 checks across 5 phases
```

The production image contains only 4 packages (openai, faiss-cpu, numpy,
streamlit) and the `rag/` package. Build time ~2 minutes, image size ~400 MB.

---

## Roadmap

Full feature specifications with implementation paths and effort estimates
are in [docs/NEXT_FEATURES.md](docs/NEXT_FEATURES.md).

**Next priorities:**

1. **Intelligent query understanding (Feature 9)** — LLM pre-filtering that
   extracts company, period, and intent from natural language automatically.
   Eliminates the need for manual sidebar filters.
2. **Cross-encoder re-ranking (Feature 4)** — re-rank top-20 candidates with
   a cross-encoder for higher retrieval precision.
3. **User feedback loop (Feature 5)** — thumbs up/down logging to accumulate
   fine-tuning data over time.

---

## License

MIT
