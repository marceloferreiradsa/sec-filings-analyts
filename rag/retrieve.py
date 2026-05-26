"""
retrieve.py — Retrieval module with metadata filtering

Loads:  data/index/index.faiss
        data/index/chunks.json
Exposes: Retriever class (used by app.py and qa.py)

WHAT THIS MODULE DOES
  Bridges vector search and the LLM. Given a question it:
    1. Embeds the question using the same model as the chunks.
    2. Searches the FAISS index for the most similar vectors.
    3. Filters results by metadata (company, year, section, etc.)
       so the LLM receives relevant chunks, not the index's most
       populated content (which without filtering would be META).
    4. Formats the retrieved chunks as a structured context string
       with source citations, ready to insert into an LLM prompt.

METADATA FILTERING STRATEGY — post-filter with amplification
  FAISS does not natively support metadata filtering. We over-retrieve
  by a factor of FILTER_AMPLIFICATION (default 15×) and then keep only
  the chunks that satisfy all filter conditions. The amplified candidate
  set (top_k × 15 = 75 candidates by default) is still searched in <5ms.

  When does this fail? If the filter is very selective (e.g., a single
  specific chunk type for a company with few documents), the 75-candidate
  window might not contain enough passing results. In that case the
  amplification is increased automatically.

  Why not build a sub-index on the filtered IDs?
  For 6,396 vectors, over-retrieval is simpler and equally fast. At >100K
  vectors, a sub-index or FAISS IDSelectorBatch would be preferable.

AUTO-DETECTION
  detect_filters() scans the question for company names/tickers and
  calendar years, and returns a filters dict. This lets simple queries
  like "What did Apple say about AI risk in 2024?" work without the
  caller needing to specify filters explicitly. Explicit filters passed
  to retrieve() always override auto-detected ones.

CONTEXT FORMAT
  Each retrieved chunk is rendered as:

    [NVDA · 10-K FY2024 · Risk Factors · chunk 3/101]
    Period: 2023-01-29 to 2024-01-28

    {chunk text}

  This format gives the LLM everything it needs to answer with a citation:
  company, form type, fiscal year, section, and the exact text.

Usage as a module (from app.py or qa.py):
    from retrieve import Retriever
    r = Retriever()
    results = r.retrieve("What drove NVIDIA's revenue growth?", top_k=5)
    context = r.format_context(results)

Usage standalone (interactive test):
    python retrieve.py
    python retrieve.py --query "Compare MSFT and GOOGL operating margins 2024"
    python retrieve.py --query "..." --company NVDA --year 2025 --top-k 8
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEX_DIR    = Path("./data/index")
INDEX_PATH   = INDEX_DIR / "index.faiss"
CHUNKS_PATH  = INDEX_DIR / "chunks.json"

EMBEDDING_MODEL      = "text-embedding-3-small"
FILTER_AMPLIFICATION = 15    # retrieve top_k × this many candidates before filtering
DEFAULT_TOP_K        = 5


# ---------------------------------------------------------------------------
# Company name → ticker mapping for auto-detection
# ---------------------------------------------------------------------------

COMPANY_ALIASES: dict[str, str] = {
    "nvidia": "NVDA",  "nvda":      "NVDA",
    "microsoft": "MSFT", "msft":    "MSFT",
    "google": "GOOGL",  "alphabet": "GOOGL", "googl": "GOOGL",
    "meta": "META",    "facebook":  "META",
    "apple": "AAPL",   "aapl":      "AAPL",
}


# ---------------------------------------------------------------------------
# Filter dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievalFilters:
    """
    All filter fields are optional. Unset fields do not constrain results.
    When a list is provided, a chunk matches if its value is in the list.
    When a single value is provided, a chunk matches only on exact equality.
    """
    company       : Optional[list[str] | str] = None
    filing_type   : Optional[str]             = None   # "10-K" | "10-Q"
    source_type   : Optional[str]             = None   # "narrative" | "financial_data"
    section_name  : Optional[str]             = None   # "Risk Factors" etc.
    fiscal_year   : Optional[list[int] | int] = None
    fiscal_period : Optional[str]             = None   # "FY" | "Q1" etc.

    def is_empty(self) -> bool:
        return all(
            v is None for v in [
                self.company, self.filing_type, self.source_type,
                self.section_name, self.fiscal_year, self.fiscal_period,
            ]
        )

    def summary(self) -> str:
        parts = []
        if self.company:
            parts.append(f"company={self.company}")
        if self.filing_type:
            parts.append(f"filing={self.filing_type}")
        if self.source_type:
            parts.append(f"source={self.source_type}")
        if self.section_name:
            parts.append(f"section={self.section_name!r}")
        if self.fiscal_year:
            parts.append(f"year={self.fiscal_year}")
        if self.fiscal_period:
            parts.append(f"period={self.fiscal_period}")
        return ", ".join(parts) if parts else "none"


def _matches(chunk_meta: dict, filters: RetrievalFilters) -> bool:
    """Return True if a chunk's metadata satisfies all active filters."""

    def check(meta_val, filter_val):
        if filter_val is None:
            return True
        if isinstance(filter_val, list):
            return str(meta_val) in [str(v) for v in filter_val]
        return str(meta_val) == str(filter_val)

    return all([
        check(chunk_meta.get("company"),       filters.company),
        check(chunk_meta.get("filing_type"),   filters.filing_type),
        check(chunk_meta.get("source_type"),   filters.source_type),
        check(chunk_meta.get("section_name"),  filters.section_name),
        check(chunk_meta.get("fiscal_year"),   filters.fiscal_year),
        check(chunk_meta.get("fiscal_period"), filters.fiscal_period),
    ])


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

def detect_filters(question: str) -> RetrievalFilters:
    """
    Scan a question for company references and calendar years,
    returning a RetrievalFilters populated with what was found.

    Examples:
      "What did NVIDIA say about AI risk in 2024?"
        → company=["NVDA"], fiscal_year=[2024]

      "Compare Microsoft and Google margins"
        → company=["MSFT", "GOOGL"]

      "What are the main risks for Apple?"
        → company=["AAPL"]

    Auto-detection is deliberately conservative — it only fires on
    explicit company names/tickers and 4-digit years in the range 2020-2026.
    It does not try to infer section type from keywords like "risk" because
    "I want to understand NVIDIA's risk profile" is better served by
    retrieving across all sections than restricting to Risk Factors alone.
    That decision belongs to the caller.
    """
    lower = question.lower()
    found_tickers: list[str] = []
    for alias, ticker in COMPANY_ALIASES.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', lower):
            if ticker not in found_tickers:
                found_tickers.append(ticker)

    year_matches = re.findall(r'\b(20(?:2[0-6]))\b', question)
    found_years  = [int(y) for y in dict.fromkeys(year_matches)]

    return RetrievalFilters(
        company     = found_tickers if found_tickers else None,
        fiscal_year = found_years   if found_years   else None,
    )


# ---------------------------------------------------------------------------
# Retrieved result container
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    score      : float
    rank       : int
    company    : str
    filing_type: str
    fiscal_year: str
    fiscal_period: str
    section_name: str
    source_type: str
    period_end : str
    chunk_index: int
    chunk_total: int
    chunk_id   : str
    content    : str
    metadata   : dict = field(repr=False)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Loads the FAISS index and chunk metadata once; retrieves on each call.

    The index and chunks are loaded at construction time and held in memory.
    At 6,396 chunks with 1,536-dimensional vectors this is ~40 MB — small
    enough that keeping it loaded is always preferable to reloading per query.

    Parameters
    ----------
    index_path  : path to the FAISS index binary
    chunks_path : path to the aligned chunks JSON
    model       : embedding model name (must match the one used in embed.py)
    verbose     : if True, print timing information on each retrieve() call
    """

    def __init__(
        self,
        index_path : Path = INDEX_PATH,
        chunks_path: Path = CHUNKS_PATH,
        model      : str  = EMBEDDING_MODEL,
        verbose    : bool = True,
    ) -> None:
        self.model   = model
        self.verbose = verbose
        self.client  = OpenAI()

        if self.verbose:
            print(f"[RETRIEVER] Loading index from {index_path}...")
        self.index = faiss.read_index(str(index_path))

        if self.verbose:
            print(f"[RETRIEVER] Loading {chunks_path.name}...")
        self.chunks: list[dict] = json.loads(
            chunks_path.read_text(encoding="utf-8")
        )

        if self.verbose:
            print(
                f"[RETRIEVER] Ready — "
                f"{self.index.ntotal:,} vectors, "
                f"{len(self.chunks):,} chunks."
            )

    # ── Core retrieval ────────────────────────────────────────────────

    def _embed_query(self, question: str) -> np.ndarray:
        """Embed a question string using the same model as the chunks."""
        response = self.client.embeddings.create(
            model=self.model,
            input=[question],
        )
        return np.array(
            response.data[0].embedding,
            dtype=np.float32,
        ).reshape(1, -1)

    def retrieve(
        self,
        question        : str,
        top_k           : int = DEFAULT_TOP_K,
        filters         : Optional[RetrievalFilters] = None,
        auto_detect     : bool = True,
        max_per_company : Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Embed the question, search the index, apply filters, return top-k.

        Parameters
        ----------
        question        : the user's question in plain English
        top_k           : number of chunks to return after filtering
        filters         : explicit metadata constraints (override auto-detection)
        auto_detect     : if True and filters is None, auto-detect company/year
                          from the question text
        max_per_company : if set, caps the number of chunks from any single
                          company. Useful for cross-company queries where
                          one company's content would otherwise dominate.
                          When filters.company contains multiple companies,
                          per-company retrieval is used automatically instead.

        Returns
        -------
        List of RetrievedChunk objects, ordered by descending similarity score.

        MULTI-COMPANY STRATEGY
          When the active filter contains a list of two or more companies,
          retrieve() delegates to _retrieve_per_company() which runs a
          separate search for each company and merges the results by score.

          This guarantees representation from every requested company
          regardless of relative scoring — critical for comparative queries
          like "How did Microsoft and Google describe cloud competition?"
          where one company's language may score higher against the query
          even though the user wants both perspectives.

          Single-company and unfiltered queries use the standard path:
          one FAISS search with post-filtering.
        """
        t_start = time.time()

        # Resolve filters
        active_filters = filters
        if active_filters is None and auto_detect:
            active_filters = detect_filters(question)

        if active_filters is None:
            active_filters = RetrievalFilters()

        if self.verbose:
            print(f"\n[RETRIEVE] Question: \"{question}\"")
            print(f"[RETRIEVE] Filters:  {active_filters.summary()}")
            print(f"[RETRIEVE] Top-k:    {top_k}")

        # ── Multi-company: retrieve per-company then merge ─────────────
        companies = active_filters.company
        if isinstance(companies, list) and len(companies) > 1:
            return self._retrieve_per_company(
                question, active_filters, top_k, t_start
            )

        # ── Single-company or unfiltered: standard path ────────────────
        t_embed = time.time()
        q_vector = self._embed_query(question)
        embed_ms = (time.time() - t_embed) * 1000
            
        #n_candidates = min(
        #    top_k * FILTER_AMPLIFICATION,
        #    self.index.ntotal,
        #)

        # IndexFlatIP always computes all inner products regardless of k.
        # Requesting all candidates costs nothing extra and guarantees
        # every matching chunk is visible to the post-filter step.
        n_candidates = self.index.ntotal
        
        t_search = time.time()
        scores, positions = self.index.search(q_vector, n_candidates)
        search_ms = (time.time() - t_search) * 1000

        results        : list[RetrievedChunk] = []
        candidates_checked = 0
        company_counts : dict[str, int] = {}

        for score, pos in zip(scores[0], positions[0]):
            if pos < 0:
                continue
            if len(results) >= top_k:
                break

            chunk = self.chunks[int(pos)]
            meta  = chunk["metadata"]
            candidates_checked += 1

            if not _matches(meta, active_filters):
                continue

            company = meta.get("company", "")
            if max_per_company is not None:
                if company_counts.get(company, 0) >= max_per_company:
                    continue
                company_counts[company] = company_counts.get(company, 0) + 1

            results.append(self._make_result(chunk, meta, float(score)))

        total_ms = (time.time() - t_start) * 1000

        # Assign ranks now that the list is final
        for i, r in enumerate(results, start=1):
            r.rank = i

        if self.verbose:
            print(
                f"[RETRIEVE] {len(results)} results from "
                f"{candidates_checked} candidates checked  |  "
                f"embed {embed_ms:.0f}ms  "
                f"search {search_ms:.2f}ms  "
                f"total {total_ms:.0f}ms"
            )
            if len(results) < top_k and not active_filters.is_empty():
                print(
                    f"[RETRIEVE] Warning: only {len(results)}/{top_k} results "
                    f"found after filtering. Consider relaxing the filter or "
                    f"increasing FILTER_AMPLIFICATION."
                )

        return results

    def _retrieve_per_company(
        self,
        question       : str,
        base_filters   : RetrievalFilters,
        top_k          : int,
        t_start        : float,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top-k/n_companies from each company separately, merge by score.

        This guarantees that every company in base_filters.company is
        represented in the final results regardless of relative scores.

        Example: top_k=6 with companies=[MSFT, GOOGL] → 3 chunks from each,
        then sorted by score. MSFT might take ranks 1, 2, 4 and GOOGL takes
        3, 5, 6 — both are present for the LLM to compare.
        """
        companies     = base_filters.company
        k_per_company = max(1, -(-top_k // len(companies)))   # ceiling division
        all_results   : list[RetrievedChunk] = []

        if self.verbose:
            print(
                f"[RETRIEVE] Multi-company mode: {len(companies)} companies, "
                f"{k_per_company} chunks each."
            )

        for company in companies:
            company_filter = RetrievalFilters(
                company       = company,
                filing_type   = base_filters.filing_type,
                source_type   = base_filters.source_type,
                section_name  = base_filters.section_name,
                fiscal_year   = base_filters.fiscal_year,
                fiscal_period = base_filters.fiscal_period,
            )
            # Temporarily suppress verbose for sub-calls
            self.verbose = False
            sub_results = self.retrieve(
                question    = question,
                top_k       = k_per_company,
                filters     = company_filter,
                auto_detect = False,
            )
            self.verbose = True

            if self.verbose:
                print(
                    f"[RETRIEVE]   {company}: "
                    f"{len(sub_results)} chunks "
                    f"(best score {sub_results[0].score:.4f})"
                    if sub_results
                    else f"[RETRIEVE]   {company}: 0 chunks found"
                )

            all_results.extend(sub_results)

        # Re-rank merged results by score, re-assign ranks
        all_results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(all_results[:top_k], start=1):
            r.rank = i

        total_ms = (time.time() - t_start) * 1000
        if self.verbose:
            print(
                f"[RETRIEVE] {len(all_results[:top_k])} merged results  |  "
                f"total {total_ms:.0f}ms"
            )

        return all_results[:top_k]

    @staticmethod
    def _make_result(
        chunk: dict,
        meta : dict,
        score: float,
    ) -> RetrievedChunk:
        """Construct a RetrievedChunk from a raw chunk dict and score."""
        return RetrievedChunk(
            score         = score,
            rank          = 0,           # caller assigns final rank
            company       = meta.get("company", ""),
            filing_type   = meta.get("filing_type", ""),
            fiscal_year   = str(meta.get("fiscal_year", "")),
            fiscal_period = str(meta.get("fiscal_period", "")),
            section_name  = meta.get("section_name", ""),
            source_type   = meta.get("source_type", ""),
            period_end    = meta.get("period_end", ""),
            chunk_index   = meta.get("chunk_index", 0),
            chunk_total   = meta.get("chunk_total", 0),
            chunk_id      = meta.get("chunk_id", ""),
            content       = chunk["page_content"],
            metadata      = meta,
        )

    # ── Context formatting ────────────────────────────────────────────

    @staticmethod
    def format_context(results: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks as a context block for the LLM.

        Each chunk is labelled with company, filing type, fiscal period,
        section, and period dates. This gives the LLM everything it needs
        to produce a cited answer.

        The separator between chunks (---) helps the LLM understand where
        one source ends and another begins.
        """
        if not results:
            return "No relevant documents found."

        parts = []
        for r in results:
            period_label = (
                f"FY{r.fiscal_year}"
                if r.fiscal_period == "FY"
                else f"FY{r.fiscal_year} {r.fiscal_period}"
            )
            header = (
                f"[{r.company} · {r.filing_type} {period_label} · "
                f"{r.section_name} · chunk {r.chunk_index+1}/{r.chunk_total}]"
            )
            if r.metadata.get("period_end"):
                start = r.metadata.get("period_start", "")
                end   = r.metadata.get("period_end", "")
                date_line = (
                    f"Period: {start} to {end}"
                    if start
                    else f"As of: {end}"
                )
            else:
                date_line = ""

            block = header
            if date_line:
                block += f"\n{date_line}"
            block += f"\n\n{r.content}"
            parts.append(block)

        return "\n\n---\n\n".join(parts)

    # ── Convenience: print a formatted result table ───────────────────

    @staticmethod
    def print_results(results: list[RetrievedChunk]) -> None:
        if not results:
            print("  (no results)")
            return

        print(
            f"\n  {'Rank':<5}  {'Score':<7}  {'Company':<8}  "
            f"{'Filing':<6}  {'Year':<5}  {'Per':<4}  "
            f"{'Source':<14}  Section"
        )
        print(f"  {'─'*90}")

        for r in results:
            print(
                f"  {r.rank:<5}  {r.score:.4f}  {r.company:<8}  "
                f"{r.filing_type:<6}  {r.fiscal_year:<5}  {r.fiscal_period:<4}  "
                f"{r.source_type:<14}  {r.section_name}"
            )

        print(f"\n  Score spread: "
              f"{results[0].score:.4f} → {results[-1].score:.4f}  "
              f"(gap {results[0].score - results[-1].score:.4f})")

        print(f"\n  Rank 1 preview:")
        print(f"  {results[0].content[:300].replace(chr(10), ' ').strip()}...")


# ---------------------------------------------------------------------------
# Interactive test
# ---------------------------------------------------------------------------

def run_interactive_tests(retriever: Retriever) -> None:
    """
    Run a set of test queries that exercise different retrieval modes:
    unfiltered, company-filtered, multi-company, and section-filtered.
    """
    test_cases = [
        {
            "label":   "Unfiltered — should show spread across companies",
            "question": "How are these companies managing AI infrastructure costs?",
            "filters":  None,
            "top_k":    5,
        },
        {
            "label":   "Auto-detected company — should be NVDA only",
            "question": "What risks does NVIDIA face in its data center business?",
            "filters":  None,
            "top_k":    5,
        },
        {
            "label":   "Auto-detected multi-company — MSFT and GOOGL",
            "question": "How did Microsoft and Google describe cloud competition in 2024?",
            "filters":  None,
            "top_k":    6,
        },
        {
            "label":   "Explicit filter: AAPL financial data only",
            "question": "What are Apple's recent revenue and margin figures?",
            "filters":  RetrievalFilters(
                company="AAPL",
                source_type="financial_data",
            ),
            "top_k":    4,
        },
        {
            "label":   "Explicit filter: annual filings, Risk Factors, 2024 — max 1 per company",
            "question": "What macroeconomic risks did companies flag in their annual reports?",
            "filters":  RetrievalFilters(
                filing_type="10-K",
                section_name="Risk Factors",
                fiscal_year=2024,
            ),
            "top_k":    5,
            "max_per_company": 1,    # enforce one chunk per company for diversity
        },
    ]

    for i, tc in enumerate(test_cases, start=1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {tc['label']}")
        print(f"{'='*60}")

        results = retriever.retrieve(
            question        = tc["question"],
            top_k           = tc["top_k"],
            filters         = tc.get("filters"),
            max_per_company = tc.get("max_per_company"),
        )
        retriever.print_results(results)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval module test.")
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Single query to run instead of the full test suite.",
    )
    parser.add_argument(
        "--company", "-c",
        type=str,
        default=None,
        help="Filter by company ticker (e.g. NVDA, MSFT).",
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        default=None,
        help="Filter by fiscal year (e.g. 2024).",
    )
    parser.add_argument(
        "--section", "-s",
        type=str,
        default=None,
        help="Filter by section name (e.g. 'Risk Factors').",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Filter by source type: narrative | financial_data.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        dest="top_k",
        help=f"Number of results to return (default {DEFAULT_TOP_K}).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY environment variable not set.")
        return

    retriever = Retriever(verbose=True)

    if args.query:
        # Single query with optional explicit filters
        filters = RetrievalFilters(
            company      = args.company,
            fiscal_year  = args.year,
            section_name = args.section,
            source_type  = args.source,
        )
        results = retriever.retrieve(
            question=args.query,
            top_k=args.top_k,
            filters=filters if not filters.is_empty() else None,
        )
        retriever.print_results(results)

        print(f"\n{'─'*60}")
        print(f"FORMATTED CONTEXT (what the LLM will see):")
        print(f"{'─'*60}")
        print(retriever.format_context(results))

    else:
        # Full test suite
        run_interactive_tests(retriever)


if __name__ == "__main__":
    main()
