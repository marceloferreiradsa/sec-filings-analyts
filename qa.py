"""
qa.py — LLM question-answering over retrieved SEC filing chunks

Imports:  retrieve.Retriever
Uses:     OpenAI gpt-4o-mini for answer synthesis
          OpenAI text-embedding-3-small for query embedding (via Retriever)

WHAT THIS MODULE DOES
  Connects the retrieval layer to a language model. For each question it:
    1. Retrieves the most relevant chunks from the FAISS index (via Retriever).
    2. Formats them as a structured context block with source labels.
    3. Sends context + question to the LLM with a strict grounding prompt.
    4. Returns the answer with inline citations.

  The LLM's role is synthesis and reasoning, not knowledge. It only knows
  what is in the retrieved context. If the answer is not there, it says so.

VERBOSITY FLAGS
  --show-context / -c
    Before each answer, prints the full context block sent to the LLM.
    Use this to understand why the model answered the way it did, or to
    debug retrieval quality. Without this flag, only the answer is shown.

  In interactive mode, type /context to toggle context display on or off
  without restarting.

LLM PROMPT DESIGN
  System prompt instructs the model to:
    - Use only the provided excerpts (no prior knowledge about these companies)
    - Cite every factual claim with [COMPANY FILING PERIOD SECTION]
    - Explicitly flag when context is insufficient rather than guessing
    - Include units and time periods with all numerical claims

  Temperature is set to 0 — financial analysis requires determinism,
  not creativity. Every run of the same question should produce the same answer.

MODEL
  gpt-4o-mini — fast, cheap ($0.15/1M input tokens), sufficient for
  financial synthesis from structured context. Typical cost per question:
  ~$0.0005 USD (2,000 input + 400 output tokens). Can be changed via
  the MODEL constant below.

Usage:
    python qa.py                         # test suite, answers only
    python qa.py --show-context          # test suite, context + answers
    python qa.py -i                      # interactive mode
    python qa.py -i --show-context       # interactive mode, context visible
    python qa.py -q "question text"      # single question
    python qa.py -q "..." -c             # single question with context
    python qa.py -q "..." --company NVDA --year 2024
"""

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

from retrieve import Retriever, RetrievalFilters, RetrievedChunk


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL       = "gpt-4o-mini"
TEMPERATURE = 0              # deterministic answers for financial analysis
MAX_TOKENS  = 1024           # answer length ceiling
DEFAULT_TOP_K = 5

SYSTEM_PROMPT = """You are a financial analyst assistant specialising in SEC filings.
You have access to excerpts from 10-K and 10-Q filings for:
  Apple (AAPL), Alphabet/Google (GOOGL), Meta (META), Microsoft (MSFT), NVIDIA (NVDA).

Rules you must follow:
1. Answer based ONLY on the provided excerpts. Do not use prior knowledge about
   these companies — if it is not in the excerpts, it is not in your answer.
2. For every factual claim, cite the source inline using the exact format:
   [COMPANY FILING PERIOD SECTION]
   Example: [NVDA 10-K FY2024 Management Discussion and Analysis]
3. Always include units and time periods with numerical data.
4. When comparing companies, address each company explicitly.
5. If the context does not contain enough information to fully answer:
   — Say so clearly: "The provided context does not contain enough information..."
   — Explain what information would be needed to answer the question.
   — Do not invent, estimate, or infer beyond what is written."""


# ---------------------------------------------------------------------------
# Answer container
# ---------------------------------------------------------------------------

@dataclass
class Answer:
    question    : str
    answer_text : str
    chunks_used : list[RetrievedChunk]
    model       : str
    input_tokens: int
    output_tokens: int
    elapsed_s   : float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        # gpt-4o-mini pricing (May 2025): $0.15/1M input, $0.60/1M output
        return (self.input_tokens * 0.15 + self.output_tokens * 0.60) / 1_000_000

    def unique_sources(self) -> list[str]:
        """Deduplicated list of source labels for the answer footer."""
        seen = []
        for c in self.chunks_used:
            period = (
                f"FY{c.fiscal_year}"
                if c.fiscal_period == "FY"
                else f"FY{c.fiscal_year} {c.fiscal_period}"
            )
            label = f"{c.company} {c.filing_type} {period} — {c.section_name}"
            if label not in seen:
                seen.append(label)
        return seen


# ---------------------------------------------------------------------------
# QA class
# ---------------------------------------------------------------------------

class QA:
    """
    Retrieves context and generates grounded answers with the LLM.

    Parameters
    ----------
    retriever : a loaded Retriever instance. If None, one is created.
    model     : OpenAI model name (default gpt-4o-mini).
    verbose   : if True, prints retrieval timing on each call.
    """

    def __init__(
        self,
        retriever : Optional[Retriever] = None,
        model     : str  = MODEL,
        verbose   : bool = True,
    ) -> None:
        self.model    = model
        self.verbose  = verbose
        self.client   = OpenAI()
        self.retriever = retriever or Retriever(verbose=verbose)

    def ask(
        self,
        question        : str,
        top_k           : int = DEFAULT_TOP_K,
        filters         : Optional[RetrievalFilters] = None,
        max_per_company : Optional[int] = None,
        show_context    : bool = False,
    ) -> Answer:
        """
        Answer a question using retrieved SEC filing excerpts.

        Parameters
        ----------
        question        : plain-English question
        top_k           : number of chunks to retrieve
        filters         : metadata constraints (company, year, section, etc.)
        max_per_company : cap chunks per company (useful for cross-company queries)
        show_context    : if True, the context block is printed before the answer

        Returns
        -------
        Answer dataclass with answer_text, sources, and token/cost metadata.
        """
        t_start = time.time()

        # ── Retrieve ──────────────────────────────────────────────────
        chunks = self.retriever.retrieve(
            question        = question,
            top_k           = top_k,
            filters         = filters,
            max_per_company = max_per_company,
        )

        if not chunks:
            return Answer(
                question     = question,
                answer_text  = "No relevant documents found for this question.",
                chunks_used  = [],
                model        = self.model,
                input_tokens = 0,
                output_tokens= 0,
                elapsed_s    = time.time() - t_start,
            )

        # ── Format context ────────────────────────────────────────────
        context = self.retriever.format_context(chunks)

        if show_context:
            _print_context_block(context, len(chunks))

        # ── Call LLM ──────────────────────────────────────────────────
        user_message = (
            f"Here are the relevant excerpts from SEC filings:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Question: {question}"
        )

        if self.verbose:
            print(f"[LLM] Sending to {self.model} "
                  f"({len(user_message):,} chars context + question)...")

        t_llm = time.time()
        response = self.client.chat.completions.create(
            model      = self.model,
            temperature= TEMPERATURE,
            max_tokens = MAX_TOKENS,
            messages   = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
        llm_elapsed = time.time() - t_llm
        total_elapsed = time.time() - t_start

        if self.verbose:
            usage = response.usage
            print(
                f"[LLM] Done in {llm_elapsed:.1f}s  |  "
                f"tokens: {usage.prompt_tokens} in + {usage.completion_tokens} out  |  "
                f"cost: ~${(usage.prompt_tokens*0.15 + usage.completion_tokens*0.60)/1_000_000:.5f}"
            )

        return Answer(
            question      = question,
            answer_text   = response.choices[0].message.content.strip(),
            chunks_used   = chunks,
            model         = self.model,
            input_tokens  = response.usage.prompt_tokens,
            output_tokens = response.usage.completion_tokens,
            elapsed_s     = total_elapsed,
        )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_context_block(context: str, n_chunks: int) -> None:
    print(f"\n{'─'*60}")
    print(f"  CONTEXT SENT TO LLM  ({n_chunks} chunks)")
    print(f"{'─'*60}")
    # Print with indentation for readability
    for line in context.splitlines():
        print(f"  {line}")
    print(f"{'─'*60}\n")


def print_answer(answer: Answer, show_context: bool = False) -> None:
    """Print a formatted answer to the terminal."""

    # Question header
    print(f"\n{'═'*60}")
    border = '═' * min(58, len(answer.question) + 4)
    print(f"  Q: {answer.question}")
    print(f"{'═'*60}")

    # Answer
    print()
    # Word-wrap the answer at 70 chars for terminal readability
    for para in answer.answer_text.split("\n"):
        if para.strip():
            print(f"  {para}")
        else:
            print()

    # Sources footer
    sources = answer.unique_sources()
    if sources:
        print(f"\n  Sources:")
        for s in sources:
            print(f"    • {s}")

    # Metadata line
    print(
        f"\n  [{answer.model} · "
        f"{answer.input_tokens}+{answer.output_tokens} tokens · "
        f"${answer.cost_usd:.5f} · "
        f"{answer.elapsed_s:.1f}s]"
    )
    print()


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "label"   : "Factual lookup — single company, financial data",
        "desc"    : "Tests precise retrieval of structured numeric data. "
                    "The financial summary chunks were designed for exactly this.",
        "question": "What was NVIDIA's revenue and net income in FY2024, "
                    "and how did margins compare to FY2023?",
        "top_k"   : 4,
        "filters" : RetrievalFilters(company="NVDA", source_type="financial_data"),
    },
    {
        "label"   : "Trend analysis — temporal reasoning across quarters",
        "desc"    : "Tests whether the model can synthesise a trend from multiple "
                    "retrieved periods. Requires ordering and interpreting changes.",
        "question": "How has Apple's gross margin evolved from FY2022 to FY2025? "
                    "Is the trend positive or negative?",
        "top_k"   : 6,
        "filters" : RetrievalFilters(company="AAPL", source_type="financial_data"),
    },
    {
        "label"   : "Comparative — two companies, narrative sections",
        "desc"    : "Tests multi-company retrieval. Both MSFT and GOOGL should "
                    "be in the context. Per-company retrieval ensures equal representation.",
        "question": "How did Microsoft and Google each describe the competitive "
                    "threat from AI in their most recent annual reports?",
        "top_k"   : 6,
        "filters" : RetrievalFilters(
            filing_type = "10-K",
            section_name= "Risk Factors",
        ),
    },
    {
        "label"   : "Cross-company synthesis — all companies, one theme",
        "desc"    : "Tests breadth retrieval with max_per_company diversity. "
                    "The model should synthesise across five companies.",
        "question": "How are these five companies each describing their capital "
                    "expenditure plans for AI infrastructure?",
        "top_k"   : 5,
        "filters" : None,
        "max_per_company": 1,
    },
    {
        "label"   : "Boundary test — question the corpus cannot answer",
        "desc"    : "Tests honest limitation handling. FY2027 data does not exist. "
                    "The model should acknowledge this rather than speculate.",
        "question": "What will NVIDIA's revenue be in FY2027?",
        "top_k"   : 4,
        "filters" : RetrievalFilters(company="NVDA"),
    },
]


def run_test_suite(qa: QA, show_context: bool) -> None:
    print(f"\n{'='*60}")
    print(f"  QA TEST SUITE  ({len(TEST_CASES)} questions)")
    print(f"  Model: {qa.model}   Context display: {'ON' if show_context else 'OFF'}")
    print(f"  Toggle context: re-run with {'--show-context' if not show_context else 'no --show-context flag'}")
    print(f"{'='*60}")

    total_cost    = 0.0
    total_tokens  = 0

    for i, tc in enumerate(TEST_CASES, start=1):
        print(f"\n{'─'*60}")
        print(f"  TEST {i}/5: {tc['label']}")
        print(f"  {tc['desc']}")
        print(f"{'─'*60}")

        answer = qa.ask(
            question        = tc["question"],
            top_k           = tc.get("top_k", DEFAULT_TOP_K),
            filters         = tc.get("filters"),
            max_per_company = tc.get("max_per_company"),
            show_context    = show_context,
        )

        print_answer(answer, show_context)
        total_cost   += answer.cost_usd
        total_tokens += answer.total_tokens

    print(f"\n{'='*60}")
    print(f"  TEST SUITE COMPLETE")
    print(f"  Total tokens: {total_tokens:,}   Total cost: ${total_cost:.5f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(qa: QA, show_context: bool) -> None:
    """
    REPL loop for interactive Q&A.

    Commands:
      /context    — toggle context display on/off
      /filters    — show current active filters
      /clear      — clear all filters
      /company X  — set company filter (e.g. /company NVDA)
      /year N     — set fiscal year filter (e.g. /year 2024)
      /help       — show this command list
      quit / exit — exit interactive mode
    """
    active_filters: Optional[RetrievalFilters] = None
    ctx = show_context

    print(f"\n{'='*60}")
    print(f"  SEC FILINGS Q&A — Interactive Mode")
    print(f"  Model: {qa.model}")
    print(f"  Context display: {'ON' if ctx else 'OFF'}  (type /context to toggle)")
    print(f"  Type /help for commands, 'quit' to exit.")
    print(f"{'='*60}\n")

    while True:
        try:
            raw = input("  Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            break

        if not raw:
            continue

        lower = raw.lower()

        # ── Commands ──────────────────────────────────────────────────
        if lower in ("quit", "exit", "q"):
            print("  Exiting.")
            break

        elif lower == "/context":
            ctx = not ctx
            print(f"  Context display: {'ON' if ctx else 'OFF'}\n")
            continue

        elif lower == "/clear":
            active_filters = None
            print("  Filters cleared.\n")
            continue

        elif lower == "/filters":
            if active_filters:
                print(f"  Active filters: {active_filters.summary()}\n")
            else:
                print("  No active filters (auto-detect from question).\n")
            continue

        elif lower.startswith("/company "):
            ticker = raw.split(None, 1)[1].strip().upper()
            if active_filters is None:
                active_filters = RetrievalFilters()
            active_filters.company = ticker
            print(f"  Company filter set: {ticker}\n")
            continue

        elif lower.startswith("/year "):
            try:
                year = int(raw.split(None, 1)[1].strip())
                if active_filters is None:
                    active_filters = RetrievalFilters()
                active_filters.fiscal_year = year
                print(f"  Year filter set: {year}\n")
            except ValueError:
                print("  Usage: /year 2024\n")
            continue

        elif lower == "/help":
            print(
                "\n  Commands:\n"
                "    /context         toggle context display on/off\n"
                "    /company TICKER  filter by company (NVDA, MSFT, GOOGL, META, AAPL)\n"
                "    /year YYYY       filter by fiscal year\n"
                "    /filters         show active filters\n"
                "    /clear           remove all filters\n"
                "    /help            show this list\n"
                "    quit             exit\n"
            )
            continue

        # ── Question ──────────────────────────────────────────────────
        try:
            answer = qa.ask(
                question     = raw,
                top_k        = DEFAULT_TOP_K,
                filters      = active_filters,
                show_context = ctx,
            )
            print_answer(answer)

        except Exception as e:
            print(f"\n  [ERROR] {e}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SEC filings Q&A with retrieval-augmented generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python qa.py                        # run test suite\n"
            "  python qa.py -c                     # test suite with context\n"
            "  python qa.py -i                     # interactive mode\n"
            "  python qa.py -i -c                  # interactive + context\n"
            "  python qa.py -q 'NVDA margins 2024' # single question\n"
            "  python qa.py -q '...' -c            # single question + context\n"
        ),
    )
    parser.add_argument(
        "--show-context", "-c",
        action="store_true",
        dest="show_context",
        help="Print the retrieved context chunks before each answer.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive Q&A session.",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Ask a single question and exit.",
    )
    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Filter by company ticker (NVDA, MSFT, GOOGL, META, AAPL).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Filter by fiscal year (e.g. 2024).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        dest="top_k",
        help=f"Number of chunks to retrieve (default {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help=f"OpenAI model to use (default {MODEL}).",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY environment variable not set.")
        return

    print(f"\n[QA] Initialising...")
    qa = QA(model=args.model, verbose=True)

    if args.query:
        # Single question from command line
        filters = RetrievalFilters(
            company     = args.company,
            fiscal_year = args.year,
        ) if (args.company or args.year) else None

        answer = qa.ask(
            question     = args.query,
            top_k        = args.top_k,
            filters      = filters,
            show_context = args.show_context,
        )
        print_answer(answer)

    elif args.interactive:
        run_interactive(qa, args.show_context)

    else:
        run_test_suite(qa, args.show_context)


if __name__ == "__main__":
    main()
