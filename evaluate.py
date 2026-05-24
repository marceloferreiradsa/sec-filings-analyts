"""
evaluate.py — RAG evaluation suite for SEC Filings Analyst

PURPOSE
  Detect regressions after any change to the pipeline. Run before committing
  changes. Compare results against the previous baseline to confirm quality
  did not degrade.

THREE EVALUATION MODES

  Smoke  — structural checks only. Did all queries return answers with citations?
            Run time: ~30 seconds. Cost: ~$0.002. Run on EVERY change.

  Assert — deterministic assertions against known exact answers.
            Run time: ~90 seconds. Cost: ~$0.006. Run on EVERY change.

  eval       — fast custom LLM evaluation. Same metrics as RAGAS. ~1 min, ~$0.01.
  ragas-real — real RAGAS library, per-claim analysis. ~5 min, ~$0.10. Major changes.

Usage:
  python evaluate.py                   # smoke + assert (default)
  python evaluate.py --mode smoke      # structural checks only
  python evaluate.py --mode assert     # known-answer assertions
  python evaluate.py --mode eval       # fast LLM eval (custom)
  python evaluate.py --mode ragas-real  # real RAGAS library (per-claim)
  python evaluate.py --save            # save results to data/eval/results.json
  python evaluate.py --compare         # compare against last saved baseline

WHAT TO DO WITH THE RESULTS
  Before making a change: python evaluate.py --save   (record baseline)
  After making a change:  python evaluate.py --compare (check for regressions)
  Any assertion failure is a regression — investigate before committing.
  Score drops of >0.05 on faithfulness or relevancy after a change warrant investigation.
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from retrieve import RetrievalFilters
from qa import QA, Answer


# ─── Assertion test cases ────────────────────────────────────────────────────
# Each case has:
#   question        — the question to ask
#   filters         — optional metadata filter
#   must_contain    — at least ONE of these strings must appear in the answer
#   must_cite       — this company ticker must appear in the sources
#   must_not_contain— NONE of these strings may appear in the answer
#   description     — what this test is verifying

ASSERTION_CASES = [
    {
        "description": "NVDA FY2024 revenue — exact figure",
        "question"   : "What was NVIDIA's revenue in FY2024?",
        "filters"    : RetrievalFilters(company="NVDA", source_type="financial_data"),
        "must_contain": ["60.92", "60.9B", "$60"],
        "must_cite"  : "NVDA",
        "must_not_contain": [],
    },
    {
        "description": "NVDA FY2024 gross margin — percentage",
        "question"   : "What was NVIDIA's gross margin in FY2024?",
        "filters"    : RetrievalFilters(company="NVDA", source_type="financial_data"),
        "must_contain": ["72.7", "72.7%"],
        "must_cite"  : "NVDA",
        "must_not_contain": [],
    },
    {
        "description": "AAPL FY2022 gross margin — historical figure",
        "question"   : "What was Apple's gross margin in FY2022?",
        "filters"    : RetrievalFilters(company="AAPL", source_type="financial_data"),
        "must_contain": ["43.3", "43.3%"],
        "must_cite"  : "AAPL",
        "must_not_contain": [],
    },
    {
        "description": "MSFT FY2024 revenue — exact figure",
        "question"   : "What was Microsoft's revenue in FY2024?",
        "filters"    : RetrievalFilters(company="MSFT", source_type="financial_data"),
        "must_contain": ["245", "245.1", "245.1B"],
        "must_cite"  : "MSFT",
        "must_not_contain": [],
    },
    {
        "description": "NVDA FY2023 revenue — historical figure",
        "question"   : "What was NVIDIA's revenue in FY2023?",
        "filters"    : RetrievalFilters(company="NVDA", source_type="financial_data"),
        "must_contain": ["26.97", "26.9B", "$26"],
        "must_cite"  : "NVDA",
        "must_not_contain": [],
    },
    {
        "description": "Boundary — future prediction must be refused",
        "question"   : "What will NVIDIA's revenue be in FY2030?",
        "filters"    : RetrievalFilters(company="NVDA"),
        "must_contain": ["does not contain", "not enough information", "cannot", "do not have"],
        "must_cite"  : None,
        "must_not_contain": ["trillion", "billion in FY2030", "will be"],
    },
    {
        "description": "Boundary — no hallucinated company",
        "question"   : "What was Tesla's revenue in FY2024?",
        "filters"    : None,
        "must_contain": ["does not contain", "not", "only", "five", "AAPL", "GOOGL",
                         "META", "MSFT", "NVDA"],
        "must_cite"  : None,
        "must_not_contain": ["Tesla revenue was", "TSLA reported"],
    },
    {
        "description": "Multi-company — both MSFT and GOOGL must be addressed",
        "question"   : "Compare Microsoft and Google cloud revenue growth in their most recent annual reports.",
        "filters"    : RetrievalFilters(filing_type="10-K"),
        "must_contain": ["Microsoft", "Google", "Alphabet"],
        "must_cite"  : None,
        "must_not_contain": [],
        "both_companies": ["MSFT", "GOOGL"],
    },
    {
        "description": "Financial data retrieval — all five companies present",
        "question"   : "What are the latest quarterly revenues for all five companies?",
        "filters"    : RetrievalFilters(source_type="financial_data"),
        "must_contain": ["Apple", "Microsoft", "NVIDIA", "Meta", "Alphabet"],
        "must_cite"  : None,
        "must_not_contain": [],
    },
    {
        "description": "Citation format — must include filing type and period",
        "question"   : "What risks did NVIDIA identify in its FY2024 annual report?",
        "filters"    : RetrievalFilters(company="NVDA", filing_type="10-K"),
        "must_contain": ["NVDA", "10-K", "Risk"],
        "must_cite"  : "NVDA",
        "must_not_contain": [],
    },
]

SMOKE_QUESTIONS = [
    ("Simple factual"    , "What was NVIDIA's revenue in FY2024?",
     RetrievalFilters(company="NVDA", source_type="financial_data")),
    ("Trend analysis"    , "How has Apple's gross margin evolved from FY2022 to FY2025?",
     RetrievalFilters(company="AAPL", source_type="financial_data")),
    ("Comparative"       , "How did Microsoft and Google describe AI competition in their 2024 annual reports?",
     RetrievalFilters(filing_type="10-K", section_name="Risk Factors")),
    ("Cross-company"     , "How are these five companies investing in AI infrastructure?",
     None),
    ("Boundary"          , "What will NVIDIA's revenue be in FY2028?",
     RetrievalFilters(company="NVDA")),
]


# ─── Result containers ───────────────────────────────────────────────────────

@dataclass
class SmokeResult:
    label       : str
    passed      : bool
    answer_len  : int
    n_sources   : int
    elapsed_s   : float
    failure_reason: str = ""

@dataclass
class AssertResult:
    description  : str
    passed       : bool
    failures     : list[str] = field(default_factory=list)
    answer_text  : str = ""
    sources      : list[str] = field(default_factory=list)
    elapsed_s    : float = 0.0
    cost_usd     : float = 0.0

@dataclass
class EvalReport:
    mode         : str
    timestamp    : str
    smoke_results: list[SmokeResult] = field(default_factory=list)
    assert_results: list[AssertResult] = field(default_factory=list)
    total_cost   : float = 0.0
    total_time   : float = 0.0

    @property
    def smoke_passed(self) -> int:
        return sum(1 for r in self.smoke_results if r.passed)

    @property
    def assert_passed(self) -> int:
        return sum(1 for r in self.assert_results if r.passed)

    @property
    def smoke_total(self) -> int:
        return len(self.smoke_results)

    @property
    def assert_total(self) -> int:
        return len(self.assert_results)


# ─── Evaluation logic ────────────────────────────────────────────────────────

def run_smoke(qa: QA) -> list[SmokeResult]:
    results = []
    for label, question, filters in SMOKE_QUESTIONS:
        t = time.time()
        answer = qa.ask(question, top_k=5, filters=filters,
                        max_per_company=2 if not filters else None)
        elapsed = time.time() - t

        failure = ""
        if not answer.answer_text.strip():
            failure = "empty answer"
        elif len(answer.answer_text) < 50:
            failure = f"answer too short ({len(answer.answer_text)} chars)"
        elif not answer.chunks_used:
            failure = "no chunks retrieved"
        elif not answer.unique_sources():
            failure = "no sources cited"

        results.append(SmokeResult(
            label         = label,
            passed        = not failure,
            answer_len    = len(answer.answer_text),
            n_sources     = len(answer.unique_sources()),
            elapsed_s     = elapsed,
            failure_reason= failure,
        ))

    return results


def run_assertions(qa: QA) -> list[AssertResult]:
    results = []

    for case in ASSERTION_CASES:
        t = time.time()
        filters = case.get("filters")
        answer  = qa.ask(
            case["question"],
            top_k           = 6,
            filters         = filters,
            max_per_company = 2 if not filters or not filters.company else None,
        )
        elapsed = time.time() - t

        answer_lower = answer.answer_text.lower()
        sources      = answer.unique_sources()
        failures     = []

        # Check must_contain: at least one match required
        must_contain = case.get("must_contain", [])
        if must_contain:
            if not any(m.lower() in answer_lower for m in must_contain):
                failures.append(
                    f"must_contain: none of {must_contain!r} found in answer"
                )

        # Check must_cite: ticker must appear in sources
        must_cite = case.get("must_cite")
        if must_cite:
            if not any(must_cite in s for s in sources):
                failures.append(f"must_cite: '{must_cite}' not in sources {sources}")

        # Check must_not_contain: none allowed
        for forbidden in case.get("must_not_contain", []):
            if forbidden.lower() in answer_lower:
                failures.append(f"must_not_contain: '{forbidden}' found in answer")

        # Check both_companies: both must appear in sources
        for ticker in case.get("both_companies", []):
            if not any(ticker in s for s in sources):
                failures.append(f"both_companies: '{ticker}' missing from sources")

        results.append(AssertResult(
            description  = case["description"],
            passed       = not failures,
            failures     = failures,
            answer_text  = answer.answer_text[:300],
            sources      = sources,
            elapsed_s    = elapsed,
            cost_usd     = answer.cost_usd,
        ))

    return results


def run_llm_eval(qa: QA) -> None:
    """
    LLM-based evaluation of faithfulness, answer relevancy, and context
    precision — implemented directly with GPT-4o-mini rather than via RAGAS.

    Why custom instead of RAGAS:
      RAGAS 0.4.x has a broken dependency on a removed langchain-community
      module (langchain_community.chat_models.vertexai). RAGAS uses GPT
      internally anyway — this implementation does the same thing explicitly,
      which gives full visibility into what is being measured and removes
      the unstable dependency chain.

    Metrics:
      Faithfulness       — fraction of claims in the answer grounded in context
      Answer Relevancy   — does the answer directly address the question?
      Context Precision  — are the retrieved chunks relevant to the question?

    Targets: Faithfulness ≥0.90 · Answer Relevancy ≥0.85 · Context Precision ≥0.75
    A drop of >0.05 on any metric after a change warrants investigation.

    Cost: ~$0.003 for 5 questions (3 LLM calls per question).
    """
    import json as _json
    from openai import OpenAI as _OpenAI
    client = _OpenAI()

    def _score(prompt: str) -> float:
        """Ask the LLM for a 0-1 score and return it."""
        resp = client.chat.completions.create(
            model       = "gpt-4o-mini",
            temperature = 0,
            max_tokens  = 100,
            messages    = [{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        # Extract first float found in the response
        import re
        nums = re.findall(r"0?\.\d+|1\.0+|[01]", text)
        return float(nums[0]) if nums else 0.5

    FAITH_PROMPT = """You are evaluating whether an AI answer is grounded in the provided source passages.

Source passages:
{context}

Answer:
{answer}

What fraction of factual claims in the answer are directly supported by the source passages?
Reply with a single decimal number between 0 and 1. 1.0 = all claims supported. 0.0 = none supported."""

    RELEV_PROMPT = """You are evaluating whether an AI answer addresses the user's question.

Question: {question}
Answer: {answer}

How completely does the answer address the question?
Reply with a single decimal number between 0 and 1. 1.0 = fully addresses. 0.0 = does not address."""

    PREC_PROMPT = """You are evaluating whether retrieved passages are relevant to a question.

Question: {question}
Retrieved passages:
{context}

What fraction of the retrieved passages are relevant to answering the question?
Reply with a single decimal number between 0 and 1. 1.0 = all relevant. 0.0 = none relevant."""

    print("\n[EVAL] LLM-based evaluation (faithfulness · relevancy · context precision)...")
    print("       Uses GPT-4o-mini as the evaluator. ~3 API calls per question.\n")

    f_scores, ar_scores, cp_scores = [], [], []

    for label, question, filters in SMOKE_QUESTIONS:
        print(f"  Evaluating: {label}...")
        answer = qa.ask(question, top_k=5, filters=filters)
        context_text = "\n\n---\n\n".join(c.content for c in answer.chunks_used)

        f  = _score(FAITH_PROMPT.format(context=context_text, answer=answer.answer_text))
        ar = _score(RELEV_PROMPT.format(question=question, answer=answer.answer_text))
        cp = _score(PREC_PROMPT.format(question=question, context=context_text))

        f_scores.append(f);  ar_scores.append(ar);  cp_scores.append(cp)
        print(f"           faithfulness={f:.2f}  relevancy={ar:.2f}  precision={cp:.2f}")

    f_mean  = sum(f_scores)  / len(f_scores)
    ar_mean = sum(ar_scores) / len(ar_scores)
    cp_mean = sum(cp_scores) / len(cp_scores)

    OK = "✓" ; WARN = "⚠"
    print(f"\n{chr(9552)*54}")
    print("  LLM EVALUATION RESULTS  (evaluator: gpt-4o-mini)")
    print(f"{chr(9552)*54}")
    print(f"  {OK if f_mean  >= 0.90 else WARN}  Faithfulness:      {f_mean:.3f}  (target ≥0.90)")
    print(f"  {OK if ar_mean >= 0.85 else WARN}  Answer Relevancy:  {ar_mean:.3f}  (target ≥0.85)")
    print(f"  {OK if cp_mean >= 0.75 else WARN}  Context Precision: {cp_mean:.3f}  (target ≥0.75)")
    print(f"{chr(9552)*54}")
    print()
    print("  Faithfulness     — are claims in the answer grounded in retrieved context?")
    print("  Answer Relevancy — does the answer directly address the question asked?")
    print("  Context Precision— are retrieved chunks relevant to the question?")
    print()


def run_real_ragas(qa: QA) -> None:
    """
    Real RAGAS evaluation using the ragas 0.4.x library.

    ragas 0.4.x has a broken import of langchain_community.chat_models.vertexai,
    which was removed from langchain-community in 0.3+. The two-line stub below
    satisfies the import without loading VertexAI — since we use OpenAI for
    evaluation, the VertexAI code path is never reached.

    The real RAGAS faithfulness metric extracts individual claims from the answer
    and checks each one against the context separately. This per-claim analysis
    is more rigorous than our custom aggregate evaluator and is why real RAGAS
    takes longer (~5 minutes) and costs more (~$0.05-0.15).

    Usage:     python evaluate.py --mode ragas-real
    Cost:      ~$0.05-0.15  (per-claim claim extraction + checking)
    Run when:  major changes — embedding model, chunk strategy, system prompt
    """
    # ── Stub fix for broken ragas 0.4.x dependency ────────────────────────
    import sys, types
    _BROKEN = "langchain_community.chat_models.vertexai"
    if _BROKEN not in sys.modules:
        _stub = types.ModuleType(_BROKEN)
        _stub.ChatVertexAI = type("ChatVertexAI", (), {})  # empty placeholder class
        sys.modules[_BROKEN] = _stub
        # Register on parent module so attribute lookup also works
        import langchain_community.chat_models as _cm
        if not hasattr(_cm, "vertexai"):
            setattr(_cm, "vertexai", _stub)
    # ── End stub fix ───────────────────────────────────────────────────────

    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        import ragas as _ragas_pkg
        _ragas_version = getattr(_ragas_pkg, "__version__", "unknown")
        print(f"  ragas version: {_ragas_version}")
    except ImportError as exc:
        print(f"\n[RAGAS] Import still failing after stub: {exc}")
        print("[RAGAS] Try: pip install 'ragas>=0.4.0'")
        return
    _ragas_version = _ragas_version  # used in result printing

    print("\n[RAGAS] Building evaluation dataset...")
    samples = []

    for label, question, filters in SMOKE_QUESTIONS:
        print(f"  Querying: {label}...")
        answer = qa.ask(question, top_k=5, filters=filters)
        samples.append(SingleTurnSample(
            user_input         = question,
            response           = answer.answer_text,
            retrieved_contexts = [c.content for c in answer.chunks_used],
        ))

    dataset = EvaluationDataset(samples=samples)

    # Official ragas 0.4.x setup (from docs.ragas.io/en/latest):
    #   AsyncOpenAI + llm_factory for LLM
    #   AsyncOpenAI + embedding_factory for embeddings (needed by ResponseRelevancy)
    from openai import AsyncOpenAI as _AsyncOpenAI
    from ragas.llms import llm_factory as _llm_factory
    _client = _AsyncOpenAI()
    _llm    = _llm_factory("gpt-4o-mini", client=_client)

    # Embeddings — needed by ResponseRelevancy (AnswerRelevancy equivalent)
    _embeddings = None
    try:
        from ragas.embeddings import embedding_factory as _emb_factory
        _embeddings = _emb_factory("text-embedding-3-small", client=_client)
        print("  Embeddings: text-embedding-3-small via embedding_factory")
    except (ImportError, Exception) as _emb_err:
        print(f"  Embeddings unavailable ({_emb_err}) — ResponseRelevancy will be skipped")

    # Build metrics list — each metric uses ragas.metrics.collections per docs
    _metrics = [Faithfulness(llm=_llm)]
    print("  Metric: Faithfulness")

    # ResponseRelevancy = AnswerRelevancy equivalent; needs both llm + embeddings
    if _embeddings is not None:
        try:
            _metrics.append(AnswerRelevancy(llm=_llm, embeddings=_embeddings))
            print("  Metric: AnswerRelevancy (Response Relevancy)")
        except Exception as _e:
            print(f"  AnswerRelevancy skipped: {_e}")

    # LLMContextPrecisionWithoutReference — ragas.metrics (not .collections)
    _cp_col = "llm_context_precision_without_reference"
    try:
        from ragas.metrics import LLMContextPrecisionWithoutReference as _CPWR
        try:
            _metrics.append(_CPWR(llm=_llm))
        except TypeError:
            _metrics.append(_CPWR())
        print("  Metric: LLMContextPrecisionWithoutReference")
    except ImportError:
        _cp_col = None
        print("  LLMContextPrecisionWithoutReference not available — skipping")

    print("\n[RAGAS] Scoring with real RAGAS library...")
    print("        Per-claim faithfulness analysis — this takes 3-8 minutes.")
    result = ragas_evaluate(
        dataset = dataset,
        metrics = _metrics,
    )

    # Extract scores — column names vary by ragas version
    try:
        df       = result.to_pandas()
        f_score  = float(df["faithfulness"].mean())
        ar_score = float(df["answer_relevancy"].mean())

        # Find whichever context precision column was produced
        cp_score = None
        for _col in ["llm_context_precision_without_reference",
                     "context_precision_without_reference",
                     "context_relevance", "context_relevancy"]:
            if _col in df.columns:
                cp_score = float(df[_col].mean())
                cp_label = _col.replace("_", " ").title()
                break
    except Exception as exc:
        print(f"[RAGAS] Could not parse results: {exc}")
        print(result)
        return

    OK = "✓" ; WARN = "⚠"
    sep = chr(9552) * 54
    print(f"\n{sep}")
    print("  RAGAS EVALUATION RESULTS  (ragas library, per-claim)")
    print(sep)
    print(f"  {OK if f_score  >= 0.90 else WARN}  Faithfulness:      {f_score:.3f}  (target ≥0.90)")
    print(f"  {OK if ar_score >= 0.85 else WARN}  Answer Relevancy:  {ar_score:.3f}  (target ≥0.85)")
    if cp_score is not None:
        print(f"  {OK if cp_score >= 0.75 else WARN}  Context Precision: {cp_score:.3f}  (target ≥0.75)  [{cp_label}]")
    else:
        print(f"  ─  Context Precision: not available in ragas {_ragas_version}")
    print(sep)
    print()
    print("  Faithfulness:     per-claim — each factual statement checked individually")
    print("  Answer Relevancy: question regeneration — does answer imply the question?")
    print()
    return result


# ─── Output formatting ───────────────────────────────────────────────────────

def print_smoke_results(results: list[SmokeResult]) -> None:
    print(f"\n{'─'*60}")
    print("  SMOKE TEST RESULTS")
    print(f"{'─'*60}")
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"  {status}  {r.label:<20}  "
              f"len={r.answer_len:>4}  sources={r.n_sources}  {r.elapsed_s:.1f}s")
        if not r.passed:
            print(f"         → {r.failure_reason}")
    passed = sum(1 for r in results if r.passed)
    print(f"\n  {passed}/{len(results)} passed")


def print_assert_results(results: list[AssertResult]) -> None:
    print(f"\n{'─'*60}")
    print("  ASSERTION RESULTS")
    print(f"{'─'*60}")
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"  {status}  {r.description}")
        for f in r.failures:
            print(f"         → {f}")
        if not r.passed:
            print(f"         Answer preview: {r.answer_text[:150].strip()!r}")
    passed = sum(1 for r in results if r.passed)
    print(f"\n  {passed}/{len(results)} passed")


def print_summary(report: EvalReport) -> None:
    all_pass = (report.smoke_passed == report.smoke_total and
                report.assert_passed == report.assert_total)
    banner = "ALL TESTS PASSED" if all_pass else "TESTS FAILED — DO NOT COMMIT"

    print(f"\n{'═'*60}")
    print(f"  {banner}")
    print(f"{'═'*60}")
    if report.smoke_results:
        print(f"  Smoke:    {report.smoke_passed}/{report.smoke_total}")
    if report.assert_results:
        print(f"  Assert:   {report.assert_passed}/{report.assert_total}")
    print(f"  Cost:     ${report.total_cost:.5f}")
    print(f"  Time:     {report.total_time:.1f}s")
    print(f"{'═'*60}\n")


def compare_to_baseline(report: EvalReport, baseline_path: Path) -> None:
    if not baseline_path.exists():
        print(f"\n[COMPARE] No baseline found at {baseline_path}")
        print("[COMPARE] Run with --save first to record a baseline.")
        return

    baseline = json.loads(baseline_path.read_text())
    print(f"\n{'─'*60}")
    print(f"  COMPARISON vs BASELINE ({baseline['timestamp'][:10]})")
    print(f"{'─'*60}")

    b_smoke  = baseline.get("smoke_passed", 0), baseline.get("smoke_total", 0)
    b_assert = baseline.get("assert_passed", 0), baseline.get("assert_total", 0)

    delta_smoke  = report.smoke_passed  - b_smoke[0]
    delta_assert = report.assert_passed - b_assert[0]

    def arrow(delta): return "▲" if delta > 0 else ("▼" if delta < 0 else "─")

    print(f"  Smoke:    {report.smoke_passed}/{report.smoke_total}  "
          f"{arrow(delta_smoke)} ({delta_smoke:+d} vs baseline)")
    print(f"  Assert:   {report.assert_passed}/{report.assert_total}  "
          f"{arrow(delta_assert)} ({delta_assert:+d} vs baseline)")

    if delta_smoke < 0 or delta_assert < 0:
        print("\n  ⚠  REGRESSION DETECTED — investigate before committing")
    elif delta_smoke > 0 or delta_assert > 0:
        print("\n  ✓  IMPROVEMENT over baseline")
    else:
        print("\n  ─  No change from baseline")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the SEC Filings RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Workflow:\n"
            "  Before a change:  python evaluate.py --save\n"
            "  After a change:   python evaluate.py --compare\n"
            "  Fast check:       python evaluate.py --mode eval\n"
            "  Deep check:       python evaluate.py --mode ragas-real"
        ),
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["smoke", "assert", "all", "eval", "ragas", "ragas-real"],
        default="all",
        help="Evaluation mode (default: all = smoke + assert). eval=fast custom LLM eval. ragas=same. ragas-real=actual ragas library.",
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save results as the new baseline for future comparisons",
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare results against the saved baseline",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set.")
        return

    eval_dir = Path("data/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = eval_dir / "baseline.json"

    print("[EVAL] Loading QA system...")
    qa = QA(verbose=False)

    report = EvalReport(
        mode      = args.mode,
        timestamp = datetime.now(timezone.utc).isoformat(),
    )

    t_start = time.time()

    if args.mode in ("smoke", "all"):
        print("\n[EVAL] Running smoke tests...")
        report.smoke_results = run_smoke(qa)
        print_smoke_results(report.smoke_results)

    if args.mode in ("assert", "all"):
        print("\n[EVAL] Running assertion tests...")
        report.assert_results = run_assertions(qa)
        print_assert_results(report.assert_results)

    if args.mode in ("ragas", "eval"):
        run_llm_eval(qa)
        return

    if args.mode == "ragas-real":
        run_real_ragas(qa)
        return

    report.total_time = time.time() - t_start
    report.total_cost = (
        sum(0.0 for _ in report.smoke_results) +   # smoke calls already counted in qa
        sum(r.cost_usd for r in report.assert_results)
    )

    print_summary(report)

    if args.compare:
        compare_to_baseline(report, baseline_path)

    if args.save:
        data = {
            "timestamp"    : report.timestamp,
            "smoke_passed" : report.smoke_passed,
            "smoke_total"  : report.smoke_total,
            "assert_passed": report.assert_passed,
            "assert_total" : report.assert_total,
            "total_cost"   : report.total_cost,
            "total_time"   : report.total_time,
        }
        baseline_path.write_text(json.dumps(data, indent=2))
        print(f"[EVAL] Baseline saved to {baseline_path}")


if __name__ == "__main__":
    main()
