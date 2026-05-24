"""
document_builder.py — Unified LangChain Document assembly

This module is the convergence point for Track A (XBRL financial data)
and Track B (HTML narrative sections). Both tracks produce LangChain
Document objects with a shared metadata schema.

METADATA SCHEMA — every Document carries:
  company         str   ticker symbol             e.g. "NVDA"
  company_name    str   full name                 e.g. "Nvidia"
  filing_type     str   form type                 "10-K" | "10-Q"
  section         str   item number or type       "1A" | "7" | "financial_data"
  section_name    str   human-readable label      "Risk Factors"
  source_type     str   which track produced it   "narrative" | "financial_data"
  fiscal_year     int   fiscal year               2024
  fiscal_period   str   period within year        "FY" | "Q1" | "Q2" | "Q3" | "Q4"
  period_end      str   ISO date of period end    "2024-01-28"
  filed_date      str   ISO date of SEC filing    "2024-02-21"

WHY this schema:
  These fields enable filtered retrieval — the most important RAG capability
  beyond basic similarity search. Examples of queries that require filtering:

    "How did Nvidia describe AI demand risk in 2024?"
    → filter: company=NVDA, section=1A, fiscal_year=2024

    "Compare Microsoft and Google's MD&A narratives in Q2 2024"
    → filter: section=7 or section=2, company in [MSFT, GOOGL],
              fiscal_period=Q2, fiscal_year=2024

    "What were Nvidia's margins across the last 4 quarters?"
    → filter: company=NVDA, source_type=financial_data, filing_type=10-Q

  Without this metadata, every query degrades to brute-force similarity
  search across the entire corpus — slower and less precise.
"""

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from edgar_api import build_financial_summaries
from html_parser import parse_filing


# ---------------------------------------------------------------------------
# Company registry
# ---------------------------------------------------------------------------

COMPANIES = {
    "NVDA":  "Nvidia",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet",
    "META":  "Meta",
    "AAPL":  "Apple",
}


# ---------------------------------------------------------------------------
# Track A: financial data documents
# ---------------------------------------------------------------------------

def build_financial_documents(
    ticker: str,
    company_name: str,
    form_types: list[str] = ("10-K", "10-Q"),
    limit_per_form: int = 4,
) -> list[Document]:
    """
    Fetch XBRL data and return LangChain Documents for financial summaries.
    One Document per fiscal period per form type.
    """
    documents = []

    for form_type in form_types:
        summaries = build_financial_summaries(
            ticker=ticker,
            company_name=company_name,
            form_type=form_type,
            limit=limit_per_form,
        )

        for summary in summaries:
            documents.append(Document(
                page_content=summary["text"],
                metadata=summary["metadata"],
            ))

    return documents


# ---------------------------------------------------------------------------
# Track B: narrative section documents
# ---------------------------------------------------------------------------

def _infer_fiscal_period(period_end: str, fiscal_year: Optional[int]) -> str:
    """
    Infer fiscal period label from period end date.
    For 10-Qs we don't always have explicit FP labels, so we approximate
    from the month of the period end date.

    This is imprecise for companies with non-calendar fiscal years (e.g.
    Apple's fiscal year ends in September). The fiscal_year field in metadata
    should be used for year-level filtering; fiscal_period for intra-year.
    """
    if not period_end:
        return ""

    try:
        month = int(period_end[5:7])
        # Rough mapping: Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec
        if month <= 3:
            return "Q1"
        elif month <= 6:
            return "Q2"
        elif month <= 9:
            return "Q3"
        else:
            return "Q4"
    except (ValueError, IndexError):
        return ""


def build_narrative_documents(
    raw_data_path: Path,
    ticker: str,
    company_name: str,
) -> list[Document]:
    """
    Walk downloaded filing directories for a company and extract
    narrative sections as LangChain Documents.

    One Document per section per filing — e.g., one Document for
    Nvidia's 10-K 2024 Risk Factors, another for its MD&A.
    """
    documents = []

    for form_type in ("10-K", "10-Q"):
        filing_base = raw_data_path / "sec-edgar-filings" / ticker / form_type

        if not filing_base.exists():
            continue

        accession_dirs = sorted(
            [d for d in filing_base.iterdir() if d.is_dir()],
            reverse=True,  # most recent first
        )

        for accession_dir in accession_dirs:
            sections, meta = parse_filing(accession_dir, form_type)

            filed_date = meta.get("filed_date", "")
            period_end = meta.get("period_end", "")

            # Approximate fiscal year from period end date
            fiscal_year = int(period_end[:4]) if period_end else None
            fiscal_period = (
                "FY" if form_type == "10-K"
                else _infer_fiscal_period(period_end, fiscal_year)
            )

            for item_num, section_data in sections.items():
                doc = Document(
                    page_content=section_data["text"],
                    metadata={
                        "company":        ticker,
                        "company_name":   company_name,
                        "filing_type":    form_type,
                        "section":        item_num,
                        "section_name":   section_data["section_name"],
                        "source_type":    "narrative",
                        "fiscal_year":    fiscal_year,
                        "fiscal_period":  fiscal_period,
                        "period_end":     period_end,
                        "filed_date":     filed_date,
                    },
                )
                documents.append(doc)

            if sections:
                print(
                    f"    {form_type} {accession_dir.name[:20]}... "
                    f"→ {len(sections)} sections "
                    f"({', '.join('Item ' + k for k in sections)})"
                )

    return documents


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_all_documents(raw_data_path: Path) -> list[Document]:
    """
    Build the full Document corpus for all companies.
    Runs both Track A (XBRL) and Track B (narrative HTML) for each company.

    Returns a flat list of LangChain Documents ready for chunking.
    """
    all_documents = []

    for ticker, company_name in COMPANIES.items():
        print(f"\n{'='*60}")
        print(f"Processing {company_name} ({ticker})")
        print(f"{'='*60}")

        # Track A — financial summaries from XBRL
        print("\n[Track A] Fetching XBRL financial data...")
        financial_docs = build_financial_documents(
            ticker=ticker,
            company_name=company_name,
            form_types=["10-K", "10-Q"],
            limit_per_form=4,
        )
        all_documents.extend(financial_docs)
        print(f"  Total financial documents: {len(financial_docs)}")

        # Track B — narrative sections from HTML
        print("\n[Track B] Parsing HTML filing sections...")
        narrative_docs = build_narrative_documents(
            raw_data_path=raw_data_path,
            ticker=ticker,
            company_name=company_name,
        )
        all_documents.extend(narrative_docs)
        print(f"  Total narrative documents: {len(narrative_docs)}")

    print(f"\n{'='*60}")
    print(f"TOTAL DOCUMENTS BUILT: {len(all_documents)}")

    # Print breakdown by source type and company
    from collections import Counter
    by_source = Counter(d.metadata["source_type"] for d in all_documents)
    by_company = Counter(d.metadata["company"] for d in all_documents)

    print("\nBy source type:")
    for source, count in by_source.items():
        print(f"  {source}: {count}")

    print("\nBy company:")
    for company, count in sorted(by_company.items()):
        print(f"  {company}: {count}")

    return all_documents
