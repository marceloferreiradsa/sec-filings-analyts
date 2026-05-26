"""
edgar_api.py — Track A: XBRL structured financial data

Design specification: EDGAR_TRACK_A_DESIGN.md
All behavioral decisions documented there; comments here reference the
relevant section of that document rather than repeating the reasoning.

WHAT THIS MODULE DOES
  Fetches XBRL-tagged financial data from the SEC's companyfacts API,
  assembles one natural-language summary per fiscal period per company,
  and returns those summaries ready to become LangChain Documents.

WHY XBRL OVER HTML TABLE EXTRACTION
  Financial tables in HTML filings are layout-heavy and extraction-lossy.
  XBRL data is already structured: concept name, value, period, form type.
  The numbers come from the same legal disclosure — the XBRL tags are part
  of the filed document, not a derived product.

DATA SOURCE
  https://data.sec.gov/api/xbrl/companyfacts/CIK{10-digit-cik}.json
  No authentication. SEC asks for a User-Agent header identifying who is
  calling (courtesy, not security).
"""

import re
import requests
import time
from datetime import date as _date
from typing import Optional

# SEC requires an identifying User-Agent — not authentication, just courtesy
HEADERS = {
    "User-Agent": "Aina-TI marcelo@ainati.com.br",
    "Accept-Encoding": "gzip, deflate",
}

# SEC public API rate limit: 10 requests/second
REQUEST_DELAY = 0.12


# ---------------------------------------------------------------------------
# XBRL taxonomy configuration
# ---------------------------------------------------------------------------
#
# WHY EACH METRIC HAS MULTIPLE CANDIDATE CONCEPTS
# See Design Spec Section 4. The short version: FASB's ASC 606 standard
# (effective 2018) renamed the canonical revenue tag. Companies that adopted
# the new standard use a different tag than companies that haven't or pre-2018
# filings from companies that later adopted it. Our five companies fall into
# both camps, so each metric needs a fallback list.
#
# TWO STRATEGIES — also Section 4 of the design spec:
#
#   COMBINE  — pool entries from ALL candidate concepts before deduplication.
#              Use when candidates are tag renames of the same concept
#              (e.g., "Revenues" → "RevenueFromContractWithCustomer...").
#              Ensures modern data is found even when legacy tags still exist.
#
#   FIRST-MATCH — use the first candidate that has entries, stop there.
#                 Use when candidates have different definitional scopes
#                 (e.g., cash with vs. without short-term investments).
#                 Keeps the definition consistent within a company's series.

FINANCIAL_CONCEPTS: dict[str, list[str]] = {
    "revenue": [                        # COMBINE strategy
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "gross_profit": [                   # first-match (single concept)
        "GrossProfit",
    ],
    "operating_income": [               # first-match (single concept)
        "OperatingIncomeLoss",
    ],
    "net_income": [                     # COMBINE strategy
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "rd_expense": [                     # first-match
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "capex": [                          # first-match
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
    ],
    "cash": [                           # first-match — candidates differ in scope
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "long_term_debt": [                 # first-match — candidates differ in scope
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
}

# Metrics in this set use the COMBINE strategy (see above)
COMBINE_STRATEGY_METRICS: frozenset[str] = frozenset({"revenue", "net_income"})

# SEC frame values that mark a canonical period entry.
# CY####     = annual (365d ±30)
# CY####Q#   = quarterly (91d ±30)
# CY####Q#I  = balance-sheet instant (point-in-time)
# We treat all three as valid frames — the I suffix distinguishes
# balance-sheet from flow, but both are SEC-curated canonical entries.
VALID_FRAME_RE = re.compile(r"^CY\d{4}(Q[1-4])?I?$")


# ---------------------------------------------------------------------------
# CIK lookup
# ---------------------------------------------------------------------------

def get_cik(ticker: str) -> str:
    """
    Return the zero-padded 10-digit CIK for a ticker symbol.
    CIK (Central Index Key) is the SEC's internal company identifier —
    required for all companyfacts API calls.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()

    for entry in resp.json().values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"CIK not found for ticker: {ticker}")


# ---------------------------------------------------------------------------
# XBRL data fetching
# ---------------------------------------------------------------------------

def fetch_company_facts(cik: str) -> dict:
    """
    Fetch the full XBRL fact set for a company.
    Returns the complete companyfacts JSON: every financial concept ever
    reported by this company across all EDGAR filings.
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return resp.json()


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _entry_span_days(e: dict) -> Optional[int]:
    """
    Number of days between an entry's start and end dates.
    Returns None for balance-sheet instant entries (no start date).
    """
    start_str = e.get("start")
    end_str   = e.get("end")
    if not start_str or not end_str:
        return None
    try:
        return (
            _date.fromisoformat(end_str) - _date.fromisoformat(start_str)
        ).days
    except (ValueError, TypeError):
        return None


def extract_metric(
    facts: dict,
    metric_name: str,
    concept_candidates: list[str],
    form_type: str,
) -> list[dict]:
    """
    Return one XBRL entry per fiscal period for the given metric.

    Three steps (Design Spec Sections 4, 5, 7):

      1. GATHER entries from candidate concepts.
         Combine-strategy metrics pool entries from all candidates.
         First-match metrics stop at the first candidate that has data.
         Both strategies accept form/A amendments alongside the base form.

      2. DEDUPLICATE per period_end.
         The companyfacts API returns the same period's value from every
         filing that referenced it. For each period_end we keep one entry:
           a. If any entry has a valid SEC frame (CY####, CY####Q#, etc.),
              prefer it — the frame marks the SEC's canonical entry for
              that calendar period.
           b. If no frame entries exist, keep the most recently filed —
              which reflects the latest restatement or amendment.

      3. RETURN one entry per period_end.
         Downstream code keys buckets by period_end only. The fy and fp
         fields on the entries are NOT used for bucketing — see Design
         Spec Section 3 for why those fields are unreliable.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    form_variants = {form_type, f"{form_type}/A"}

    # ── Step 1: gather ────────────────────────────────────────────────────
    raw: list[dict] = []

    if metric_name in COMBINE_STRATEGY_METRICS:
        for concept in concept_candidates:
            if concept not in us_gaap:
                continue
            entries = us_gaap[concept].get("units", {}).get("USD", [])
            raw.extend(e for e in entries if e.get("form") in form_variants)
    else:
        for concept in concept_candidates:
            if concept not in us_gaap:
                continue
            entries = us_gaap[concept].get("units", {}).get("USD", [])
            entries = [e for e in entries if e.get("form") in form_variants]
            if entries:
                raw = entries
                break

    if not raw:
        return []

    # ── Step 2: deduplicate per period_end ────────────────────────────────
    per_period: dict[str, dict] = {}

    for e in raw:
        period = e.get("end", "")
        if not period:
            continue

        frame = e.get("frame", "")
        e_has_frame = bool(frame and VALID_FRAME_RE.match(frame))

        existing = per_period.get(period)
        if existing is None:
            per_period[period] = e
            continue

        x_frame = existing.get("frame", "")
        x_has_frame = bool(x_frame and VALID_FRAME_RE.match(x_frame))

        if e_has_frame and not x_has_frame:
            per_period[period] = e                  # frame beats no-frame
        elif e_has_frame == x_has_frame:
            if e.get("filed", "") > existing.get("filed", ""):
                per_period[period] = e              # same frame status: most recent wins

    return list(per_period.values())


# ---------------------------------------------------------------------------
# Summary assembly helpers
# ---------------------------------------------------------------------------

def _infer_fiscal_period(period_end: str) -> str:
    """
    Infer quarter label from the month of the period end date.
    Approximate — works correctly for calendar-year companies (MSFT, META,
    GOOGL). Off by one quarter for non-calendar-year companies (NVDA ends
    January, AAPL ends September). Acknowledged limitation in Design Spec
    Section 7; fiscal_year derived from period_end[:4] is the reliable
    temporal anchor for filtering.
    """
    try:
        month = int(period_end[5:7])
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


def _fmt_billions(val: Optional[float]) -> str:
    """Format a USD value as $X.XXB. Returns 'N/A' for missing data."""
    if val is None:
        return "N/A"
    return f"${val / 1e9:.2f}B"


def _fmt_margin(
    numerator: Optional[float],
    denominator: Optional[float],
) -> str:
    """Format a ratio as XX.X%. Returns 'N/A' for missing or zero denominator."""
    if numerator is None or denominator is None or denominator == 0:
        return "N/A"
    return f"{(numerator / denominator) * 100:.1f}%"


# ---------------------------------------------------------------------------
# Financial summary builder
# ---------------------------------------------------------------------------

def build_financial_summaries(
    ticker: str,
    company_name: str,
    form_type: str = "10-K",
    limit: int = 4,
) -> list[dict]:
    """
    Fetch XBRL data and return natural-language financial summaries,
    one per fiscal period, ready to become LangChain Documents.

    WHY natural language instead of raw numbers:
      Embedding models understand "Revenue grew 126% to $60.9B" far better
      than numeric arrays. The semantic content — growth rate, scale, margin
      profile — is what makes these chunks retrievable by real queries.

    PERIOD BUCKETING (Design Spec Section 3 and 7):
      Keyed by period_end alone. The fy and fp fields on XBRL entries reflect
      the FILING year, not the PERIOD year. For example: NVDA's FY2024 revenue
      entry has fy=2026 because it was last reported as a comparative in the
      FY2026 10-K. Using fy in the key would fragment the same period into
      multiple buckets when different metrics' entries came from different
      filing years. fiscal_year is derived from period_end[:4] instead.

    FILED DATE NOTE:
      filed_date in the output reflects the date of the most recently filed
      entry for this period (which may be a later filing's comparative year
      entry). It is correct within 1-2 years and suitable for approximate
      filtering; period_end and fiscal_year are the reliable temporal anchors.
    """
    print(f"  [{ticker}] Fetching CIK...")
    cik = get_cik(ticker)

    print(f"  [{ticker}] Fetching XBRL facts (CIK: {cik})...")
    facts = fetch_company_facts(cik)

    # ── Collect metrics into period buckets ───────────────────────────────
    # One bucket per period_end. Revenue is processed first (first key in
    # FINANCIAL_CONCEPTS), so it populates period_start for each bucket —
    # flow entries have a start date; balance-sheet entries do not.

    periods: dict[str, dict] = {}

    for metric_name, concepts in FINANCIAL_CONCEPTS.items():
        entries = extract_metric(facts, metric_name, concepts, form_type)

        for entry in entries:
            period_end   = entry.get("end", "")
            period_start = entry.get("start", "")
            filed_date   = entry.get("filed", "")
            value        = entry.get("val")

            if not period_end or value is None:
                continue

            if period_end not in periods:
                periods[period_end] = {
                    "period_end":   period_end,
                    "period_start": period_start,   # "" for balance-sheet instants
                    "filed_date":   filed_date,
                    "metrics":      {},
                }
            else:
                # If this is a flow entry (has start date) and the bucket
                # doesn't have one yet, set it. Revenue always arrives first
                # and sets this; subsequent flow metrics are no-ops here.
                if period_start and not periods[period_end]["period_start"]:
                    periods[period_end]["period_start"] = period_start

            periods[period_end]["metrics"][metric_name] = value

    # ── Sort and build summaries up to limit ─────────────────────────────
    #
    # WHY limit is applied inside the loop, not by slicing the list first:
    #
    # 10-Q filings contain two balance-sheet dates — the current quarter-end
    # AND the most recent fiscal year-end as a mandatory comparative. Both
    # entries carry form=10-Q, so the fiscal year-end creates a period bucket
    # containing only cash/debt and no revenue. If we sliced to [:limit]
    # before checking for revenue, one of those slots would be consumed by
    # this ghost bucket, consistently producing limit-1 valid summaries.
    #
    # Iterating all periods and stopping once we have `limit` valid summaries
    # skips ghost buckets without counting them against the limit.

    sorted_periods = sorted(
        periods.values(),
        key=lambda x: x["period_end"],
        reverse=True,
    )

    summaries: list[dict] = []

    for data in sorted_periods:
        if len(summaries) >= limit:
            break

        m          = data["metrics"]
        rev        = m.get("revenue")
        period_end = data["period_end"]

        if not rev:
            # Ghost bucket: balance-sheet-only period with no income statement
            # data. Skip without consuming a limit slot.
            continue

        # Derive fiscal year and period from period_end date.
        # Do NOT use the fy/fp fields from XBRL entries — see module docstring.
        fiscal_year = int(period_end[:4])
        fiscal_period = (
            "FY"
            if form_type == "10-K"
            else _infer_fiscal_period(period_end)
        )
        period_label = (
            f"FY{fiscal_year}"
            if fiscal_period == "FY"
            else f"FY{fiscal_year} {fiscal_period}"
        )

        # Period range for display
        period_start = data["period_start"]
        period_range = (
            f"{period_start} to {period_end}"
            if period_start
            else f"as of {period_end}"
        )

        # YTD detection for 10-Q (Design Spec Section 6).
        # If the revenue entry covers more than 120 days, the filer tagged
        # year-to-date rather than the standalone quarter. Add a note so
        # the LLM and user understand what the absolute values represent.
        # Margin ratios are still valid because all income line items share
        # the same period span.
        ytd_note = ""
        if form_type == "10-Q" and period_start:
            span = _entry_span_days({"start": period_start, "end": period_end})
            if span and span > 120:
                ytd_note = (
                    "\nNote: Revenue and income values cover a year-to-date "
                    "period, not the standalone quarter.\n"
                    "This reflects how this company tags its 10-Q XBRL data. "
                    "Margin ratios remain valid."
                )

        text = (
            f"{company_name} ({ticker}) — {form_type} Financial Summary\n"
            f"Period: {period_label} | Range: {period_range} | "
            f"Filed: {data['filed_date']}"
            f"{ytd_note}\n"
            f"\n"
            f"Revenue:             {_fmt_billions(rev)}\n"
            f"Gross Profit:        {_fmt_billions(m.get('gross_profit'))}"
            f" (Gross Margin: {_fmt_margin(m.get('gross_profit'), rev)})\n"
            f"Operating Income:    {_fmt_billions(m.get('operating_income'))}"
            f" (Operating Margin: {_fmt_margin(m.get('operating_income'), rev)})\n"
            f"Net Income:          {_fmt_billions(m.get('net_income'))}"
            f" (Net Margin: {_fmt_margin(m.get('net_income'), rev)})\n"
            f"R&D Expense:         {_fmt_billions(m.get('rd_expense'))}"
            f" (R&D as % Revenue: {_fmt_margin(m.get('rd_expense'), rev)})\n"
            f"Capital Expenditure: {_fmt_billions(m.get('capex'))}\n"
            f"Cash & Equivalents:  {_fmt_billions(m.get('cash'))}\n"
            f"Long-term Debt:      {_fmt_billions(m.get('long_term_debt'))}"
        )

        summaries.append({
            "text": text,
            "metadata": {
                "company":       ticker,
                "company_name":  company_name,
                "filing_type":   form_type,
                "section":       "financial_data",
                "section_name":  "Financial Summary",
                "source_type":   "financial_data",
                "fiscal_year":   fiscal_year,       # derived from period_end
                "fiscal_period": fiscal_period,     # derived from form_type / month
                "period_end":    period_end,
                "filed_date":    data["filed_date"],
            },
        })

    print(
        f"  [{ticker}] Built {len(summaries)} financial summary documents "
        f"({form_type})"
    )
    return summaries
