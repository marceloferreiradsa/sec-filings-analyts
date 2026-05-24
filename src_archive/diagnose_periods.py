"""
diagnose_periods.py — Investigate why some companies return 3 financial
summary documents instead of the expected 4.

Hypothesis: period bucket key fragmentation. The compound key
(period_end, fiscal_year, fiscal_period) creates ghost buckets when
different metrics produce entries with different fp/fy values for the
same economic period. The top-4 limit then consumes a ghost slot,
reducing summaries from 4 to 3.

This script replicates the period-collection logic of build_financial_summaries
and prints the full periods dict before the limit is applied, so we can see
exactly which buckets exist, what they contain, and why some lack revenue.

Run:
    python diagnose_periods.py
"""

from edgar_api import get_cik, fetch_company_facts, FINANCIAL_CONCEPTS, extract_metric

TICKERS = ["NVDA", "GOOGL"]   # the two companies that returned 3 instead of 4
FORM_TYPE = "10-K"
LIMIT = 4


def collect_periods(ticker: str) -> dict:
    """
    Replicate build_financial_summaries period-collection logic exactly,
    but return the full periods dict before limit/filter is applied.
    """
    cik = get_cik(ticker)
    facts = fetch_company_facts(cik)
    periods: dict[tuple, dict] = {}

    for metric_name, concepts in FINANCIAL_CONCEPTS.items():
        entries = extract_metric(facts, concepts, FORM_TYPE)

        for entry in entries:
            period_end    = entry.get("end", "")
            fiscal_year   = entry.get("fy", "")
            fiscal_period = entry.get("fp", "")
            filed_date    = entry.get("filed", "")
            value         = entry.get("val")

            if not period_end or value is None:
                continue

            key = (period_end, fiscal_year, fiscal_period)

            if key not in periods:
                periods[key] = {
                    "period_end":    period_end,
                    "fiscal_year":   fiscal_year,
                    "fiscal_period": fiscal_period,
                    "filed_date":    filed_date,
                    "metrics":       {},
                }

            existing = periods[key]["metrics"].get(metric_name)
            if existing is None or filed_date >= periods[key]["filed_date"]:
                periods[key]["metrics"][metric_name] = value

    return periods


def report(ticker: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {ticker} — {FORM_TYPE} periods before limit/filter")
    print(f"{'='*70}")

    periods = collect_periods(ticker)

    # Sort by period_end descending — same order as build_financial_summaries
    sorted_periods = sorted(
        periods.values(), key=lambda x: x["period_end"], reverse=True
    )

    print(f"\n  Total period buckets collected: {len(sorted_periods)}")
    print(f"\n  TOP {LIMIT * 2} by period_end (showing double the limit to see the gap):\n")

    all_metrics = list(FINANCIAL_CONCEPTS.keys())

    # Header
    print(f"  {'period_end':<14} {'fy':<6} {'fp':<5} {'revenue':>12}  "
          + "  ".join(f"{m[:8]:>8}" for m in all_metrics))
    print(f"  {'-'*14} {'-'*6} {'-'*5} {'-'*12}  "
          + "  ".join(f"{'-'*8}" for _ in all_metrics))

    for i, data in enumerate(sorted_periods[:LIMIT * 2]):
        m = data["metrics"]
        rev = m.get("revenue")
        flag = "← NO REVENUE" if not rev else ("← LIMIT CUT" if i >= LIMIT else "")

        vals = []
        for metric in all_metrics:
            v = m.get(metric)
            vals.append("Y" if v is not None else "N")

        print(f"  {data['period_end']:<14} "
              f"{str(data['fiscal_year']):<6} "
              f"{str(data['fiscal_period']):<5} "
              f"{'$'+str(round(rev/1e9,1))+'B' if rev else 'MISSING':>12}  "
              + "  ".join(f"{v:>8}" for v in vals)
              + f"  {flag}")

    # Specifically check for same period_end with different keys (the fragmentation)
    from collections import defaultdict
    by_end: dict[str, list] = defaultdict(list)
    for key, data in periods.items():
        by_end[data["period_end"]].append((key, data))

    fragments = {end: entries for end, entries in by_end.items() if len(entries) > 1}
    if fragments:
        print(f"\n  ⚠  FRAGMENTED period_end values (same date, multiple buckets):")
        for end_date in sorted(fragments.keys(), reverse=True)[:8]:
            print(f"\n     period_end = {end_date}")
            for key, data in fragments[end_date]:
                rev = data["metrics"].get("revenue")
                print(f"       key={key}  revenue={'$'+str(round(rev/1e9,1))+'B' if rev else 'MISSING'}  "
                      f"metrics={sorted(data['metrics'].keys())}")
    else:
        print(f"\n  ✓  No fragmentation found — each period_end has exactly one bucket.")

    # Show the actual summaries that would be built
    summaries_built = sum(
        1 for d in sorted_periods[:LIMIT]
        if d["metrics"].get("revenue")
    )
    print(f"\n  Summaries that would be built with limit={LIMIT}: {summaries_built}")


if __name__ == "__main__":
    for ticker in TICKERS:
        try:
            report(ticker)
        except Exception as e:
            print(f"\n[{ticker}] ERROR: {e}")
