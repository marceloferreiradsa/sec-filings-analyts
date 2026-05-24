"""
diagnose_xbrl.py — Inspect which financial concepts each company actually
tags in their XBRL data, broken down by form type.

Run this once when build_financial_summaries returns 0 documents for a company.
The output shows, for each metric in FINANCIAL_CONCEPTS, which candidate tags
exist in the company's facts and how many entries each has per form type.

If a metric returns "(no candidates found)" for revenue, this script also
scans the full concept namespace for any tag containing "revenue" or "sales"
in its name — that's how we discover tags we should add to FINANCIAL_CONCEPTS.

Usage:
    python diagnose_xbrl.py
"""

from edgar_api import get_cik, fetch_company_facts, FINANCIAL_CONCEPTS

COMPANIES = ["NVDA", "MSFT", "GOOGL", "META", "AAPL"]
FORMS = ["10-K", "10-Q"]


def inspect(ticker: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {ticker}")
    print(f"{'='*70}")

    cik = get_cik(ticker)
    facts = fetch_company_facts(cik)
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    print(f"  Total us-gaap concepts available: {len(us_gaap)}")

    for metric_name, concepts in FINANCIAL_CONCEPTS.items():
        print(f"\n  [{metric_name}]")

        any_found = False
        for concept in concepts:
            if concept not in us_gaap:
                continue
            any_found = True

            entries = us_gaap[concept].get("units", {}).get("USD", [])

            # Count entries by form type
            form_counts: dict[str, int] = {}
            for e in entries:
                f = e.get("form", "<no form>")
                form_counts[f] = form_counts.get(f, 0) + 1

            print(f"    {concept}: {len(entries)} total")
            for form in sorted(form_counts.keys()):
                # Only show forms we care about plus their amendments
                if form == "10-K" or form == "10-Q" or form.endswith("/A"):
                    print(f"      form={form!r}: {form_counts[form]} entries")

            # Show one sample entry per form_type to see the structure
            for form_type in FORMS:
                samples = [e for e in entries if e.get("form") == form_type]
                if samples:
                    # Show most recent by 'end' date
                    s = max(samples, key=lambda e: e.get("end", ""))
                    print(f"      sample (form={form_type}): "
                          f"end={s.get('end')}, "
                          f"start={s.get('start', '(instant)')}, "
                          f"fp={s.get('fp')}, "
                          f"fy={s.get('fy')}, "
                          f"val={s.get('val')}, "
                          f"filed={s.get('filed')}")

        if not any_found:
            print(f"    (no candidate concepts found)")

            # For revenue specifically, scan for related tag names so we
            # know what to add to FINANCIAL_CONCEPTS
            if metric_name == "revenue":
                related = [
                    k for k in us_gaap.keys()
                    if "revenue" in k.lower() or "sales" in k.lower()
                ]
                if related:
                    print(f"    Possible alternative tags found in us-gaap namespace:")
                    for k in sorted(related)[:15]:
                        units = us_gaap[k].get("units", {})
                        usd_count = len(units.get("USD", []))
                        print(f"      {k} ({usd_count} USD entries)")


def main() -> None:
    print("XBRL diagnostic — inspecting which financial concepts each company tags")
    for ticker in COMPANIES:
        try:
            inspect(ticker)
        except Exception as e:
            print(f"\n  [{ticker}] ERROR: {e}")


if __name__ == "__main__":
    main()
