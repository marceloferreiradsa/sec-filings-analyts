from edgar_api import get_cik, fetch_company_facts
facts = fetch_company_facts(get_cik("MSFT"))
samples = facts["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"]
# Show frame presence for most recent 5 10-K entries
k_entries = sorted([e for e in samples if e.get("form") == "10-K"], key=lambda e: e.get("end",""), reverse=True)[:5]
for e in k_entries: print(e.get("end"), e.get("frame", "(no frame)"), e.get("fp"), e.get("val"))