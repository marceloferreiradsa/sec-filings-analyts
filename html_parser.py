"""
html_parser.py — Track B: narrative section extraction from HTML filings

SEC filings ship in an SGML submission envelope. sec-edgar-downloader (5.x)
saves this envelope as `full-submission.txt`, a single file embedding every
piece of the filing — the primary 10-K/10-Q HTML, exhibits, XBRL data — as
<DOCUMENT> blocks. This module unwraps the envelope, picks the primary
document, and extracts its narrative sections.

WHY section-aware over semantic chunking for this domain:
  10-K and 10-Q filings have a legally mandated section structure.
  "Item 1A. Risk Factors" always means the same thing across all filers.
  Using these natural boundaries as chunk boundaries is more semantically
  faithful than similarity-based splits, which can split mid-argument.

  Semantic chunking would also be confused by boilerplate — legal disclaimers
  and cover page text are semantically similar to real content but meaningless
  for retrieval. Section-awareness lets us exclude them by design.

SECTIONS WE INDEX (and why we exclude others):
  10-K:
    Item 1  (Business)       — company strategy, products, competitive position
    Item 1A (Risk Factors)   — management's own risk catalogue, tracks over time
    Item 7  (MD&A)           — management narrative, forward-looking language
  Excluded:
    Item 2  (Properties)     — real estate inventory, rarely queried
    Item 3  (Legal)          — litigation list, low signal-to-noise for our use
    Item 8  (Financials)     — handled by Track A (XBRL) with cleaner data
    Item 9A (Controls)       — compliance boilerplate

  10-Q:
    Item 2  (MD&A)           — quarterly management narrative
    Item 1A (Risk Factors)   — quarterly risk updates (often minimal, but meaningful
                               when something changes)
"""

import re
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Section definitions
# ---------------------------------------------------------------------------

SECTIONS_10K = {
    "1":   "Business",
    "1A":  "Risk Factors",
    "1B":  "Unresolved Staff Comments",
    "2":   "Properties",
    "3":   "Legal Proceedings",
    "4":   "Mine Safety Disclosures",
    "5":   "Market for Registrant Equity",
    "7":   "Management Discussion and Analysis",
    "7A":  "Quantitative and Qualitative Disclosures About Market Risk",
    "8":   "Financial Statements",
    "9":   "Changes in and Disagreements with Accountants",
    "9A":  "Controls and Procedures",
    "9B":  "Other Information",
}

SECTIONS_10Q = {
    "1":   "Financial Statements",
    "2":   "Management Discussion and Analysis",
    "3":   "Quantitative and Qualitative Disclosures About Market Risk",
    "4":   "Controls and Procedures",
    "1A":  "Risk Factors",
    "2A":  "Unregistered Sales of Equity Securities",
    "5":   "Other Information",
    "6":   "Exhibits",
}

# Only these sections get indexed — rest are discarded
SECTIONS_TO_INDEX = {
    "10-K": {"1", "1A", "7"},
    "10-Q": {"2", "1A"},
}

# Minimum character length for a real section vs. a table-of-contents entry
MIN_SECTION_LENGTH = 800


# ---------------------------------------------------------------------------
# SGML submission handling
# ---------------------------------------------------------------------------
#
# A `full-submission.txt` looks like:
#
#   <SEC-DOCUMENT>0001045810-24-000029.txt : 20240221
#   <SEC-HEADER>0001045810-24-000029.hdr.sgml : 20240221
#     ...
#     CONFORMED PERIOD OF REPORT:     20240128
#     FILED AS OF DATE:               20240221
#     ...
#   </SEC-HEADER>
#   <DOCUMENT>
#     <TYPE>10-K
#     <SEQUENCE>1
#     <FILENAME>nvda-20240128.htm
#     <TEXT>
#       ...the actual HTML (with inline XBRL) lives here...
#     </TEXT>
#   </DOCUMENT>
#   <DOCUMENT>
#     <TYPE>EX-21.1
#     ...exhibits, certifications, XBRL instance docs, etc...
#   </DOCUMENT>
#   ...
#   </SEC-DOCUMENT>
#
# The primary filing is the first <DOCUMENT> block whose <TYPE> matches the
# form being filed (10-K, 10-Q, or their /A amendments). Its <TEXT>...</TEXT>
# payload is the HTML body we want.
# ---------------------------------------------------------------------------

# Form types we treat as the primary document inside a submission
PRIMARY_FORM_TYPES = {"10-K", "10-Q", "10-K/A", "10-Q/A"}

SGML_DOCUMENT_PATTERN = re.compile(
    r'<DOCUMENT>(.*?)</DOCUMENT>',
    re.DOTALL | re.IGNORECASE,
)
SGML_TYPE_PATTERN = re.compile(r'<TYPE>\s*([^\s<]+)', re.IGNORECASE)
SGML_TEXT_PATTERN = re.compile(r'<TEXT>(.*?)</TEXT>', re.DOTALL | re.IGNORECASE)


def find_primary_document(filing_dir: Path) -> Optional[Path]:
    """
    Identify the primary filing artifact in a downloaded accession directory.

    Modern sec-edgar-downloader saves the complete SGML submission as
    `full-submission.txt`. Older versions or other tooling may produce
    individual .htm files instead — we accept both.

    Preference order:
      1. full-submission.txt (canonical, includes header metadata)
      2. Largest .htm/.html file (legacy fallback)
    """
    submission = filing_dir / "full-submission.txt"
    if submission.exists():
        return submission

    html_candidates = list(filing_dir.glob("*.htm")) + list(filing_dir.glob("*.html"))
    if html_candidates:
        return max(html_candidates, key=lambda f: f.stat().st_size)

    return None


def _to_iso_date(yyyymmdd: str) -> str:
    """Convert SEC's YYYYMMDD date format to ISO YYYY-MM-DD."""
    if len(yyyymmdd) == 8 and yyyymmdd.isdigit():
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
    return ""


def _parse_sgml_header(raw: str) -> dict:
    """
    Extract filing metadata from the <SEC-HEADER> block.

    The header is plain key/value text inside the SGML wrapper. We only
    need two fields downstream: when the period covered by the filing
    ended, and when the filing was submitted to the SEC.
    """
    period_match = re.search(r'CONFORMED PERIOD OF REPORT:\s*(\d{8})', raw)
    filed_match  = re.search(r'FILED AS OF DATE:\s*(\d{8})', raw)

    return {
        "period_end": _to_iso_date(period_match.group(1)) if period_match else "",
        "filed_date": _to_iso_date(filed_match.group(1))  if filed_match  else "",
    }


def _extract_primary_html_from_sgml(raw: str) -> str:
    """
    Pull the primary 10-K / 10-Q HTML body out of an SGML submission.

    Iterates <DOCUMENT> blocks in order; returns the <TEXT> payload of the
    first block whose <TYPE> matches a primary form. Exhibits, certifications,
    and XBRL instance documents come later in the submission and are skipped.
    """
    for doc_match in SGML_DOCUMENT_PATTERN.finditer(raw):
        block = doc_match.group(1)

        type_match = SGML_TYPE_PATTERN.search(block)
        if not type_match:
            continue

        doc_type = type_match.group(1).strip().upper()
        if doc_type not in PRIMARY_FORM_TYPES:
            continue

        text_match = SGML_TEXT_PATTERN.search(block)
        if text_match:
            return text_match.group(1)

    return ""


def _load_primary_html(filepath: Path) -> tuple[str, dict]:
    """
    Load primary HTML content and any available header metadata.

    For full-submission.txt: parse SGML, extract embedded HTML + header fields.
    For raw .htm/.html: return file contents directly, no header metadata.
    """
    raw = filepath.read_text(encoding="utf-8", errors="replace")

    if filepath.name == "full-submission.txt":
        return _extract_primary_html_from_sgml(raw), _parse_sgml_header(raw)

    return raw, {}


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

def load_and_clean(html: str) -> BeautifulSoup:
    """
    Parse and clean HTML content.

    Inline XBRL (iXBRL) is increasingly common in modern filings.
    It wraps financial values in tags like <ix:nonfraction> and adds
    namespace declarations that pollute text extraction.
    We unwrap those tags, keeping their text content.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()

    # Unwrap inline XBRL tags (keep content, remove wrapper)
    ixbrl_pattern = re.compile(r'^ix:', re.IGNORECASE)
    for tag in soup.find_all(ixbrl_pattern):
        tag.unwrap()

    return soup


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

# Matches SEC section headers in various formatting styles:
#   "Item 1A."  "ITEM 1A."  "Item 1A —"  "Item 1A\n"  "ITEM 7."
# Captures the item number (e.g., "1A", "7", "9A")
ITEM_HEADER_PATTERN = re.compile(
    r'^\s*item\s+([\dA-Z]+)\s*[.\-\u2014\u2013]',
    re.IGNORECASE | re.MULTILINE
)


def extract_sections(soup: BeautifulSoup, form_type: str) -> dict[str, dict]:
    """
    Extract text content per section from a parsed HTML filing.

    Strategy:
      1. Get full text from the parsed HTML
      2. Find all "Item X" markers using regex
      3. Extract text between consecutive markers
      4. If an item appears more than once (TOC + real content),
         keep the longer occurrence — that's always the real section

    This heuristic approach is necessary because SEC filers use wildly
    different HTML structures. A robust parser based on HTML tags would
    need to handle hundreds of edge cases across filers and years.
    The text-based approach degrades gracefully: it may occasionally
    include a few lines of the next section's header, but it won't miss
    entire sections due to unexpected tag structures.

    Returns: dict mapping item_number → {text, section_name}
    """
    sections_map = SECTIONS_10K if form_type == "10-K" else SECTIONS_10Q
    target = SECTIONS_TO_INDEX.get(form_type, set())

    full_text = soup.get_text(separator="\n")

    # Find all section boundaries
    boundaries = []
    for match in ITEM_HEADER_PATTERN.finditer(full_text):
        item_num = match.group(1).upper().strip()
        if item_num in sections_map:
            boundaries.append({
                "item":  item_num,
                "start": match.start(),
                "label": match.group(0),
            })

    if not boundaries:
        return {}

    # Extract text between consecutive boundaries
    raw_sections: dict[str, list[str]] = {}

    for i, boundary in enumerate(boundaries):
        item_num = boundary["item"]

        if item_num not in target:
            continue

        start = boundary["start"]
        end = boundaries[i + 1]["start"] if i + 1 < len(boundaries) else len(full_text)

        section_text = full_text[start:end].strip()
        section_text = _clean_text(section_text)

        if len(section_text) < MIN_SECTION_LENGTH:
            continue  # likely a table-of-contents entry

        if item_num not in raw_sections:
            raw_sections[item_num] = []
        raw_sections[item_num].append(section_text)

    # For each item, keep the longest occurrence (= real section, not TOC)
    extracted = {}
    for item_num, candidates in raw_sections.items():
        best = max(candidates, key=len)
        extracted[item_num] = {
            "text":         best,
            "section_name": sections_map[item_num],
        }

    return extracted


def _clean_text(text: str) -> str:
    """
    Normalize extracted text:
    - Collapse runs of blank lines to at most two
    - Collapse runs of spaces
    - Remove page break artifacts (sequences of dots used as leaders in TOCs)
    """
    # Remove TOC leader dots: "Risk Factors ..... 23"
    text = re.sub(r'\.{4,}', '', text)

    # Collapse whitespace
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def parse_filing(
    filing_dir: Path,
    form_type: str,
) -> tuple[dict[str, dict], dict]:
    """
    Parse a single filing directory.

    Returns a 2-tuple:
      sections: dict mapping item_number → {text, section_name}
      metadata: dict with period_end and filed_date (ISO format)

    Either field may be empty if the corresponding source data is missing.
    Returns ({}, {}) if no primary document is found at all.
    """
    primary = find_primary_document(filing_dir)

    if not primary:
        print(f"    Warning: no submission file found in {filing_dir.name}")
        return {}, {}

    html, header_meta = _load_primary_html(primary)

    if not html:
        print(f"    Warning: no primary 10-K/10-Q document inside {filing_dir.name}")
        return {}, header_meta

    soup = load_and_clean(html)
    sections = extract_sections(soup, form_type)

    return sections, header_meta
