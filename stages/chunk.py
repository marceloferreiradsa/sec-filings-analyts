"""
chunk.py — Split LangChain Documents into retrieval-sized chunks

Reads:  data/processed/documents.json   (160 documents from ingest.py)
Writes: data/processed/chunks.json      (chunks ready for embed.py)

DESIGN DECISIONS

  Chunk size: 1500 characters (~375 tokens at the rough 4 chars/token ratio).
  Designed for the local target model BAAI/bge-large-en-v1.5 (512-token
  context window). Also well within OpenAI text-embedding-3-small (8191
  tokens). Using the tighter constraint means the same chunks work for both
  without re-running this step when switching models.

  Overlap: 200 characters (~50 tokens). Ensures that a sentence falling on
  a chunk boundary is represented in both adjacent chunks. Without overlap,
  a query whose answer straddles a boundary retrieves neither chunk reliably.

  Financial summary documents are kept whole regardless of length.
  They average ~400 chars — below the chunk threshold — and their
  tabular structure (Revenue: $X | Gross Margin: Y%) is designed
  to be read as a unit. Splitting would separate a margin from its
  revenue base, making individual lines uninterpretable.

  Narrative documents are split with RecursiveCharacterTextSplitter.
  Separator priority: paragraph → line → sentence → word → character.
  This respects the argumentative structure of regulatory text: a Risk
  Factor argument should not be split mid-paragraph if a paragraph break
  is available nearby.

CHUNK ID FORMAT
  {source_doc_index:05d}_{chunk_index:04d}
  Stable across runs (index is position in the input JSON, not a hash).
  Used for evaluation tracing and debugging.

Usage:
    python chunk.py
"""

import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from langchain_core.documents import Document

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_PATH  = Path("./data/processed/documents.json")
OUTPUT_PATH = Path("./data/processed/chunks.json")

# Target chunk size in characters.
# At ~4 chars/token: 1500 chars ≈ 375 tokens.
# Fits inside a 512-token embedding model with room for query prepending.
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 200

# Documents shorter than this are kept as a single chunk without splitting.
# Financial summary documents (~400 chars) always fall below this.
SPLIT_THRESHOLD = CHUNK_SIZE

# Chunks below this length are flagged as potential noise in the log.
# Very short chunks (section-ending fragments, headers) carry little
# semantic content and may dilute retrieval precision.
MIN_MEANINGFUL_LENGTH = 200


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_documents(path: Path) -> list[Document]:
    print(f"\n[LOAD] Reading {path.resolve()}...")

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"Run ingest.py first to generate it."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in raw
    ]

    print(f"[LOAD] {len(docs)} documents loaded.")
    return docs


def print_load_summary(docs: list[Document]) -> None:
    by_source  = Counter(d.metadata["source_type"] for d in docs)
    by_company = Counter(d.metadata["company"] for d in docs)

    lengths_by_source: dict[str, list[int]] = defaultdict(list)
    for d in docs:
        lengths_by_source[d.metadata["source_type"]].append(len(d.page_content))

    print("\n[LOAD] Breakdown by source type:")
    for source, count in sorted(by_source.items()):
        lengths = lengths_by_source[source]
        print(
            f"  {source:<22} {count:>4} docs  |  "
            f"avg {int(statistics.mean(lengths)):>7,} chars  "
            f"min {min(lengths):>7,}  max {max(lengths):>8,}"
        )

    print("\n[LOAD] Breakdown by company:")
    print("  " + "   ".join(
        f"{co}: {ct}" for co, ct in sorted(by_company.items())
    ))

    print(f"\n[LOAD] Strategy preview:")
    n_financial = by_source.get("financial_data", 0)
    n_narrative = by_source.get("narrative", 0)
    narrative_lengths = lengths_by_source.get("narrative", [])
    large_narrative = sum(1 for l in narrative_lengths if l > SPLIT_THRESHOLD)
    print(
        f"  {n_financial} financial_data docs → kept whole "
        f"(all below {SPLIT_THRESHOLD:,}-char threshold)"
    )
    print(
        f"  {n_narrative} narrative docs → {large_narrative} will be split, "
        f"{n_narrative - large_narrative} kept whole"
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def build_splitter() -> RecursiveCharacterTextSplitter:
    """
    RecursiveCharacterTextSplitter attempts separators in order, using the
    first one that produces chunks at or below chunk_size:
      \n\n  paragraph break   (preferred — preserves full arguments)
      \n    line break
      '. '  sentence boundary
      ' '   word boundary
      ''    character (last resort — never desirable)

    For regulatory/financial prose, paragraph breaks are frequent and
    the splitter almost always finds one within the chunk window.
    """
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )


def chunk_document(
    doc: Document,
    doc_index: int,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """
    Chunk a single document into retrieval-sized pieces.

    Decision logic:
      - financial_data source type → always kept whole
        (tabular structure must not be split)
      - narrative, length <= SPLIT_THRESHOLD → kept whole
        (already small enough; splitting adds no value)
      - narrative, length > SPLIT_THRESHOLD → split

    Each output chunk carries the source document's full metadata
    plus chunk-specific fields: chunk_id, chunk_index, chunk_total,
    source_length.
    """
    source_type    = doc.metadata.get("source_type", "")
    content_length = len(doc.page_content)

    should_split = (
        source_type == "narrative"
        and content_length > SPLIT_THRESHOLD
    )

    if should_split:
        texts = splitter.split_text(doc.page_content)
    else:
        texts = [doc.page_content]

    chunks = []
    for chunk_index, text in enumerate(texts):
        chunks.append(Document(
            page_content=text,
            metadata={
                **doc.metadata,
                "chunk_id":      f"{doc_index:05d}_{chunk_index:04d}",
                "chunk_index":   chunk_index,
                "chunk_total":   len(texts),
                "source_length": content_length,
            },
        ))

    return chunks


def print_document_line(doc: Document, chunks: list[Document]) -> None:
    """Print one summary line per document showing the chunking outcome."""
    meta         = doc.metadata
    tag          = "FIN" if meta.get("source_type") == "financial_data" else "NAR"
    filing_type  = meta.get("filing_type", "")
    fiscal_year  = str(meta.get("fiscal_year", ""))
    fiscal_period = str(meta.get("fiscal_period", ""))
    section_name = meta.get("section_name", "")[:22]
    length       = len(doc.page_content)
    n_chunks     = len(chunks)

    if n_chunks == 1:
        outcome = "kept whole"
    else:
        avg_chunk = int(statistics.mean(len(c.page_content) for c in chunks))
        outcome   = f"→ {n_chunks} chunks  (avg {avg_chunk:,} chars)"

    print(
        f"    [{tag}] {filing_type:<5} {fiscal_year:<5} {fiscal_period:<3}  "
        f"{section_name:<24}  {length:>8,} chars   {outcome}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_chunks(chunks: list[Document]) -> int:
    """
    Verify metadata integrity. Returns the number of validation errors found.
    Every chunk must carry the full metadata schema from the source document
    plus the four chunk-specific fields added here.
    """
    required = [
        "company", "company_name", "filing_type", "section",
        "section_name", "source_type", "fiscal_year", "fiscal_period",
        "period_end", "filed_date",
        "chunk_id", "chunk_index", "chunk_total", "source_length",
    ]

    print(f"\n[VALIDATE] Checking {len(chunks):,} chunks for metadata integrity...")

    errors = 0
    missing_by_field: dict[str, int] = {}

    for field in required:
        missing = sum(1 for c in chunks if field not in c.metadata)
        if missing:
            missing_by_field[field] = missing
            errors += missing

    if missing_by_field:
        print(f"  [ERROR] Missing metadata fields detected:")
        for field, count in missing_by_field.items():
            print(f"    {field:<22} missing from {count} chunks")
    else:
        print(f"  [OK] All {len(required)} required metadata fields "
              f"present on every chunk.")

    # Check for empty content
    empty = sum(1 for c in chunks if not c.page_content.strip())
    if empty:
        print(f"  [ERROR] {empty} chunks have empty content.")
        errors += empty
    else:
        print(f"  [OK] No empty chunks.")

    # Check chunk_id uniqueness
    ids = [c.metadata["chunk_id"] for c in chunks]
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        print(f"  [ERROR] {duplicates} duplicate chunk_id values.")
        errors += duplicates
    else:
        print(f"  [OK] All {len(ids)} chunk_ids are unique.")

    return errors


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_chunk_statistics(chunks: list[Document]) -> None:
    lengths    = [len(c.page_content) for c in chunks]
    by_source  = Counter(c.metadata["source_type"] for c in chunks)
    by_company = Counter(c.metadata["company"] for c in chunks)

    chunks_by_source: dict[str, list[Document]] = defaultdict(list)
    for c in chunks:
        chunks_by_source[c.metadata["source_type"]].append(c)

    print(f"\n{'='*60}")
    print("CHUNK STATISTICS")
    print(f"{'='*60}")

    print(f"\n  Total chunks:        {len(chunks):,}")
    print(f"  Total content chars: {sum(lengths):,}")
    est_tokens = sum(lengths) // 4
    print(f"  Estimated tokens:    ~{est_tokens:,}  (at 4 chars/token)")

    print(f"\n  By source type:")
    for source, count in sorted(by_source.items()):
        src_lengths = [len(c.page_content) for c in chunks_by_source[source]]
        print(
            f"    {source:<22} {count:>5} chunks  "
            f"avg {int(statistics.mean(src_lengths)):,} chars"
        )

    print(f"\n  By company:")
    for company, count in sorted(by_company.items()):
        print(f"    {company:<8} {count:>5} chunks")

    print(f"\n  Chunk size distribution (chars):")
    print(f"    min:    {min(lengths):,}")
    print(f"    max:    {max(lengths):,}")
    print(f"    mean:   {int(statistics.mean(lengths)):,}")
    print(f"    median: {int(statistics.median(lengths)):,}")
    print(f"    stdev:  {int(statistics.stdev(lengths)):,}")

    tiny   = [c for c in chunks if len(c.page_content) <  MIN_MEANINGFUL_LENGTH]
    medium = [c for c in chunks if MIN_MEANINGFUL_LENGTH <= len(c.page_content) < CHUNK_SIZE]
    large  = [c for c in chunks if len(c.page_content) >= CHUNK_SIZE]

    print(f"\n  Size buckets:")
    print(
        f"    < {MIN_MEANINGFUL_LENGTH:,} chars  (may be noise):  "
        f"{len(tiny):>4}  ({100*len(tiny)/len(chunks):.1f}%)"
    )
    print(
        f"    {MIN_MEANINGFUL_LENGTH:,}–{CHUNK_SIZE:,} chars  (target range):  "
        f"{len(medium):>4}  ({100*len(medium)/len(chunks):.1f}%)"
    )
    print(
        f"    ≥ {CHUNK_SIZE:,} chars  (at/over limit): "
        f"{len(large):>4}  ({100*len(large)/len(chunks):.1f}%)"
    )

    if large:
        print(f"\n  [WARNING] {len(large)} chunks at or above the size limit.")
        print(f"  These occur when no separator was found in a {CHUNK_SIZE}-char window")
        print(f"  (e.g., a table row with no spaces). They will embed but may")
        print(f"  exceed some local models' token limits. Sample:")
        for c in large[:3]:
            meta = c.metadata
            print(
                f"    [{meta['company']} {meta['filing_type']} "
                f"{meta.get('section_name','')[:20]}] "
                f"{len(c.page_content)} chars"
            )

    if tiny:
        print(f"\n  [INFO] {len(tiny)} short chunks (< {MIN_MEANINGFUL_LENGTH} chars).")
        print(f"  These are typically section-ending fragments and will embed")
        print(f"  with lower semantic richness. Sample content:")
        for c in tiny[:3]:
            meta = c.metadata
            print(
                f"    [{meta['company']} {meta['filing_type']} "
                f"{meta.get('section_name','')[:20]}]  "
                f"'{c.page_content[:60].strip()}'"
            )


def print_sample_chunk(chunks: list[Document]) -> None:
    """Print one complete narrative chunk to verify content and structure."""
    # Find a mid-document narrative chunk with meaningful content
    candidates = [
        c for c in chunks
        if c.metadata.get("source_type") == "narrative"
        and len(c.page_content) > 600
        and c.metadata.get("chunk_index", 0) > 0
    ]

    if not candidates:
        candidates = chunks

    sample = candidates[len(candidates) // 3]

    print(f"\n{'='*60}")
    print(f"SAMPLE CHUNK — {sample.metadata.get('chunk_id', 'n/a')}")
    print(f"{'='*60}")

    print(f"\n  Metadata:")
    display_fields = [
        "company", "filing_type", "section_name", "source_type",
        "fiscal_year", "fiscal_period", "period_end",
        "chunk_index", "chunk_total", "source_length", "chunk_id",
    ]
    for field in display_fields:
        value = sample.metadata.get(field, "—")
        print(f"    {field:<22} {value}")

    content_len = len(sample.page_content)
    print(f"\n  Content ({content_len:,} chars):")
    print(f"  {'─'*54}")
    preview = sample.page_content[:500]
    for line in preview.splitlines()[:10]:
        print(f"  {line}")
    if content_len > 500:
        print(f"  ... [{content_len - 500} more chars]")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = [
        {"page_content": c.page_content, "metadata": c.metadata}
        for c in chunks
    ]

    path.write_text(
        json.dumps(serialized, indent=2, default=str),
        encoding="utf-8",
    )

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\n[SAVE] {len(chunks):,} chunks written to {path}")
    print(f"       File size: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    print("\n" + "="*60)
    print("CHUNKING PIPELINE")
    print("="*60)
    print(f"\n  Input:      {INPUT_PATH}")
    print(f"  Output:     {OUTPUT_PATH}")
    print(f"  Chunk size: {CHUNK_SIZE:,} chars  (~{CHUNK_SIZE//4} tokens)")
    print(f"  Overlap:    {CHUNK_OVERLAP:,} chars  (~{CHUNK_OVERLAP//4} tokens)")
    print(f"  Splitter:   RecursiveCharacterTextSplitter")
    print(f"  Strategy:   financial_data = whole | narrative = split if > {SPLIT_THRESHOLD:,} chars")

    # ── Load ─────────────────────────────────────────────────────────
    docs = load_documents(INPUT_PATH)
    print_load_summary(docs)

    # ── Chunk ─────────────────────────────────────────────────────────
    splitter = build_splitter()
    all_chunks: list[Document] = []

    docs_by_company: dict[str, list[tuple[int, Document]]] = defaultdict(list)
    for i, doc in enumerate(docs):
        docs_by_company[doc.metadata["company"]].append((i, doc))

    print(f"\n[CHUNK] Processing {len(docs)} source documents...\n")

    for company in sorted(docs_by_company.keys()):
        company_entries = docs_by_company[company]
        company_chunks: list[Document] = []

        print(f"  {company} ({len(company_entries)} source documents):")

        for doc_index, doc in company_entries:
            chunks = chunk_document(doc, doc_index, splitter)
            company_chunks.extend(chunks)
            all_chunks.extend(chunks)
            print_document_line(doc, chunks)

        print(
            f"\n  {company} subtotal: {len(company_entries)} source docs "
            f"→ {len(company_chunks)} chunks\n"
        )

    # ── Validate ──────────────────────────────────────────────────────
    errors = validate_chunks(all_chunks)
    if errors:
        print(f"\n  [ABORT] {errors} validation errors. Fix before embedding.")
        return

    # ── Statistics ────────────────────────────────────────────────────
    print_chunk_statistics(all_chunks)

    # ── Sample ────────────────────────────────────────────────────────
    print_sample_chunk(all_chunks)

    # ── Save ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    save_chunks(all_chunks, OUTPUT_PATH)

    elapsed = time.time() - t_start
    print(f"       Elapsed:    {elapsed:.1f}s")
    print(f"\n  Next step: run embed.py to generate embedding vectors")
    print(f"  Model target: text-embedding-3-small (OpenAI) or BAAI/bge-large-en-v1.5 (local)")


if __name__ == "__main__":
    main()
